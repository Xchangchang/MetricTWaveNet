#!/usr/bin/env python3
"""
Loss functions for metric learning and imbalanced classification

Implements triplet loss with various hard mining strategies and utilities
for handling class imbalance in T-phase seismic event classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TripletLoss(nn.Module):
    """
    Triplet loss with hard mining strategies for metric learning
    
    Supports different mining strategies to handle challenging samples
    and class imbalance through weighted loss computation.
    """

    def __init__(self, margin=0.3, class_weights=None, mining_strategy='batch-hard'):
        """
        Initialize triplet loss
        
        Args:
            margin (float): Margin for triplet loss
            class_weights (tensor): Per-class weights for handling imbalance
            mining_strategy (str): Mining strategy - 'batch-hard', 'batch-semi-hard', 'batch-all'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.class_weights = class_weights
        self.mining_strategy = mining_strategy
        
        # Validate mining strategy
        valid_strategies = ['batch-hard', 'batch-semi-hard', 'batch-all']
        if mining_strategy not in valid_strategies:
            raise ValueError(f"Mining strategy must be one of {valid_strategies}")

    def forward(self, embeddings, labels):
        """
        Compute triplet loss
        
        Args:
            embeddings (torch.Tensor): Feature embeddings [batch_size, embedding_dim]
            labels (torch.Tensor): Class labels [batch_size]
            
        Returns:
            torch.Tensor: Computed triplet loss
        """
        batch_size = embeddings.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Compute pairwise distances
        dist_matrix = self._pairwise_distances(embeddings)

        # Create label matching matrix
        labels = labels.view(-1, 1)
        matches = (labels == labels.t()).float()

        # Apply mining strategy
        if self.mining_strategy == 'batch-hard':
            basic_loss = self._batch_hard_triplet_loss(dist_matrix, matches)
        elif self.mining_strategy == 'batch-semi-hard':
            basic_loss = self._batch_semi_hard_triplet_loss(dist_matrix, matches)
        else:  # batch-all
            basic_loss = self._batch_all_triplet_loss(dist_matrix, matches)

        # Apply class weights if provided
        if self.class_weights is not None:
            loss = self._apply_class_weights(basic_loss, labels.view(-1))
        else:
            loss = basic_loss.mean()

        return loss

    def _pairwise_distances(self, embeddings):
        """Compute pairwise Euclidean distances between embeddings"""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # Ensure non-negative
        
        return torch.sqrt(distances + 1e-8)  # Add epsilon for numerical stability

    def _batch_hard_triplet_loss(self, dist_matrix, matches):
        """Batch hard mining: select hardest positive and negative samples"""
        # Get hardest positive distances (furthest positive)
        pos_dist = dist_matrix * matches
        diagonal_mask = torch.eye(dist_matrix.size(0), device=dist_matrix.device)
        pos_dist = pos_dist * (1 - diagonal_mask)  # Remove self-distances
        hardest_pos_dist, _ = pos_dist.max(dim=1)
        
        # Handle samples with no positives
        has_positive = (pos_dist.sum(dim=1) > 0).float()
        hardest_pos_dist = hardest_pos_dist * has_positive
        
        # Get hardest negative distances (closest negative)
        max_dist = dist_matrix.max().item()
        neg_dist = dist_matrix * (1 - matches) + matches * max_dist
        hardest_neg_dist, _ = neg_dist.min(dim=1)
        
        # Compute triplet loss
        basic_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        
        # Only keep losses for samples that have both positives and negatives
        has_negative = ((1 - matches).sum(dim=1) > 0).float()
        valid_triplets = has_positive * has_negative
        
        return basic_loss * valid_triplets

    def _batch_semi_hard_triplet_loss(self, dist_matrix, matches):
        """Batch semi-hard mining: select triplets where d(a,p) < d(a,n) < d(a,p) + margin"""
        batch_size = dist_matrix.size(0)
        loss = torch.tensor(0.0, device=dist_matrix.device)
        valid_triplets = 0

        # Remove diagonal elements
        diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=dist_matrix.device)
        
        for anchor_idx in range(batch_size):
            # Find positive and negative indices for this anchor
            pos_mask = matches[anchor_idx] & (~diagonal_mask[anchor_idx])
            neg_mask = ~matches[anchor_idx]
            
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
                
            # Get distances from anchor to positives and negatives
            anchor_pos_dist = dist_matrix[anchor_idx, pos_indices]
            anchor_neg_dist = dist_matrix[anchor_idx, neg_indices]
            
            # For each positive, find semi-hard negatives
            for pos_dist in anchor_pos_dist:
                # Semi-hard condition: pos_dist < neg_dist < pos_dist + margin
                semi_hard_mask = (anchor_neg_dist > pos_dist) & (anchor_neg_dist < pos_dist + self.margin)
                
                if semi_hard_mask.sum() > 0:
                    # Randomly sample up to 5 semi-hard negatives for efficiency
                    semi_hard_neg_dist = anchor_neg_dist[semi_hard_mask]
                    max_samples = min(5, len(semi_hard_neg_dist))
                    
                    if max_samples > 1:
                        indices = torch.randperm(len(semi_hard_neg_dist))[:max_samples]
                        selected_neg_dist = semi_hard_neg_dist[indices]
                    else:
                        selected_neg_dist = semi_hard_neg_dist
                    
                    # Compute triplet loss for selected negatives
                    triplet_losses = F.relu(pos_dist - selected_neg_dist + self.margin)
                    loss += triplet_losses.sum()
                    valid_triplets += len(selected_neg_dist)

        if valid_triplets > 0:
            return loss / valid_triplets
        else:
            return torch.tensor(0.0, device=dist_matrix.device, requires_grad=True)

    def _batch_all_triplet_loss(self, dist_matrix, matches):
        """Batch all mining: use all valid triplets"""
        batch_size = dist_matrix.size(0)
        
        # Reshape for triplet computation
        anchor_pos_dist = dist_matrix.unsqueeze(2)  # [batch, batch, 1]
        anchor_neg_dist = dist_matrix.unsqueeze(1)  # [batch, 1, batch]
        
        # Compute triplet loss: d(a,p) - d(a,n) + margin
        triplet_loss = anchor_pos_dist - anchor_neg_dist + self.margin
        
        # Create mask for valid triplets
        pos_mask = matches.unsqueeze(2)  # [batch, batch, 1]
        neg_mask = (1 - matches).unsqueeze(1)  # [batch, 1, batch]
        valid_mask = pos_mask * neg_mask
        
        # Remove self-comparisons
        diagonal_mask = torch.eye(batch_size, device=dist_matrix.device).unsqueeze(2)
        valid_mask = valid_mask * (1 - diagonal_mask)
        
        # Apply ReLU and mask
        triplet_loss = F.relu(triplet_loss) * valid_mask
        
        # Compute average loss over valid triplets
        num_valid = valid_mask.sum()
        if num_valid > 0:
            return triplet_loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=dist_matrix.device, requires_grad=True)

    def _apply_class_weights(self, basic_loss, labels):
        """Apply class weights to the loss"""
        # Ensure class weights are on the same device
        if not isinstance(self.class_weights, torch.Tensor):
            weight_tensor = torch.tensor(self.class_weights, 
                                       dtype=torch.float32,
                                       device=basic_loss.device)
        else:
            weight_tensor = self.class_weights.to(basic_loss.device)
        
        # Get weights for each sample
        valid_labels = torch.clamp(labels, 0, len(weight_tensor) - 1)
        sample_weights = weight_tensor[valid_labels]
        
        # Compute weighted average
        weighted_loss = basic_loss * sample_weights
        return weighted_loss.mean()

def get_class_weights(class_counts, beta=0.999):
    """
    Calculate class weights using effective number of samples
    
    This method helps handle class imbalance by assigning higher weights
    to minority classes based on the effective number formula.
    
    Args:
        class_counts (array-like): Number of samples per class
        beta (float): Smoothing parameter (0 < beta < 1)
        
    Returns:
        numpy.ndarray: Normalized class weights
    """
    if not isinstance(class_counts, np.ndarray):
        class_counts = np.array(class_counts)
    
    num_classes = len(class_counts)
    weights = np.zeros(num_classes)
    
    for i in range(num_classes):
        count = class_counts[i]
        if count > 0:
            # Effective number formula: (1 - beta^n) / (1 - beta)
            effective_num = 1.0 - np.power(beta, count)
            weights[i] = (1.0 - beta) / max(effective_num, 1e-8)
        else:
            weights[i] = 0.0
    
    # Normalize weights so they sum to the number of classes
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight * num_classes
    
    return weights

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha (float): Weighting factor for rare class
            gamma (float): Focusing parameter
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        Args:
            inputs (torch.Tensor): Predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        """
        Initialize Label Smoothing Loss
        
        Args:
            num_classes (int): Number of classes
            smoothing (float): Smoothing parameter
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        """
        Compute label smoothing loss
        
        Args:
            inputs (torch.Tensor): Predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        return loss.mean()
