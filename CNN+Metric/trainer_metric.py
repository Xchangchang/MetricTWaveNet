#!/usr/bin/env python3
"""
MetricTWaveNet Trainer Class

Contains the training logic for metric learning model with triplet loss.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
from collections import defaultdict

# Set non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from data_loader import get_dataloaders
from model import MetricTWaveNet
from loss import TripletLoss, get_class_weights

class MetricLearningTrainer:
    """MetricTWaveNet trainer for T-phase classification"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create data loaders
        try:
            self.train_loader, self.test_loader = get_dataloaders(
                config['data_root'],
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                class_names=config['class_names']
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create dataloaders: {e}")

        # Create model
        self.model = MetricTWaveNet(
            embedding_dim=config['embedding_dim'],
            num_classes=len(config['class_names'])
        ).to(self.device)

        # Calculate class weights
        num_classes = len(config['class_names'])
        class_counts = np.zeros(num_classes, dtype=np.int64)
        for _, labels in self.train_loader:
            for c in range(num_classes):
                class_counts[c] += (labels == c).sum().item()
        print(f"Class counts: {class_counts}")

        weights = get_class_weights(class_counts, beta=0.999)
        print(f"Class weights: {weights}")

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(self.device)
        )
        self.triplet_loss = TripletLoss(
            margin=config['margin'],
            class_weights=torch.tensor(weights, dtype=torch.float32),
            mining_strategy=config['mining_strategy']
        )

        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Training history
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []

        # Create save directory
        os.makedirs(config['save_dir'], exist_ok=True)

    def compute_and_save_centroids(self):
        """Compute and save embedding centroids for each class"""
        self.model.eval()
        centroids = defaultdict(list)
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.train_loader, desc="Computing centroids"):
                inputs = inputs.to(self.device)
                _, embeddings = self.model(inputs)  # Get embeddings
                embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)  # Normalize
                
                for emb, lbl in zip(embeddings.cpu().numpy(), targets.cpu().numpy()):
                    centroids[lbl].append(emb)

        # Calculate mean for each class and normalize
        centroid_dict = {}
        for label in centroids:
            centroid = np.mean(centroids[label], axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroid_dict[label] = centroid

        # Save centroids
        centroids_path = os.path.join(self.config['save_dir'], 'class_centroids.npy')
        np.save(centroids_path, centroid_dict)
        print(f"Saved centroids to: {centroids_path}")
        return centroid_dict

    def train(self):
        """Main training loop with early stopping based on test loss"""
        best_loss = float('inf')
        patience = 15
        epochs_no_improve = 0
        best_acc = 0.0

        for epoch in range(self.config['epochs']):
            # Train one epoch
            train_loss, train_acc = self._train_epoch(epoch)

            # Evaluate model
            test_loss, test_acc = self._eval_epoch(epoch)

            # Update scheduler
            self.scheduler.step(test_loss)

            # Record training history
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)

            # Save best model based on test loss
            if test_loss < best_loss:
                best_loss = test_loss
                best_acc = test_acc
                self._save_model(os.path.join(self.config['save_dir'], 'best_model.pth'))
                print(f"Saved best model with test loss: {best_loss:.4f}, accuracy: {best_acc:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"Epoch {epoch + 1}/{self.config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            # Save latest model
            self._save_model(os.path.join(self.config['save_dir'], 'latest_model.pth'))

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in test loss")
                break

        # Save training history
        self._save_training_history()
        
        # Post-training tasks
        self._plot_training_process()
        self._final_evaluation()
        self.compute_and_save_centroids()

        return best_acc

    def _train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, embeddings = self.model(inputs)

            # Calculate losses
            ce_loss = self.ce_loss(logits, targets)
            triplet_loss = self.triplet_loss(embeddings, targets)

            # Total loss
            loss = ce_loss + self.config['triplet_weight'] * triplet_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return total_loss / len(self.train_loader), correct / total

    def _eval_epoch(self, epoch):
        """Evaluate one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Epoch {epoch + 1} [Test]")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                logits, embeddings = self.model(inputs)

                # Calculate losses
                ce_loss = self.ce_loss(logits, targets)
                triplet_loss = self.triplet_loss(embeddings, targets)

                # Total loss
                loss = ce_loss + self.config['triplet_weight'] * triplet_loss

                # Statistics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        return total_loss / len(self.test_loader), correct / total

    def _save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
        }, path)

    def _load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def _save_training_history(self):
        """Save training history as numpy arrays"""
        save_dir = self.config['save_dir']
        np.save(os.path.join(save_dir, 'train_losses.npy'), np.array(self.train_losses))
        np.save(os.path.join(save_dir, 'test_losses.npy'), np.array(self.test_losses))
        np.save(os.path.join(save_dir, 'train_accs.npy'), np.array(self.train_accs))
        np.save(os.path.join(save_dir, 'test_accs.npy'), np.array(self.test_accs))

    def _plot_training_process(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Testing Loss')

        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.test_accs, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Testing Accuracy')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config['save_dir'], 'training_process.png')
        plt.savefig(plot_path)
        plt.close()

    def _final_evaluation(self):
        """Final model evaluation with confusion matrix and classification report"""
        # Load best model
        best_model_path = os.path.join(self.config['save_dir'], 'best_model.pth')
        self._load_model(best_model_path)

        # Evaluate on test set
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Final Evaluation"):
                inputs = inputs.to(self.device)
                logits, _ = self.model(inputs)
                _, preds = logits.max(1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Save predictions
        np.save(os.path.join(self.config['save_dir'], 'predictions.npy'), all_preds)
        np.save(os.path.join(self.config['save_dir'], 'targets.npy'), all_targets)

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.config['class_names'],
                    yticklabels=self.config['class_names'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Save confusion matrix
        cm_path = os.path.join(self.config['save_dir'], 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

        # Classification report
        report = classification_report(
            all_targets, all_preds,
            target_names=self.config['class_names'],
            digits=4
        )

        # Save classification report
        report_path = os.path.join(self.config['save_dir'], 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Final evaluation completed. Results saved to: {self.config['save_dir']}")
        print("\nClassification Report:")
        print(report)
