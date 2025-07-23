#!/usr/bin/env python3
"""
Neural network models for T-phase seismic event classification

Includes CNN baseline and metric learning models for spectrogram-based
seismic T-phase classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and MaxPooling"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class TFeatureExtractor(nn.Module):
    """Feature extractor for metric learning"""
    
    def __init__(self, embedding_dim=128):
        super(TFeatureExtractor, self).__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)  # Raw embedding (not normalized)
        return embedding

class MetricTWaveNet(nn.Module):
    """MetricTWaveNet: Metric learning model for T-phase classification"""
    
    def __init__(self, embedding_dim=128, num_classes=3):
        super(MetricTWaveNet, self).__init__()
        self.feature_extractor = TFeatureExtractor(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        raw_feat = self.feature_extractor(x)  # Raw features
        logits = self.classifier(raw_feat)   # For cross-entropy loss
        triplet_feat = F.normalize(raw_feat, p=2, dim=1)  # Normalized for triplet loss
        return logits, triplet_feat

class CNN(nn.Module):
    """Simple CNN baseline for T-phase classification"""
    
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_layers(x)
        return logits
