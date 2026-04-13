"""
In-Context Learning Module
Novel ICL method with cross-attention and adaptive weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class CrossAttentionICL(nn.Module):
    """
    Novel ICL method using cross-attention between query and support features
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Adaptive weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor,
                support_labels: torch.Tensor, num_classes: int):
        """
        Args:
            query_features: (batch_size, feature_dim)
            support_features: (batch_size, num_support, feature_dim)
            support_labels: (batch_size, num_support)
            num_classes: Number of classes
            
        Returns:
            icl_logits: (batch_size, num_classes)
            attention_weights: (batch_size, num_support)
        """
        batch_size = query_features.size(0)
        
        # Expand query features for attention
        query_expanded = query_features.unsqueeze(1)  # (batch_size, 1, feature_dim)
        
        # Cross-attention: query attends to support examples
        attended_features, attention_weights = self.cross_attention(
            query=query_expanded,
            key=support_features,
            value=support_features
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)  # (batch_size, feature_dim)
        attention_weights = attention_weights.squeeze(1)  # (batch_size, num_support)
        
        # Refine features
        refined_features = self.refinement(attended_features)
        combined_features = query_features + refined_features
        
        # Compute class prototypes using attention weights
        icl_logits = []
        
        for i in range(num_classes):
            # Get mask for current class
            class_mask = (support_labels == i).float()  # (batch_size, num_support)
            
            # Weighted average of support features for this class
            weighted_attention = attention_weights * class_mask
            
            # Normalize weights per class
            sum_weights = weighted_attention.sum(dim=1, keepdim=True) + 1e-8
            normalized_weights = weighted_attention / sum_weights
            
            # Compute class prototype
            class_prototype = torch.sum(
                support_features * normalized_weights.unsqueeze(-1),
                dim=1
            )  # (batch_size, feature_dim)
            
            # Compute similarity to prototype
            similarity = F.cosine_similarity(combined_features, class_prototype, dim=-1)
            icl_logits.append(similarity)
        
        icl_logits = torch.stack(icl_logits, dim=1)  # (batch_size, num_classes)
        
        return icl_logits, attention_weights


class PrototypeICL(nn.Module):
    """
    Prototype-based ICL with learnable metric
    """
    
    def __init__(self, feature_dim: int, temperature: float = 0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Learnable metric transformation
        self.metric_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor,
                support_labels: torch.Tensor, num_classes: int):
        """
        Args:
            query_features: (batch_size, feature_dim)
            support_features: (batch_size, num_support, feature_dim)
            support_labels: (batch_size, num_support)
            num_classes: Number of classes
            
        Returns:
            icl_logits: (batch_size, num_classes)
        """
        batch_size = query_features.size(0)
        
        # Transform features to learned metric space
        query_transformed = self.metric_transform(query_features)
        support_transformed = self.metric_transform(support_features.view(-1, self.feature_dim))
        support_transformed = support_transformed.view(batch_size, -1, self.feature_dim)
        
        # Compute class prototypes
        prototypes = []
        for i in range(num_classes):
            # Get support examples for class i
            class_mask = (support_labels == i).unsqueeze(-1).float()
            class_examples = support_transformed * class_mask
            
            # Average to get prototype
            num_examples = class_mask.sum(dim=1, keepdim=True) + 1e-8
            prototype = class_examples.sum(dim=1) / num_examples
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes, dim=1)  # (batch_size, num_classes, feature_dim)
        
        # Compute distances to prototypes
        query_expanded = query_transformed.unsqueeze(1)  # (batch_size, 1, feature_dim)
        distances = torch.cdist(query_expanded, prototypes, p=2).squeeze(1)
        
        # Convert distances to logits
        icl_logits = -distances / self.temperature
        
        return icl_logits


class AdaptiveICLFusion(nn.Module):
    """
    Adaptive fusion of zero-shot and ICL predictions
    """
    
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        
        # Confidence predictor for zero-shot
        self.zero_shot_confidence = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Confidence predictor for ICL
        self.icl_confidence = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion = nn.Linear(num_classes * 2, num_classes)
    
    def forward(self, zero_shot_logits: torch.Tensor, icl_logits: torch.Tensor,
                query_features: torch.Tensor):
        """
        Args:
            zero_shot_logits: (batch_size, num_classes)
            icl_logits: (batch_size, num_classes)
            query_features: (batch_size, feature_dim)
            
        Returns:
            fused_logits: (batch_size, num_classes)
            alpha: (batch_size, 1) - weight for ICL
        """
        # Predict confidences
        conf_zero = self.zero_shot_confidence(query_features)
        conf_icl = self.icl_confidence(query_features)
        
        # Normalize confidences
        total_conf = conf_zero + conf_icl + 1e-8
        alpha_zero = conf_zero / total_conf
        alpha_icl = conf_icl / total_conf
        
        # Weighted fusion
        weighted_zero = zero_shot_logits * alpha_zero
        weighted_icl = icl_logits * alpha_icl
        
        # Concatenate and fuse
        combined = torch.cat([weighted_zero, weighted_icl], dim=1)
        fused_logits = self.fusion(combined)
        
        return fused_logits, alpha_icl


def compute_similarity(query_features: torch.Tensor, support_features: torch.Tensor,
                      metric: str = "cosine") -> torch.Tensor:
    """
    Compute similarity between query and support features
    
    Args:
        query_features: (batch_size, feature_dim)
        support_features: (batch_size, num_support, feature_dim)
        metric: Similarity metric ('cosine', 'euclidean')
        
    Returns:
        similarities: (batch_size, num_support)
    """
    if metric == "cosine":
        # Normalize features
        query_norm = F.normalize(query_features, dim=-1)
        support_norm = F.normalize(support_features, dim=-1)
        
        # Compute cosine similarity
        similarities = torch.bmm(
            query_norm.unsqueeze(1),
            support_norm.transpose(1, 2)
        ).squeeze(1)
        
    elif metric == "euclidean":
        # Compute Euclidean distance
        query_expanded = query_features.unsqueeze(1)
        distances = torch.cdist(query_expanded, support_features, p=2).squeeze(1)
        
        # Convert to similarity (negative distance)
        similarities = -distances
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarities


def weighted_knn_predict(query_features: torch.Tensor, support_features: torch.Tensor,
                        support_labels: torch.Tensor, num_classes: int,
                        temperature: float = 0.07, metric: str = "cosine") -> torch.Tensor:
    """
    Weighted k-NN prediction based on feature similarity
    
    Args:
        query_features: (batch_size, feature_dim)
        support_features: (batch_size, num_support, feature_dim)
        support_labels: (batch_size, num_support)
        num_classes: Number of classes
        temperature: Temperature for softmax
        metric: Similarity metric
        
    Returns:
        logits: (batch_size, num_classes)
    """
    # Compute similarities
    similarities = compute_similarity(query_features, support_features, metric)
    
    # Apply temperature scaling
    similarities = similarities / temperature
    
    # Compute weights using softmax
    weights = F.softmax(similarities, dim=-1)
    
    # Aggregate predictions
    batch_size = query_features.size(0)
    num_support = support_features.size(1)
    
    # One-hot encode support labels
    support_one_hot = F.one_hot(support_labels, num_classes).float()
    
    # Weighted sum of one-hot labels
    logits = torch.bmm(weights.unsqueeze(1), support_one_hot).squeeze(1)
    
    return logits