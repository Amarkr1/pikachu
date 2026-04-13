"""
Baseline: DINOv2 (Vision-only)
Strategy: Linear Probe Classification
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import os

from config import ExperimentConfig
from dataset import create_dataloaders
from utils import compute_metrics, print_metrics, plot_confusion_matrix, save_results


class DINOv2LinearProbe(nn.Module):
    """
    DINOv2 backbone with a small trainable classifier (linear probe).
    """
    
    def __init__(self, model_name: str, num_classes: int, feature_dim: int):
        super().__init__()
        
        # Load DINOv2 backbone
        print(f"Loading DINOv2 model: {model_name}")
        self.dinov2 = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Freeze DINOv2 backbone
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # Linear classifier (Probe)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, pixel_values):
        # Ensure DINOv2 remains in eval mode during inference/training of the probe
        with torch.no_grad():
            outputs = self.dinov2(pixel_values)
            
            # Use the pooled output features (last hidden state of the [CLS] token)
            features = outputs.last_hidden_state[:, 0, :]
            
            # Normalize features (optional but good practice)
            features = features / features.norm(dim=-1, keepdim=True)
            
        return self.classifier(features)
    
    def extract_features(self, pixel_values):
        """Helper for feature extraction (not used in this baseline, but useful)"""
        with torch.no_grad():
            outputs = self.dinov2(pixel_values)
            features = outputs.last_hidden_state[:, 0, :]
            features = features / features.norm(dim=-1, keepdim=True)
        return features


def train_probe(model, train_loader, optimizer, criterion, device, epochs):
    """Training loop for the linear probe"""
    print(f"\nStarting training for {epochs} epochs...")
    
    # Ensure the DINOv2 backbone is frozen and the probe is in training mode
    model.dinov2.eval() 
    model.classifier.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training Probe")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            
            # Forward pass: only probe weights are updated
            logits = model(images)
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} finished. Avg Loss: {avg_loss:.4f}")


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            logits = model(images)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_preds, all_labels, all_probs


def main():
    """Run DINOv2 baseline (Linear Probe)"""
    config = ExperimentConfig()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets for Linear Probe...")
    train_loader, test_loader, _, test_dataset = create_dataloaders(config, icl_mode=False)
    
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")
    
    # DINOv2-Base feature dimension
    feature_dim = 768 
    
    print(f"\nInitializing DINOv2 model (Feature Dim: {feature_dim})...")
    model = DINOv2LinearProbe(
        model_name=config.model.dinov2_model,
        num_classes=num_classes,
        feature_dim=feature_dim
    )

    model.to(device)

    learning_rate = getattr(config.model, 'learning_rate_probe', 1e-3)
    epochs = getattr(config.model, 'epochs_probe', 10)
    
    # Only classifier (probe) parameters are trainable
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "="*60)
    print("BASELINE: DINOv2 (Linear Probe)")
    print("="*60)
    train_probe(model, train_loader, optimizer, criterion, device, epochs)

    print("\nRunning inference on test set...")
    metrics, preds, labels, probs = evaluate(model, test_loader, device)
    
    print_metrics(metrics, title="DINOv2 Baseline Results (Linear Probe)")
    
    results_dir = os.path.join(config.results_dir, "baseline")
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(
        {
            'model': 'DINOv2',
            'type': 'baseline',
            'approach': 'linear-probe',
            'config': vars(config.model),
            'metrics': metrics
        },
        os.path.join(results_dir, 'dinov2_baseline_results.json')
    )
    
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(results_dir, 'dinov2_baseline_confusion_matrix.png')
    )
    
    print(f"\nResults saved to: {results_dir}")
    
    return metrics


if __name__ == "__main__":
    main()