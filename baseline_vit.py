"""
Baseline: Vision Transformer (ViT)
Foundation Model: Google ViT
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import numpy as np
import os

from config import ExperimentConfig
from dataset import create_dataloaders
from utils import compute_metrics, print_metrics, plot_confusion_matrix, save_results


class ViTClassifier:
    """Vision Transformer for classification"""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224", 
                 num_classes: int = 2, device: str = "cuda"):
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Load pretrained ViT
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        ).to(device)
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
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
    """Run baseline ViT evaluation (inference only, no training)"""
    # Load configuration
    config = ExperimentConfig()
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    # Baseline only needs test data from CSV (no training/support samples)
    print("Loading datasets...")
    _, test_loader, _, test_dataset = create_dataloaders(config, icl_mode=False)
    
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Test samples (from CSV): {len(test_dataset)}")
    
    # Initialize pretrained model (no training)
    print("\nInitializing pretrained ViT model...")
    vit_model = ViTClassifier(
        model_name=config.model.vit_model,
        num_classes=num_classes,
        device=device
    )
    
    # Inference only
    print("\n" + "="*60)
    print("BASELINE: Vision Transformer (Pretrained - Zero-shot)")
    print("="*60)
    
    # Evaluate on test set
    print("\nRunning inference on test set...")
    metrics, preds, labels, probs = vit_model.evaluate(test_loader)
    
    # Print results
    print_metrics(metrics, title="ViT Baseline Results")
    
    # Save results
    results_dir = os.path.join(config.results_dir, "baseline")
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(
        {
            'model': 'ViT',
            'type': 'baseline',
            'approach': 'zero-shot',
            'config': vars(config.model),
            'metrics': metrics
        },
        os.path.join(results_dir, 'vit_baseline_results.json')
    )
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(results_dir, 'vit_baseline_confusion_matrix.png')
    )
    
    print(f"\nResults saved to: {results_dir}")
    
    return metrics


if __name__ == "__main__":
    main()