"""
Baseline: SigLIP2
Foundation Model: Google SigLIP2 (Vision-Language Model)
Uses text prompts for zero-shot classification
"""

import torch
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import os

from config import ExperimentConfig
from dataset import create_dataloaders
from utils import compute_metrics, print_metrics, plot_confusion_matrix, save_results


class SigLIPClassifier:
    """SigLIP2 for zero-shot classification using text prompts"""
    
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", 
                 class_names: list = None, device: str = "cuda"):
        self.device = device
        self.class_names = class_names
        
        # Load pretrained SigLIP2
        print(f"Loading SigLIP2 model: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="sdpa"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Create text prompts for each class
        # Using multiple templates for robustness
        self.text_templates = [
            "This is a photo of {}.",
            "A photo of {}.",
            "An image of {}.",
            "This image shows {}.",
            "A medical image of {}.",
        ]
        
        # Prepare text embeddings for all classes
        self._prepare_text_embeddings()
    
    def _prepare_text_embeddings(self):
        """Precompute text embeddings for all class prompts"""
        print("Preparing text embeddings for classes...")
        
        all_texts = []
        for class_name in self.class_names:
            for template in self.text_templates:
                all_texts.append(template.format(class_name))
        
        text_inputs = self.processor(
            text=all_texts, 
            padding="max_length", 
            max_length=64, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            # Get text features
            text_outputs = self.model.get_text_features(**text_inputs)
            # Normalize
            text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        
        # Reshape: [num_classes, num_templates, feature_dim]
        num_templates = len(self.text_templates)
        self.text_features = text_features.view(len(self.class_names), num_templates, -1)
        
        # Average across templates for each class
        # self.text_features_avg is currently on the device assigned by self.model.device
        self.text_features_avg = self.text_features.mean(dim=1)  # [num_classes, feature_dim]
        
        print(f"Text embeddings prepared: {self.text_features_avg.shape}")
    
    def evaluate(self, dataloader):
        """Evaluate model using vision-language similarity"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                pixel_values = images.to(self.model.device)
                image_inputs = {"pixel_values": pixel_values}
                
                # Get image features
                image_outputs = self.model.get_image_features(**image_inputs)
                
                # Normalize
                image_features = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
                
                target_device = image_features.device
                text_features_on_device = self.text_features_avg.to(target_device)
                
                logits_per_image = image_features @ text_features_on_device.t()
                
                # Apply sigmoid (SigLIP uses sigmoid, not softmax)
                probs = torch.sigmoid(logits_per_image)
                
                # Get predictions (highest probability)
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
    """Run baseline SigLIP evaluation (zero-shot with text prompts)"""
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
    print("\nInitializing pretrained SigLIP2 model...")
    siglip_model = SigLIPClassifier(
        model_name=config.model.siglip_model,
        class_names=class_names,
        device=device
    )
    
    # Inference only
    print("\n" + "="*60)
    print("BASELINE: SigLIP2 (Pretrained - Zero-shot with Text)")
    print("="*60)
    print(f"Using {len(siglip_model.text_templates)} text templates per class")
    
    # Evaluate on test set
    print("\nRunning inference on test set...")
    metrics, preds, labels, probs = siglip_model.evaluate(test_loader)
    
    # Print results
    print_metrics(metrics, title="SigLIP2 Baseline Results")
    
    # Save results
    results_dir = os.path.join(config.results_dir, "baseline")
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(
        {
            'model': 'SigLIP2',
            'type': 'baseline',
            'approach': 'zero-shot-text',
            'num_text_templates': len(siglip_model.text_templates),
            'config': vars(config.model),
            'metrics': metrics
        },
        os.path.join(results_dir, 'siglip_baseline_results.json')
    )
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(results_dir, 'siglip_baseline_confusion_matrix.png')
    )
    
    print(f"\nResults saved to: {results_dir}")
    
    return metrics


if __name__ == "__main__":
    main()
