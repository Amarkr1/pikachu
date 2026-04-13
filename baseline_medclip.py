"""
Baseline: PubMedCLIP (BiomedCLIP)
Foundation Model: Microsoft BiomedCLIP (OpenCLIP variant)
Uses text prompts for zero-shot classification
"""

import torch
from tqdm import tqdm
import numpy as np
import os
import json

from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS 

from config import ExperimentConfig
from dataset import create_dataloaders
from utils import compute_metrics, print_metrics, plot_confusion_matrix, save_results


class PubMedCLIPClassifier:
    """PubMedCLIP for zero-shot classification using text prompts (using OpenCLIP loading)"""
    
    def __init__(self, model_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 
                 class_names: list = None, device: str = "cuda"):
        self.device = device
        self.class_names = class_names
        self.repo_id = model_id
        self.checkpoint_dir = "checkpoints_pubmedclip"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load model using OpenCLIP method
        self._load_openclip_model()
        
        # Freeze and set eval mode
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # Define text prompt parameters
        self.context_length = 256 # From the example snippet
        self.text_templates = [
            "A medical image showing {}.",
            "Diagnosis of {}.",
            "This image is a photo of {}.",
            "Indication of {}.",
        ]
        
        # Prepare text embeddings for all classes
        self._prepare_text_embeddings()
    
    def _load_openclip_model(self):
        """Downloads files and loads model via OpenCLIP factory."""
        print(f"Loading PubMedCLIP model from {self.repo_id}...")
        
        # 1. Download model and config files
        hf_hub_download(
            repo_id=self.repo_id,
            filename="open_clip_pytorch_model.bin",
            local_dir=self.checkpoint_dir
        )
        hf_hub_download(
            repo_id=self.repo_id,
            filename="open_clip_config.json",
            local_dir=self.checkpoint_dir
        )
        
        # 2. Load config
        with open(os.path.join(self.checkpoint_dir, "open_clip_config.json"), "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        model_name = "biomedclip_local" # Use a local placeholder name

        # 3. Register model config if not standard
        if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg

        # 4. Load tokenizer, model, and preprocessor
        self.tokenizer = get_tokenizer(model_name)
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=os.path.join(self.checkpoint_dir, "open_clip_pytorch_model.bin"),
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        
        # Move model to device
        self.model.to(self.device)
        print("PubMedCLIP model successfully loaded.")

    
    def _prepare_text_embeddings(self):
        """Precompute text embeddings for all class prompts"""
        print("Preparing text embeddings for classes...")
        
        all_texts = []
        for class_name in self.class_names:
            for template in self.text_templates:
                all_texts.append(template.format(class_name))
        
        # Tokenize texts
        texts = self.tokenizer(all_texts, context_length=self.context_length)
        texts = texts.to(self.device)
        
        with torch.no_grad():
            # Get text features
            text_outputs = self.model.encode_text(texts)
            # Normalize
            text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        
        # Reshape: [num_classes, num_templates, feature_dim]
        num_templates = len(self.text_templates)
        feature_dim = text_features.shape[-1]
        self.text_features = text_features.view(len(self.class_names), num_templates, feature_dim)
        
        # Average across templates for each class
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
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                
                # Apply the model's specific preprocessing to each image in the batch
                images_processed = torch.stack([
                    self.preprocess(to_pil(img)) for img in images
                ]).to(self.device)
                
                # Get image features
                image_outputs = self.model.encode_image(images_processed)
                # Normalize
                image_features = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
                
                # Compute similarity with text features
                text_features_on_device = self.text_features_avg.to(image_features.device)
                
                # Logits = logit_scale * image_features @ text_features.t()
                # We extract logit_scale from the model
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features_on_device.t())
                
                # Apply softmax 
                probs = torch.softmax(logits_per_image, dim=-1)
                
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
    """Run baseline PubMedCLIP evaluation (zero-shot with text prompts)"""
    config = ExperimentConfig()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading datasets...")
    _, test_loader, _, test_dataset = create_dataloaders(config, icl_mode=False)
    
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Test samples (from CSV): {len(test_dataset)}")
    
    print("\nInitializing pretrained PubMedCLIP model...")
    model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    
    pubmedclip_model = PubMedCLIPClassifier(
        model_id=model_id,
        class_names=class_names,
        device=device
    )
    
    print("\n" + "="*60)
    print("BASELINE: PubMedCLIP (Zero-shot with Text)")
    print("="*60)
    print(f"Using {len(pubmedclip_model.text_templates)} text templates per class")
    
    print("\nRunning inference on test set...")
    metrics, preds, labels, probs = pubmedclip_model.evaluate(test_loader)
    
    print_metrics(metrics, title="PubMedCLIP Baseline Results")
    
    results_dir = os.path.join(config.results_dir, "baseline")
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(
        {
            'model': 'PubMedCLIP',
            'type': 'baseline',
            'approach': 'zero-shot-text',
            'num_text_templates': len(pubmedclip_model.text_templates),
            'config': vars(config.model),
            'metrics': metrics
        },
        os.path.join(results_dir, 'pubmedclip_baseline_results.json')
    )
    
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(results_dir, 'pubmedclip_baseline_confusion_matrix.png')
    )
    
    print(f"\nResults saved to: {results_dir}")
    
    return metrics


if __name__ == "__main__":
    main()