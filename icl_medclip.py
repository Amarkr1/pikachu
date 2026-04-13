"""
ICL-Enhanced PubMedCLIP (BiomedCLIP)
Strategy: Pure Prototypical Network (ProtoNet) for Few-Shot Learning
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import json

from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS 
from torchvision.transforms import ToPILImage

from config import ExperimentConfig
from dataset import create_dataloaders
from utils import compute_metrics, print_metrics, plot_confusion_matrix, save_results

class ICL_ProtoNet(nn.Module):
    """
    Pure Prototypical Network for Few-Shot Classification.
    Predicts based purely on distance to class prototypes formed from support features.
    """
    
    def __init__(self, model_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 num_classes: int = 2, feature_dim: int = 512):
        super().__init__()
        
        self.num_classes = num_classes
        self.repo_id = model_id
        self.checkpoint_dir = "checkpoints_pubmedclip_protonet"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self._load_openclip_model()
        
        # Freeze backbone
        for param in self.pubmedclip.parameters():
            param.requires_grad = False
        
        self.feature_dim = feature_dim
        
        # Learnable scale (temperature) parameter
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    def _load_openclip_model(self):
        """Downloads files and loads model via OpenCLIP factory."""
        print(f"Loading PubMedCLIP model from {self.repo_id}...")
        
        hf_hub_download(repo_id=self.repo_id, filename="open_clip_pytorch_model.bin", local_dir=self.checkpoint_dir)
        hf_hub_download(repo_id=self.repo_id, filename="open_clip_config.json", local_dir=self.checkpoint_dir)

        with open(os.path.join(self.checkpoint_dir, "open_clip_config.json"), "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        model_name = "biomedclip_protonet" 
        if (not model_name.startswith(HF_HUB_PREFIX) and model_name not in _MODEL_CONFIGS and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg

        self.tokenizer = get_tokenizer(model_name)
        self.pubmedclip, _, self.preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=os.path.join(self.checkpoint_dir, "open_clip_pytorch_model.bin"),
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        print("PubMedCLIP model successfully loaded.")

    def extract_features(self, images: list):
        """Extract PubMedCLIP vision features (used by forward and train)"""
        
        model_device = next(self.pubmedclip.parameters()).device 

        with torch.no_grad():
            images_processed = torch.stack([self.preprocess(img) for img in images]).to(model_device)
            image_outputs = self.pubmedclip.encode_image(images_processed)
            features = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
        
        return features
    
    def forward(self, query_images: list, support_images: list, support_labels: torch.Tensor):
        """
        Forward pass calculates distances to prototypes.
        """
        query_features = self.extract_features(query_images)
        
        support_features_list = []
        for support_batch in support_images:
            support_feats = self.extract_features(support_batch)
            support_features_list.append(support_feats)
        support_features = torch.stack(support_features_list) # [B, N*K, D]
        
        # 1. Flatten features and labels
        flat_support_features = support_features.view(-1, self.feature_dim)
        flat_support_labels = support_labels.long().view(-1)
        
        # 2. Compute Prototypes
        prototypes = []
        for class_idx in range(self.num_classes):
            mask = flat_support_labels == class_idx
            if mask.any():
                prototypes.append(flat_support_features[mask].mean(dim=0))
            else:
                prototypes.append(torch.zeros(self.feature_dim, device=query_features.device))
        
        prototypes = torch.stack(prototypes) # [C, D]
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
        
        logits = (query_features @ prototypes.t()) * self.log_temperature.exp()

        return logits, None

def prepare_icl_batch(batch, num_classes, device, to_pil):
    """Utility to prepare and convert ICL batch data for the model"""
    query_images, query_labels, support_sets = batch
    
    query_images_pil = [to_pil(img) for img in query_images]
    
    support_images_list = []
    support_labels_list = []
    
    for support_set in support_sets:
        support_imgs = []
        support_lbls = []
        for class_idx in range(num_classes):
            for img, label in support_set[class_idx]:
                support_imgs.append(to_pil(img))
                support_lbls.append(label)
        support_images_list.append(support_imgs)
        support_labels_list.append(torch.tensor(support_lbls))
    
    support_labels = torch.stack(support_labels_list).to(device)
    query_labels = query_labels.to(device)
    
    return query_images_pil, query_labels, support_images_list, support_labels


def train(model, train_loader, optimizer, criterion, device, epochs):
    """Training loop for the ProtoNet temperature scale"""
    print(f"\nStarting training for {epochs} epochs...")
    to_pil = ToPILImage()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")
        for batch in pbar:
            optimizer.zero_grad()
            
            query_images_pil, query_labels, support_images_list, support_labels = prepare_icl_batch(
                batch, model.num_classes, device, to_pil
            )
            
            # Forward pass: calculates logits based on distance to support prototypes
            logits, _ = model(query_images_pil, support_images_list, support_labels)
            
            # Loss calculation
            loss = criterion(logits, query_labels.long())
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} finished. Avg Loss: {avg_loss:.4f}")


def evaluate(model, dataloader, device, use_icl=True):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    to_pil = ToPILImage()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # The ProtoNet always requires ICL samples (support set)
            query_images, query_labels, _ = batch
            
            query_images_pil, query_labels_on_device, support_images_list, support_labels = prepare_icl_batch(
                batch, model.num_classes, device, to_pil
            )
            
            # Forward pass
            logits, _ = model(query_images_pil, support_images_list, support_labels)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(query_labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_preds, all_labels, all_probs


def main():
    """Run Pure ProtoNet enhanced PubMedCLIP evaluation"""
    config = ExperimentConfig()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets for ICL...")
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(config, icl_mode=True)
    
    num_classes = len(test_dataset.base_dataset.classes)
    class_names = test_dataset.base_dataset.classes
    print(f"Classes: {class_names}")
    print(f"K-shot: {config.icl.num_support_shots}")
    print(f"Support samples (from folder): {len(train_dataset.base_dataset)}")
    print(f"Test samples (from CSV): {len(test_dataset.base_dataset)}")
    
    print("\nInitializing ICL-enhanced PubMedCLIP ProtoNet...")
    model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    feature_dim = 512 # Set to the actual output dimension

    model = ICL_ProtoNet(
        model_id=model_id,
        num_classes=num_classes,
        feature_dim=feature_dim
    )
    
    # Move the entire model to the device.
    model.to(device)
    
    print(f"Feature dimension: {model.feature_dim}")

    learning_rate = getattr(config.icl, 'learning_rate', 1e-4)
    epochs = getattr(config.icl, 'epochs', 10)
    
    trainable_params = [model.log_temperature] 
    
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "="*60)
    print("ICL-ENHANCED: PubMedCLIP with Pure Prototypical Network")
    print(f"Training temperature scale for {epochs} epochs...")
    print("="*60)
    
    train(model, train_loader, optimizer, criterion, device, epochs)
    
    print("\nRunning inference (Evaluation) on test set...")
    metrics, preds, labels, probs = evaluate(model, test_loader, device, use_icl=True)
    
    print_metrics(metrics, title="ICL-PubMedCLIP ProtoNet Results")
    
    results_dir = os.path.join(config.results_dir, "icl")
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(
        {
            'model': 'PubMedCLIP',
            'type': 'icl',
            'icl_method': 'pure_protonet',
            'k_shot': config.icl.num_support_shots,
            'approach': 'few-shot',
            'config': vars(config.icl),
            'metrics': metrics
        },
        os.path.join(results_dir, 'pubmedclip_protonet_results.json')
    )
    
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(results_dir, 'pubmedclip_protonet_confusion_matrix.png')
    )
    
    print(f"\nResults saved to: {results_dir}")
    
    return metrics


if __name__ == "__main__":
    main()