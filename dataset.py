import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import random


class ISICDataset(Dataset):
    """
    ISIC 2019 Dataset with CSV annotations
    Supports multi-class classification
    """
    
    def __init__(self, csv_file: str, image_dir: str, transform=None, 
                 return_path: bool = False, classes_to_use: Optional[List[str]] = None,
                 balance_classes: bool = True, balance_seed: int = 42):
        """
        Args:
            csv_file: Path to CSV file with ground truth (e.g., ISIC_2019_Training_GroundTruth.csv)
            image_dir: Directory containing images (e.g., 'ISIC_2019_Training_Input/')
            transform: Image transformations
            return_path: Whether to return image path along with image and label
            classes_to_use: List of class columns to use (if None, uses all classes)
            balance_classes: If True, automatically balance dataset by min class (DEFAULT: True)
            balance_seed: Random seed for balancing (DEFAULT: 42)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.return_path = return_path
        
        # Read CSV
        self.df = pd.read_csv(csv_file)
        
        # Available classes in ISIC 2019
        self.all_class_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        
        # Use specified classes or all classes
        if classes_to_use is not None:
            self.class_columns = [c for c in classes_to_use if c in self.all_class_columns]
        else:
            self.class_columns = self.all_class_columns
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_columns)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        # For compatibility with old code
        self.classes = self.class_columns
        
        # Filter to only include samples with selected classes
        if classes_to_use is not None:
            # Keep only rows where one of the selected classes is 1
            mask = self.df[self.class_columns].sum(axis=1) > 0
            self.df = self.df[mask].reset_index(drop=True)
        
        # Get samples list
        self.samples = []
        for idx, row in self.df.iterrows():
            image_name = row['image']
            
            # Find which class this image belongs to
            label_values = row[self.class_columns].values
            if label_values.sum() > 0:  # Has at least one label
                label = np.argmax(label_values)
                
                # Try common image extensions
                image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_path = os.path.join(image_dir, image_name + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if image_path and os.path.exists(image_path):
                    self.samples.append((image_path, label))
        
        # Organize samples by class for ICL
        self.class_samples = {i: [] for i in range(len(self.class_columns))}
        for idx, (path, label) in enumerate(self.samples):
            self.class_samples[label].append(idx)
        
        print(f"Loaded {len(self.samples)} images from CSV")
        print(f"Classes: {self.class_columns}")
        print(f"Class distribution: {[len(self.class_samples[i]) for i in range(len(self.class_columns))]}")
        
        # Automatically balance if requested
        if balance_classes:
            print(f"\n🔄 Auto-balancing dataset (balance_classes=True)...")
            self.balance_by_min_class(seed=balance_seed)
    
    def balance_by_min_class(self, seed: int = 42):
        """
        Balance dataset by sampling min(class_counts) samples from each class.
        This ensures equal representation and prevents class imbalance issues.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Find minimum class count
        class_counts = [len(self.class_samples[i]) for i in range(len(self.class_columns))]
        min_count = min(class_counts)
        
        print(f"\n{'='*60}")
        print(f"BALANCING DATASET BY MIN CLASS COUNT")
        print(f"{'='*60}")
        print(f"Original distribution: {class_counts}")
        print(f"Minimum class count: {min_count}")
        print(f"Target: {min_count} samples per class")
        
        # Sample min_count samples from each class
        balanced_samples = []
        balanced_class_samples = {i: [] for i in range(len(self.class_columns))}
        
        for class_idx in range(len(self.class_columns)):
            # Get all indices for this class
            class_indices = self.class_samples[class_idx].copy()
            
            # Sample min_count indices
            if len(class_indices) <= min_count:
                selected_indices = class_indices
            else:
                selected_indices = random.sample(class_indices, min_count)
            
            # Add to balanced samples
            for idx in selected_indices:
                new_idx = len(balanced_samples)
                balanced_samples.append(self.samples[idx])
                balanced_class_samples[class_idx].append(new_idx)
            
            print(f"  Class {self.idx_to_class[class_idx]}: {len(class_indices)} → {len(selected_indices)}")
        
        # Shuffle the balanced samples
        combined = list(zip(balanced_samples, range(len(balanced_samples))))
        random.shuffle(combined)
        balanced_samples, shuffle_order = zip(*combined)
        
        # Rebuild class_samples with new indices
        new_class_samples = {i: [] for i in range(len(self.class_columns))}
        for new_idx, (path, label) in enumerate(balanced_samples):
            new_class_samples[label].append(new_idx)
        
        # Update dataset
        self.samples = list(balanced_samples)
        self.class_samples = new_class_samples
        
        new_counts = [len(self.class_samples[i]) for i in range(len(self.class_columns))]
        print(f"\nBalanced distribution: {new_counts}")
        print(f"Total samples: {len(self.samples)} (was {sum(class_counts)})")
        print(f"{'='*60}\n")
        
        return self
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        return image, label
    
    def get_support_set(self, num_shots: int, exclude_idx: Optional[int] = None, 
                       seed: Optional[int] = None) -> Dict[int, List[Tuple]]:
        """
        Get support set for in-context learning
        
        Args:
            num_shots: Number of examples per class
            exclude_idx: Index to exclude (typically the query image)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping class_idx -> list of (image, label) tuples
        """
        if seed is not None:
            random.seed(seed)
        
        support_set = {}
        
        for class_idx in range(len(self.class_columns)):
            # Get all indices for this class
            class_indices = self.class_samples[class_idx].copy()
            
            # Exclude query index if specified
            if exclude_idx is not None:
                query_label = self.samples[exclude_idx][1]
                if class_idx == query_label and exclude_idx in class_indices:
                    class_indices.remove(exclude_idx)
            
            # Sample num_shots examples
            if len(class_indices) < num_shots:
                print(f"Warning: Class {self.idx_to_class[class_idx]} has only {len(class_indices)} samples, requesting {num_shots}")
                selected_indices = class_indices
            else:
                selected_indices = random.sample(class_indices, num_shots)
            
            # Load selected examples
            support_set[class_idx] = []
            for idx in selected_indices:
                img, label = self[idx]
                support_set[class_idx].append((img, label))
        
        return support_set


class MedicalImageDataset(Dataset):
    """Legacy dataset for folder-based organization (backward compatibility)"""
    
    def __init__(self, root_dir: str, transform=None, return_path: bool = False, 
                 classes_to_use: Optional[List[str]] = None,
                 balance_classes: bool = True, balance_seed: int = 42):
        """
        Args:
            root_dir: Path to dataset directory (e.g., 'data/train/skin_isic')
            transform: Image transformations
            return_path: Whether to return image path along with image and label
            classes_to_use: List of class names to use (if None, uses all classes found)
            balance_classes: If True, automatically balance dataset by min class (DEFAULT: True)
            balance_seed: Random seed for balancing (DEFAULT: 42)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        
        # Get all class folders
        all_classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        
        # Filter classes if classes_to_use is specified
        if classes_to_use is not None:
            self.classes = [c for c in all_classes if c in classes_to_use]
            print(f"Filtering to classes: {self.classes} (from available: {all_classes})")
        else:
            self.classes = all_classes
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        # Organize samples by class for ICL
        self.class_samples = {i: [] for i in range(len(self.classes))}
        for idx, (path, label) in enumerate(self.samples):
            self.class_samples[label].append(idx)
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
        print(f"Class distribution: {[len(self.class_samples[i]) for i in range(len(self.classes))]}")
        
        # Automatically balance if requested
        if balance_classes:
            print(f"\n🔄 Auto-balancing dataset (balance_classes=True)...")
            self.balance_by_min_class(seed=balance_seed)
    
    def balance_by_min_class(self, seed: int = 42):
        """
        Balance dataset by sampling min(class_counts) samples from each class.
        This ensures equal representation and prevents class imbalance issues.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Find minimum class count
        class_counts = [len(self.class_samples[i]) for i in range(len(self.classes))]
        min_count = min(class_counts)
        
        print(f"\n{'='*60}")
        print(f"BALANCING DATASET BY MIN CLASS COUNT")
        print(f"{'='*60}")
        print(f"Original distribution: {class_counts}")
        print(f"Minimum class count: {min_count}")
        print(f"Target: {min_count} samples per class")
        
        # Sample min_count samples from each class
        balanced_samples = []
        balanced_class_samples = {i: [] for i in range(len(self.classes))}
        
        for class_idx in range(len(self.classes)):
            # Get all indices for this class
            class_indices = self.class_samples[class_idx].copy()
            
            # Sample min_count indices
            if len(class_indices) <= min_count:
                selected_indices = class_indices
            else:
                selected_indices = random.sample(class_indices, min_count)
            
            # Add to balanced samples
            for idx in selected_indices:
                new_idx = len(balanced_samples)
                balanced_samples.append(self.samples[idx])
                balanced_class_samples[class_idx].append(new_idx)
            
            print(f"  Class {self.idx_to_class[class_idx]}: {len(class_indices)} → {len(selected_indices)}")
        
        # Shuffle the balanced samples
        combined = list(zip(balanced_samples, range(len(balanced_samples))))
        random.shuffle(combined)
        balanced_samples, shuffle_order = zip(*combined)
        
        # Rebuild class_samples with new indices
        new_class_samples = {i: [] for i in range(len(self.classes))}
        for new_idx, (path, label) in enumerate(balanced_samples):
            new_class_samples[label].append(new_idx)
        
        # Update dataset
        self.samples = list(balanced_samples)
        self.class_samples = new_class_samples
        
        new_counts = [len(self.class_samples[i]) for i in range(len(self.classes))]
        print(f"\nBalanced distribution: {new_counts}")
        print(f"Total samples: {len(self.samples)} (was {sum(class_counts)})")
        print(f"{'='*60}\n")
        
        return self
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        return image, label
    
    def get_support_set(self, num_shots: int, exclude_idx: Optional[int] = None, 
                       seed: Optional[int] = None) -> Dict[int, List[Tuple]]:
        """Get support set for in-context learning"""
        if seed is not None:
            random.seed(seed)
        
        support_set = {}
        
        for class_idx in range(len(self.classes)):
            class_indices = self.class_samples[class_idx].copy()
            
            if exclude_idx is not None:
                query_label = self.samples[exclude_idx][1]
                if class_idx == query_label and exclude_idx in class_indices:
                    class_indices.remove(exclude_idx)
            
            if len(class_indices) < num_shots:
                print(f"Warning: Class {self.idx_to_class[class_idx]} has only {len(class_indices)} samples, requesting {num_shots}")
                selected_indices = class_indices
            else:
                selected_indices = random.sample(class_indices, num_shots)
            
            support_set[class_idx] = []
            for idx in selected_indices:
                img, label = self[idx]
                support_set[class_idx].append((img, label))
        
        return support_set


class ICLDataset(Dataset):
    """Dataset wrapper for In-Context Learning experiments"""
    
    def __init__(self, base_dataset, num_support_shots: int = 5):
        """
        Args:
            base_dataset: Base medical image dataset (ISICDataset or MedicalImageDataset)
            num_support_shots: Number of support examples per class (K-shot)
        """
        self.base_dataset = base_dataset
        self.num_support_shots = num_support_shots
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            query_img: Query image
            query_label: Query label
            support_set: Dictionary of support examples {class_idx: [(img, label), ...]}
        """
        # Get query sample
        query_img, query_label = self.base_dataset[idx]
        
        # Get support set (excluding the query image)
        support_set = self.base_dataset.get_support_set(
            num_shots=self.num_support_shots,
            exclude_idx=idx,
            seed=idx  # Use idx as seed for reproducibility
        )
        
        return query_img, query_label, support_set


def get_transforms(image_size: int = 224, augment: bool = False):
    """Get image transforms for preprocessing"""
    
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(config, icl_mode: bool = False):
    """
    Create train and test dataloaders
    
    Supports four modes:
    1. HYBRID (DEFAULT for ISIC): Folder train + CSV test
    2. FOLDER-BASED (for OCT/CXR): Folder train + Folder test
    3. CSV-based: CSV for both train and test with split
    4. Pure folder-based: Folders for both train and test
    """
    
    # Create transforms
    train_transform = get_transforms(config.data.image_size, augment=True)
    test_transform = get_transforms(config.data.image_size, augment=False)
    
    # Check what data sources are available
    train_dir_path = os.path.join(config.data.data_root, config.data.train_dir)
    test_dir_path = os.path.join(config.data.data_root, config.data.test_dir) if config.data.test_dir else None
    has_train_folder = os.path.exists(train_dir_path)
    has_test_folder = test_dir_path and os.path.exists(test_dir_path)
    has_csv = hasattr(config.data, 'csv_file') and config.data.csv_file and \
              hasattr(config.data, 'image_dir') and config.data.image_dir
    
    # PRIORITY 1: FOLDER-BASED MODE (for OCT/CXR data)
    # If both train and test folders exist, use them
    if has_train_folder and has_test_folder:
        print("="*70)
        print("FOLDER-BASED MODE: Folder train + Folder test")
        print("="*70)
        
        # Get classes_to_use filter
        classes_to_use = config.data.classes_to_use if hasattr(config.data, 'classes_to_use') else None
        
        # Get balancing parameters from config
        balance_train = getattr(config.data, 'balance_train_classes', False)  # Default False for train
        balance_test = getattr(config.data, 'balance_classes', True)  # Default True for test
        balance_seed = getattr(config.data, 'balance_seed', 42)
        
        # Load training data from folder
        print(f"\n1. Loading TRAINING data from folder: {train_dir_path}")
        train_dataset = MedicalImageDataset(
            train_dir_path, 
            transform=train_transform,
            classes_to_use=classes_to_use,
            balance_classes=balance_train,
            balance_seed=balance_seed
        )
        print(f"   ✓ Loaded {len(train_dataset)} training samples")
        print(f"   ✓ Classes: {train_dataset.classes}")
        print(f"   ✓ Distribution: {[len(train_dataset.class_samples[i]) for i in range(len(train_dataset.classes))]}")
        
        # Load test data from folder
        print(f"\n2. Loading TEST data from folder: {test_dir_path}")
        test_dataset = MedicalImageDataset(
            test_dir_path, 
            transform=test_transform,
            classes_to_use=classes_to_use,
            balance_classes=balance_test,
            balance_seed=balance_seed
        )
        
        print(f"   ✓ Loaded {len(test_dataset)} test samples")
        print(f"   ✓ Classes: {test_dataset.classes}")
        print(f"   ✓ Distribution: {[len(test_dataset.class_samples[i]) for i in range(len(test_dataset.classes))]}")
        
        print("\n" + "="*70)
        print(f"SUMMARY: {len(train_dataset)} train / {len(test_dataset)} test")
        print("="*70 + "\n")
    
    # PRIORITY 2: HYBRID MODE (for ISIC data - folder train + CSV test)
    elif has_train_folder and has_csv:
        print("="*70)
        print("HYBRID MODE: Folder train + CSV test")
        print("="*70)
        
        # Get class filters
        # For folder train: use folder names (classes_to_use)
        # For CSV test: use CSV column names (classes_to_use_csv)
        classes_to_use_folder = config.data.classes_to_use if hasattr(config.data, 'classes_to_use') else None
        classes_to_use_csv = config.data.classes_to_use_csv if hasattr(config.data, 'classes_to_use_csv') else None
        
        # Get balancing parameters from config
        balance_train = getattr(config.data, 'balance_train_classes', False)  # Default False for train
        balance_test = getattr(config.data, 'balance_classes', True)  # Default True for test
        balance_seed = getattr(config.data, 'balance_seed', 42)
        
        # Load training data from folder
        print(f"\n1. Loading TRAINING data from folder: {train_dir_path}")
        train_dataset = MedicalImageDataset(
            train_dir_path, 
            transform=train_transform,
            classes_to_use=classes_to_use_folder,
            balance_classes=balance_train,
            balance_seed=balance_seed
        )
        print(f"   ✓ Loaded {len(train_dataset)} training samples")
        print(f"   ✓ Classes: {train_dataset.classes}")
        print(f"   ✓ Distribution: {[len(train_dataset.class_samples[i]) for i in range(len(train_dataset.classes))]}")
        
        # Load test data from CSV
        print(f"\n2. Loading TEST data from CSV: {config.data.csv_file}")
        
        # Get balancing parameters from config
        balance_test = getattr(config.data, 'balance_classes', True)  # Default True
        balance_seed = getattr(config.data, 'balance_seed', 42)
        
        test_dataset = ISICDataset(
            csv_file=config.data.csv_file,
            image_dir=config.data.image_dir,
            transform=test_transform,
            classes_to_use=classes_to_use_csv,  # Use CSV column names
            balance_classes=balance_test,
            balance_seed=balance_seed
        )
        
        print(f"   ✓ Loaded {len(test_dataset)} test samples")
        
        print("\n" + "="*70)
        print(f"SUMMARY: {len(train_dataset)} train / {len(test_dataset)} test")
        print("="*70 + "\n")
    
    # Check if using CSV-based for both train and test
    elif has_csv:
        # CSV-based ISIC dataset
        print("Using CSV-based ISIC dataset for both train and test")
        
        # You can specify which classes to use
        classes_to_use = None  # Use all classes
        if hasattr(config.data, 'classes_to_use'):
            classes_to_use = config.data.classes_to_use
        
        # Get balancing parameters - don't balance the full dataset before splitting
        balance_seed = getattr(config.data, 'balance_seed', 42)
        
        # Create full dataset (no balancing yet - will balance after split)
        full_dataset = ISICDataset(
            csv_file=config.data.csv_file,
            image_dir=config.data.image_dir,
            transform=train_transform,
            classes_to_use=classes_to_use,
            balance_classes=False,  # Don't balance before splitting
            balance_seed=balance_seed
        )
        
        # If you have a separate test CSV
        if hasattr(config.data, 'test_csv_file') and config.data.test_csv_file:
            balance_test = getattr(config.data, 'balance_classes', True)
            
            test_dataset = ISICDataset(
                csv_file=config.data.test_csv_file,
                image_dir=config.data.test_image_dir if hasattr(config.data, 'test_image_dir') else config.data.image_dir,
                transform=test_transform,
                classes_to_use=classes_to_use,
                balance_classes=balance_test,
                balance_seed=balance_seed
            )
            train_dataset = full_dataset
        else:
            # Use configurable split for ICL
            total_size = len(full_dataset.samples)
            train_ratio = config.data.train_split_ratio
            train_size = int(train_ratio * total_size)
            
            print(f"\nICL Data Split:")
            print(f"  Support set: {train_size} examples ({train_ratio*100:.1f}%)")
            print(f"  Test set: {total_size - train_size} examples ({(1-train_ratio)*100:.1f}%)")
            print(f"  Ratio: {train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% test (few-shot ICL)")
            
            # Split samples
            train_samples = full_dataset.samples[:train_size]
            test_samples = full_dataset.samples[train_size:]
            
            # Get balancing parameters
            balance_train = getattr(config.data, 'balance_train_classes', False)
            balance_test = getattr(config.data, 'balance_classes', True)
            
            # Create train dataset (support set for ICL)
            train_dataset = ISICDataset(
                csv_file=config.data.csv_file,
                image_dir=config.data.image_dir,
                transform=train_transform,
                classes_to_use=classes_to_use,
                balance_classes=False,  # Don't auto-balance, will assign samples manually
                balance_seed=balance_seed
            )
            train_dataset.samples = train_samples
            train_dataset.class_samples = {i: [] for i in range(len(train_dataset.class_columns))}
            for idx, (path, label) in enumerate(train_dataset.samples):
                train_dataset.class_samples[label].append(idx)
            
            # Manually balance train if requested
            if balance_train:
                print(f"\n🔄 Balancing training split...")
                train_dataset.balance_by_min_class(seed=balance_seed)
            
            print(f"  Support set class distribution: {[len(train_dataset.class_samples[i]) for i in range(len(train_dataset.class_columns))]}")
            
            # Create test dataset (main evaluation set)
            test_dataset = ISICDataset(
                csv_file=config.data.csv_file,
                image_dir=config.data.image_dir,
                transform=test_transform,
                classes_to_use=classes_to_use,
                balance_classes=False,  # Don't auto-balance, will assign samples manually
                balance_seed=balance_seed
            )
            test_dataset.samples = test_samples
            test_dataset.class_samples = {i: [] for i in range(len(test_dataset.class_columns))}
            for idx, (path, label) in enumerate(test_dataset.samples):
                test_dataset.class_samples[label].append(idx)
            
            # Manually balance test if requested
            if balance_test:
                print(f"\n🔄 Balancing test split...")
                test_dataset.balance_by_min_class(seed=balance_seed)
            
            print(f"  Test set class distribution: {[len(test_dataset.class_samples[i]) for i in range(len(test_dataset.class_columns))]}")
    
    else:
        # Folder-based dataset for both train and test
        print("Using folder-based dataset for both train and test")
        train_dir = os.path.join(config.data.data_root, config.data.train_dir)
        test_dir = os.path.join(config.data.data_root, config.data.test_dir)
        
        # Get classes_to_use filter
        classes_to_use = config.data.classes_to_use if hasattr(config.data, 'classes_to_use') else None
        
        # Get balancing parameters from config
        balance_train = getattr(config.data, 'balance_train_classes', False)
        balance_test = getattr(config.data, 'balance_classes', True)
        balance_seed = getattr(config.data, 'balance_seed', 42)
        
        train_dataset = MedicalImageDataset(
            train_dir, 
            transform=train_transform,
            classes_to_use=classes_to_use,
            balance_classes=balance_train,
            balance_seed=balance_seed
        )
        test_dataset = MedicalImageDataset(
            test_dir, 
            transform=test_transform,
            classes_to_use=classes_to_use,
            balance_classes=balance_test,
            balance_seed=balance_seed
        )
    
    if icl_mode:
        # Wrap datasets for ICL
        train_dataset = ICLDataset(train_dataset, config.icl.num_support_shots)
        test_dataset = ICLDataset(test_dataset, config.icl.num_support_shots)
        
        # Custom collate function for ICL
        def icl_collate_fn(batch):
            query_images = torch.stack([item[0] for item in batch])
            query_labels = torch.tensor([item[1] for item in batch])
            support_sets = [item[2] for item in batch]
            return query_images, query_labels, support_sets
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=icl_collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=icl_collate_fn
        )
    else:
        # Standard dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
    
    return train_loader, test_loader, train_dataset, test_dataset