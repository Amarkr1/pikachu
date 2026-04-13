import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DataConfig:
    """Configuration for dataset"""
    # Folder-based configuration
    data_root: str = "data"
    
    # DATASET SELECTION: Choose one of the following
    # Option 1: ISIC Skin Cancer (folder train + CSV test)
    train_dir: str = "train/skin_isic"
    test_dir: str = ""  # Empty for CSV-based test
    csv_file: str = "data/ISIC_2019_Training_GroundTruth.csv"
    image_dir: str = "/scratch/a/amarkr1/data/isic/ISIC_2019_Training_Input/"
    # For folder-based train: use folder names (melanoma, nevus, ak, bcc, bkl)
    # For CSV-based test: classes_to_use_csv uses CSV column names (MEL, NV, AK, BCC, BKL)
    classes: List[str] = field(default_factory=lambda: ["melanoma", "nevus"])
    classes_to_use: Optional[List[str]] = field(default_factory=lambda: ["melanoma", "nevus"])  # Folder names
    classes_to_use_csv: Optional[List[str]] = field(default_factory=lambda: ["MEL", "NV"])  # CSV columns
    # For 5-class classification (uncomment these):
    # classes: List[str] = field(default_factory=lambda: ["melanoma", "nevus", "ak", "bcc", "bkl"])
    # classes_to_use: Optional[List[str]] = field(default_factory=lambda: ["melanoma", "nevus", "ak", "bcc", "bkl"])
    # classes_to_use_csv: Optional[List[str]] = field(default_factory=lambda: ["MEL", "NV", "AK", "BCC", "BKL"])
    
    # Option 2: OCT (Optical Coherence Tomography) - folder train + folder test
    # train_dir: str = "train/oct"
    # test_dir: str = "/scratch/a/amarkr1/data/OCT2017/test_v2"
    # csv_file: Optional[str] = None  # Not used for OCT
    # image_dir: Optional[str] = None  # Not used for OCT
    # # For 4-class:
    # # classes: List[str] = field(default_factory=lambda: ["CNV", "DME", "DRUSEN", "NORMAL"])
    # # classes_to_use: Optional[List[str]] = field(default_factory=lambda: ["CNV", "DME", "DRUSEN", "NORMAL"])
    # # For 2-class:
    # classes: List[str] = field(default_factory=lambda: ["CNV", "NORMAL"])
    # classes_to_use: Optional[List[str]] = field(default_factory=lambda: ["CNV", "NORMAL"])
    # classes_to_use_csv: Optional[List[str]] = None  # Not used for OCT
    
    # Option 3: Diabetic Retinopathy (DR) - folder train + folder test
    # Classes are numeric: 0, 1, 2, 3, 4 representing severity levels
    # train_dir: str = "train/dr"
    # test_dir: str = "/scratch/a/amarkr1/data/dr/test"
    # csv_file: Optional[str] = None  # Not used for DR
    # image_dir: Optional[str] = None  # Not used for DR
    # # For 5-class (all severity levels):
    # classes: List[str] = field(default_factory=lambda: ["0", "1", "2", "3", "4"])
    # # classes_to_use: Optional[List[str]] = field(default_factory=lambda: ["0", "1", "2", "3", "4"])
    # # classes_to_use_csv: Optional[List[str]] = None  # Not used for DR
    # # For binary (healthy vs diseased):
    # classes: List[str] = field(default_factory=lambda: ["0", "4"])  # 0=no DR, 1=mild DR
    # classes_to_use: Optional[List[str]] = field(default_factory=lambda: ["0", "4"])
    # # For 3-class (healthy, mild, severe):
    # # classes: List[str] = field(default_factory=lambda: ["0", "1", "4"])  # 0=no DR, 1=mild, 4=severe
    # # classes_to_use: Optional[List[str]] = field(default_factory=lambda: ["0", "1", "4"])
    
    # Option 4: CXR (Chest X-ray) - folder train + folder test
    # train_dir: str = "train/cxr"
    # test_dir: str = "test/cxr"
    # csv_file: Optional[str] = None
    # image_dir: Optional[str] = None
    # classes: List[str] = field(default_factory=lambda: ["class1", "class2"])  # Update with actual classes
    # classes_to_use: Optional[List[str]] = field(default_factory=lambda: ["class1", "class2"])
    # classes_to_use_csv: Optional[List[str]] = None
    
    test_csv_file: Optional[str] = None  # Separate test CSV if needed
    test_image_dir: Optional[str] = None  # Separate test images if needed
    
    # Data splitting (for CSV mode only)
    train_split_ratio: float = 1.0  # Use all samples from train_dir
    num_support_samples: Optional[int] = None  # Not used when using folder structure
    use_folder_train_csv_test: bool = False  # Set True for ISIC, False for OCT/CXR
    
    # Class balancing
    balance_classes: bool = True  # Auto-balance test sets by min class (RECOMMENDED: True)
    balance_train_classes: bool = False  # Balance training sets (usually False for ICL support sets)
    balance_seed: int = 42  # Random seed for reproducible balancing
    
    # Data loading
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    
@dataclass
class ModelConfig:
    """Configuration for foundation models"""
    # Vision-Language Foundation Models
    clip_model: str = "openai/clip-vit-base-patch32"
    blip_model: str = "Salesforce/blip-image-captioning-base"
    biomed_clip_model: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    siglip_model: str = "google/siglip2-base-patch16-224"
    
    # Vision Foundation Models
    dinov2_model: str = "facebook/dinov2-base"
    vit_model: str = "google/vit-base-patch16-224"
    sam_model: str = "facebook/sam-vit-base"
    
@dataclass
class ICLConfig:
    """Configuration for In-Context Learning"""
    num_support_shots: int = 5  # K-shot learning
    icl_method: str = "prototypical_attention"  # 'cross_attention_knn' or 'prototypical_attention'
    similarity_metric: str = "cosine"  # cosine, euclidean
    aggregation_method: str = "attention"  # average, weighted, attention, prototype
    use_text_guidance: bool = True  # For VLMs
    temperature: float = 0.07
    alpha: float = 0.6  # ICL vs zero-shot weight
    use_adaptive_weights: bool = True
    cross_attention_heads: int = 8
    
@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    seed: int = 42
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    icl: ICLConfig = field(default_factory=ICLConfig)
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    
    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)