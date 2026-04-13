# Official implementation of the paper PIKACHU: Prototypical In-context Knowledge Adaptation for Clinical Heterogeneous Usage

## Accepted to MIDL 2026 (poster)

## Folder structure

This README reflects the **current files** in this repository.

```text
pikachu/
├── .git/
├── README.md
├── config.py
├── dataset.py
├── utils.py
├── icl_module.py
├── run_experiments.py
├── baseline_dinov2.py
├── baseline_medclip.py
├── baseline_siglip.py
├── baseline_vit.py
├── icl_dinov2.py
├── icl_medclip.py
├── icl_siglip.py
├── icl_vit.py
└── data/   (currently empty)
```

## File descriptions

- `config.py`: dataclass-based configuration for data, model, ICL, and output directories.
- `dataset.py`: dataset classes and dataloader creation for folder-based, CSV-based, and hybrid modes.
- `utils.py`: metric computation, confusion matrix/ROC plotting, result saving, and baseline-vs-ICL comparison plots.
- `icl_module.py`: reusable ICL building blocks (cross-attention ICL, prototype ICL, adaptive fusion, weighted kNN).

### Baseline scripts
- `baseline_dinov2.py`: DINOv2 linear-probe baseline.
- `baseline_medclip.py`: PubMedCLIP/BiomedCLIP zero-shot baseline.
- `baseline_siglip.py`: SigLIP2 zero-shot baseline.
- `baseline_vit.py`: ViT baseline.

### ICL scripts
- `icl_dinov2.py`: DINOv2 ProtoNet-style ICL.
- `icl_medclip.py`: PubMedCLIP ProtoNet-style ICL.
- `icl_siglip.py`: SigLIP2 ProtoNet-style ICL.
- `icl_vit.py`: ViT ProtoNet-style ICL.

## Known limitation in `run_experiments.py`

`run_experiments.py` matches some renamed files (`icl_dinov2.py`, `icl_vit.py`), but it still imports CLIP modules that are **not present** in this repository:
- missing: `baseline_clip.py`, `icl_clip.py`
- available related files: `baseline_medclip.py`, `icl_medclip.py`

As a result, DINOv2/ViT paths can run, but CLIP steps fail unless imports are updated or matching files are added.

## Data expectations

The `data/` directory is currently empty. Before running experiments, provide datasets and update paths in `config.py` (for example: `data_root`, `train_dir`, `test_dir`, `csv_file`, `image_dir`).

## Typical usage

1. Edit dataset and experiment settings in `config.py`.
2. Run one script at a time, e.g.:
   - `python baseline_dinov2.py`
   - `python baseline_medclip.py`
   - `python icl_dinov2.py`
3. Check outputs in `results/` and `checkpoints/` (created automatically).
