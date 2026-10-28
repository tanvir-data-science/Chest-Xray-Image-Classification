# ResNet-18 baseline

This experiment trains a two-class classifier using ImageNet-pretrained ResNet-18. The backbone is frozen and the final layer is trained with class-weighted cross entropy. A seeded, stratified, group-aware split reserves approximately 20% of data/train for validation; data/val is not used because it contains only 16 images. Patient groups are approximated from filenames and exact duplicates are kept together. These heuristics cannot guarantee patient separation.

## Windows setup

From the repository root in Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install torch torchvision scikit-learn pillow numpy
python baseline_resnet18/train.py
```

The first run downloads pretrained weights. Use a compatible PyTorch CUDA installation if you have an NVIDIA GPU; CPU training may be slow. Change settings with `python baseline_resnet18/train.py --help`. On the original machine, if the repository root is the outer data folder, its inner data directory contains train/test/val; defaults work from that root.

Outputs are saved under baseline_resnet18/outputs: split.csv, best_model.pt and, when test evaluation is allowed, metrics.json. The best checkpoint is selected by validation loss. Confusion matrix rows are true labels and columns predicted labels, ordered NORMAL, PNEUMONIA. The model converts images to RGB for the pretrained network.

## Data leakage warning

The repository listing shows 320 test images with filename-derived patient IDs also present in training. IDs may not uniquely identify patients across categories, so this needs investigation using authoritative metadata. The script checks shared inferred IDs and exact image hashes, then skips test evaluation by default if there is overlap. It still trains and saves the checkpoint. After confirming the splits are valid, `--allow-test-overlap` explicitly evaluates the test set; such results may be biased if overlap reflects the same patients. Re-running retrains the model. The original 16-image validation set is not used.

This is a research baseline, not a clinically validated diagnostic tool. No trained model or performance results are included.
