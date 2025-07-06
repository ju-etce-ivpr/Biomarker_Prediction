# Glioma Biomarker Prediction

Predicts 5 genetic markers (IDH, 1p/19q, ATRX, MGMT, TERT) from Whole Slide Images using a deep learning pipeline with a composite loss function.

---

## 🧠 Biomarkers Predicted
- **IDH** (Isocitrate Dehydrogenase)
- **1p/19q codeletion**
- **ATRX**
- **MGMT**
- **TERT**

---

## 📁 Repository Structure
```
glioma-biomarker-predictor/
├── main.py                # CLI entry point
├── config.yaml            # All configuration settings
├── train.py               # Training script
├── test.py                # Inference script
│
├── data/
│   ├── dataset.py         # PyTorch Dataset class
│   ├── patch_extractor.py # Patch generator from .svs WSI
│
├── models/
│   ├── backbone.py        # KimiaNet-like CNN
│   ├── fcn_heads.py       # Parallel FCNs (1 per biomarker)
│   ├── loss.py            # Composite loss function
```

---

## 🔧 Setup
```bash
pip install -r requirements.txt
```
Make sure OpenSlide is installed (e.g., `libopenslide-dev`, `python-openslide`).

---

## 🖼️ Patch Extraction
```python
from data.patch_extractor import process_wsi_directory

process_wsi_directory(
    wsi_dir="./data/wsis",
    patch_output_root="./data/patches",
    patch_size=224,
    levels=(0, 2),  # 20x, 5x
    max_patches_per_level=(2000, 200)
)
```

---

## 🚆 Train
```bash
python main.py --mode train --config config.yaml
```
Trains two models (local/global) and saves checkpoints to `./checkpoints`.

---

## 🧪 Test
```bash
python main.py --mode test --config config.yaml --checkpoint ./checkpoints
```
Prints biomarker predictions per WSI using soft-voting over local/global outputs.

---

## 📌 Notes
- Uses a composite loss:
  - Multi-label Weighted Cross Entropy
  - Conditional Probability Loss
  - Spectral Graph Loss
- Designed for use with .svs format H&E-stained WSIs

---

## 📄 Citation
Based on:
> "Predicting Genetic Markers for Brain Tumors Using a Composite Loss" — Arijit De et al. (2021)

---

## 🔜 Future Work
- Add visualization & explainability (e.g., heatmaps)
- Support for additional biomarkers
- GUI for clinical users
