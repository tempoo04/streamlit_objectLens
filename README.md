# StreamLENS 🔬

Upload a photo of small objects — coins, seeds, pills, screws, fruits — and ObjectLens will detect, segment, and measure every instance using **Mask R-CNN ResNet-50 FPN v2** or **SAM ViT-B**, both pretrained on COCO/SA-1B.

---

## What it does

- Detects objects and draws per-instance segmentation masks
- Computes shape descriptors for each object: **roundness, aspect ratio, eccentricity, area, perimeter**
- Colors the overlay by any metric (or by unique instance color)
- Shows an interactive distribution histogram
- Exports all results as a CSV

---

Mask R-CNN weights (~170 MB) download automatically via torchvision on first run. SAM ViT-B weights (~375 MB) download once to `weights/`.

---

## Project structure

```
objectlens/
├── app.py              # entrypoint — wires everything together
├── requirements.txt
├── core/
│   ├── model.py        # Mask R-CNN loading & inference
│   ├── features.py     # shape descriptor computation
│   └── overlay.py      # mask colorization & rendering
└── ui/
    ├── sidebar.py      # sidebar controls
    ├── results.py      # metric cards, histogram, table
    └── styles.py       # CSS injection
```

---

## Shape descriptors

| Metric | Formula | Range |
|---|---|---|
| Roundness | `4π · area / perimeter²` | 0–1 (1 = perfect circle) |
| Aspect ratio | `bbox width / bbox height` | ≥ 1 |
| Eccentricity | from fitted ellipse `√(1 − (minor/major)²)` | 0–1 (0 = circle) |

---

## Stack

`streamlit` · `torchvision` · `segment-anything` · `opencv-python` · `pandas` · `plotly`

---

**Author:** Turgut Nasrullayev — [LinkedIn](https://linkedin.com/in/turgut-nasrullayev-735047158/) · [GitHub](https://github.com/tempoo04)  
**License:** MIT
