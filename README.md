# ObjectLens

Upload a photo of small objects, and ObjectLens detects, segments, and measures every instance using **Mask R-CNN ResNet-50 FPN v2**, **SAM ViT-B**, or **SAM2 Hiera-S**.

---

## What it does

- Detects objects and draws per-instance segmentation masks
- Computes shape descriptors for each object: **roundness, aspect ratio, eccentricity, area, perimeter**
- Colors the overlay by any metric (or by unique instance color)
- Shows an interactive distribution histogram
- Exports all results as a CSV

---

Mask R-CNN weights download automatically through torchvision on first run. SAM ViT-B downloads to `weights/` by default, or to the directory configured in `OBJECTLENS_WEIGHTS_DIR`. SAM2 downloads through Hugging Face cache.

---

## Run locally

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the deterministic core tests:

```bash
python -m unittest discover
```

---

## Deployment

The repository includes Streamlit deployment defaults in `.streamlit/config.toml` and pins Python through `.python-version`.

For Streamlit Community Cloud or similar platforms:

1. Set the app entry point to `app.py`.
2. Use Python 3.11.
3. Ensure the host has enough memory for PyTorch model loading. SAM/SAM2 are heavier than Mask R-CNN.
4. Do not commit downloaded model files. `weights/` is ignored by git.

---

## Project structure

```
objectlens/
├── app.py              # entrypoint — wires everything together
├── .streamlit/
│   └── config.toml     # deployment/runtime config
├── tests/
│   └── test_core.py    # deterministic tests for features and overlay
├── requirements.txt
├── core/
│   ├── model.py        # model loading and inference
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
| Aspect ratio | `max(width, height) / min(width, height)` | ≥ 1 |
| Eccentricity | from fitted ellipse `√(1 − (minor/major)²)` | 0–1 (0 = circle) |

---

## Stack

`streamlit` · `torch` · `torchvision` · `segment-anything` · `sam2` · `opencv-python-headless` · `pandas` · `plotly`

---
