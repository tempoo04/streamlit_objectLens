"""
core/model.py — unified model registry

Each model exposes the same interface:
    load_<name>()  →  cached resource
    infer_<name>(img_np, score_thresh, mask_thresh)  →  list of detection dicts

Detection dict schema (all models must return this):
    {
        "id":          int,
        "category":    str,
        "score":       float,   # 0–1 confidence (SAM uses predicted_iou)
        "box":         [x1,y1,x2,y2],
        "mask_raw":    np.ndarray (H,W) float,
        "mask_thresh": float,
    }

Public entry points:
    MODELS          — ordered dict of {display_name: key}
    load_model(key) — returns cached model object(s)
    run_inference(image_bytes, model_key, score_thresh, mask_thresh)
"""

from __future__ import annotations
import io
import numpy as np
from PIL import Image
import streamlit as st

# ── registry ──────────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "Mask R-CNN ResNet-101  (accurate)": "maskrcnn101",
    "SAM ViT-B  (segment anything)":     "sam_vitb",
    "YOLO11-seg  (fastest)":             "yolo11",
}

# ─────────────────────────────────────────────────────────────────────────────
# Mask R-CNN ResNet-101
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_maskrcnn101():
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model   = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    return model, weights.meta["categories"]


def _infer_maskrcnn101(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,
) -> list[dict]:
    import torch, torchvision

    model, categories = _load_maskrcnn101()
    tensor = torchvision.transforms.functional.to_tensor(
        Image.fromarray(img_np)
    )
    with torch.no_grad():
        out = model([tensor])[0]

    boxes  = out["boxes"].numpy()
    scores = out["scores"].numpy()
    labels = out["labels"].numpy()
    masks  = out["masks"].numpy()          # (N,1,H,W)

    keep = scores >= score_thresh
    detections = []
    for i, (box, score, label, mask) in enumerate(
        zip(boxes[keep], scores[keep], labels[keep], masks[keep])
    ):
        detections.append({
            "id":          i + 1,
            "category":    categories[label],
            "score":       float(score),
            "box":         box.tolist(),
            "mask_raw":    mask[0],
            "mask_thresh": mask_thresh,
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# SAM ViT-B  (automatic mask generator — no prompts needed)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_sam_vitb():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import urllib.request, os, torch

    ckpt = "sam_vit_b_01ec64.pth"
    url  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    if not os.path.exists(ckpt):
        with st.spinner("Downloading SAM ViT-B weights (~375 MB) — once only…"):
            urllib.request.urlretrieve(url, ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam    = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.to(device)
    # CPU-safe settings:
    # points_per_side=16  -> 256 points instead of 1024 (4x faster)
    # crop_n_layers=0     -> skip multi-scale cropping (main memory killer)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.90,
        min_mask_region_area=150,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
    )
    return generator


def _infer_sam_vitb(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,   # used as min stability score override
) -> list[dict]:
    generator = _load_sam_vitb()
    masks_out = generator.generate(img_np)   # list of mask dicts

    H, W = img_np.shape[:2]
    detections = []
    for i, m in enumerate(masks_out):
        score = float(m["predicted_iou"])
        if score < score_thresh:
            continue
        seg   = m["segmentation"].astype(np.uint8)   # bool → uint8
        bbox  = m["bbox"]                             # [x, y, w, h] XYWH
        x, y, w, h = bbox
        detections.append({
            "id":          i + 1,
            "category":    "object",          # SAM is class-agnostic
            "score":       score,
            "box":         [x, y, x + w, y + h],
            "mask_raw":    seg.astype(np.float32),
            "mask_thresh": 0.5,               # already binary
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# YOLO11-seg
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_yolo11():
    from ultralytics import YOLO
    import os

    # Store weights in a fixed local folder next to this file so Windows
    # can always resolve the path, regardless of the working directory.
    weights_dir  = os.path.join(os.path.dirname(__file__), "..", "weights")
    weights_path = os.path.join(weights_dir, "yolo11n-seg.pt")
    os.makedirs(weights_dir, exist_ok=True)

    # On first run Ultralytics downloads the file; we tell it exactly where.
    return YOLO(weights_path if os.path.exists(weights_path) else "yolo11n-seg.pt",
                task="segment")


def _infer_yolo11(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,
) -> list[dict]:
    model   = _load_yolo11()
    results = model(img_np, conf=score_thresh, verbose=False)[0]

    detections = []
    if results.masks is None:
        return detections

    masks  = results.masks.data.cpu().numpy()   # (N, H, W)
    boxes  = results.boxes
    names  = model.names

    for i, (mask, box) in enumerate(zip(masks, boxes)):
        score = float(box.conf[0])
        cls   = int(box.cls[0])
        xyxy  = box.xyxy[0].tolist()

        # YOLO masks are resized to input size; upsample to original if needed
        if mask.shape != img_np.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8)).resize(
                (img_np.shape[1], img_np.shape[0]), PILImage.NEAREST
            )
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        detections.append({
            "id":          i + 1,
            "category":    names[cls],
            "score":       score,
            "box":         xyxy,
            "mask_raw":    mask,
            "mask_thresh": mask_thresh,
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
_LOADERS = {
    "maskrcnn101": _load_maskrcnn101,
    "sam_vitb":    _load_sam_vitb,
    "yolo11":      _load_yolo11,
}

_INFER = {
    "maskrcnn101": _infer_maskrcnn101,
    "sam_vitb":    _infer_sam_vitb,
    "yolo11":      _infer_yolo11,
}


def load_model(key: str):
    """Pre-warm the selected model (optional — inference also triggers load)."""
    return _LOADERS[key]()


@st.cache_data
def run_inference(
    image_bytes: bytes,
    model_key: str,
    score_thresh: float,
    mask_thresh: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Unified inference entry point. Cached by (image, model, thresholds).
    Switching the overlay color metric never re-runs inference.
    """
    img_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    detections = _INFER[model_key](img_np, score_thresh, mask_thresh)
    return img_np, detections