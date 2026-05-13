"""
core/model.py — model registry (Mask R-CNN · SAM ViT-B)

Detection dict schema returned by all inference functions:
    {
        "id":          int,
        "category":    str,
        "score":       float,
        "box":         [x1, y1, x2, y2],
        "mask_raw":    np.ndarray (H, W) float32,
        "mask_thresh": float,
    }
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
    tensor = torchvision.transforms.functional.to_tensor(Image.fromarray(img_np))
    with torch.no_grad():
        out = model([tensor])[0]

    boxes  = out["boxes"].numpy()
    scores = out["scores"].numpy()
    labels = out["labels"].numpy()
    masks  = out["masks"].numpy()   # (N, 1, H, W)

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
# SAM ViT-B — optimised for CPU / weak machines
#
# Speed knobs (weakest → strongest effect):
#   points_per_side   : 12  → 144 grid points   (vs 32²=1024 default)
#   _SAM_INPUT_SIZE   : 512 → ViT encoder input  (vs 1024 default, ~4× faster)
#   crop_n_layers     : 0   → no multi-scale crops (was the main memory killer)
# ─────────────────────────────────────────────────────────────────────────────
_SAM_INPUT_SIZE = 512   # resize longest side to this before encoding


@st.cache_resource
def _load_sam_vitb():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import urllib.request, os, torch

    # Store weights next to this file so path is always stable on Windows
    weights_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "weights"))
    os.makedirs(weights_dir, exist_ok=True)
    ckpt = os.path.join(weights_dir, "sam_vit_b_01ec64.pth")
    url  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    if not os.path.exists(ckpt):
        with st.spinner("Downloading SAM ViT-B weights (~375 MB) — once only…"):
            urllib.request.urlretrieve(url, ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam    = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.to(device)

    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=12,           # 144 points — good balance on CPU
        points_per_batch=32,          # process points in small batches → lower peak RAM
        pred_iou_thresh=0.82,
        stability_score_thresh=0.90,
        box_nms_thresh=0.70,
        min_mask_region_area=200,     # filter tiny noise masks early
        crop_n_layers=0,              # no multi-scale cropping
        crop_n_points_downscale_factor=1,
    )
    return generator


def _resize_for_sam(img_np: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Downscale so the longest side == _SAM_INPUT_SIZE.
    Returns (resized_image, scale_factor).
    scale_factor < 1 means we shrank the image.
    Never upscales.
    """
    H, W   = img_np.shape[:2]
    scale  = min(_SAM_INPUT_SIZE / max(H, W), 1.0)
    if scale == 1.0:
        return img_np, 1.0
    new_W, new_H = int(W * scale), int(H * scale)
    resized = np.array(
        Image.fromarray(img_np).resize((new_W, new_H), Image.BILINEAR)
    )
    return resized, scale


def _infer_sam_vitb(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,   # not used — SAM masks are already binary
) -> list[dict]:
    import cv2

    generator          = _load_sam_vitb()
    img_small, scale   = _resize_for_sam(img_np)
    orig_H, orig_W     = img_np.shape[:2]

    masks_out = generator.generate(img_small)

    # Filter by score first — avoids resizing masks we'll discard anyway
    masks_out = [m for m in masks_out if m["predicted_iou"] >= score_thresh]

    detections = []
    for i, m in enumerate(masks_out):
        seg  = m["segmentation"].astype(np.uint8)   # bool → uint8, still at small size
        x, y, w, h = m["bbox"]                      # XYWH in small-image space

        # Single cv2.resize call per mask (much faster than PIL in a loop)
        mask_full = cv2.resize(
            seg, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)

        # Scale bbox back to original image space
        inv = 1.0 / scale
        detections.append({
            "id":          i + 1,
            "category":    "object",
            "score":       float(m["predicted_iou"]),
            "box":         [x * inv, y * inv, (x + w) * inv, (y + h) * inv],
            "mask_raw":    mask_full,
            "mask_thresh": 0.5,
        })

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
_LOADERS = {
    "maskrcnn101": _load_maskrcnn101,
    "sam_vitb":    _load_sam_vitb,
}

_INFER = {
    "maskrcnn101": _infer_maskrcnn101,
    "sam_vitb":    _infer_sam_vitb,
}


def load_model(key: str):
    """Pre-warm the selected model. Optional — inference also triggers load."""
    return _LOADERS[key]()


@st.cache_data
def run_inference(
    image_bytes: bytes,
    model_key: str,
    score_thresh: float,
    mask_thresh: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Unified inference entry point. Result is cached by (image, model, thresholds).
    Changing overlay color or chart metric never re-runs inference.
    """
    img_np     = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    detections = _INFER[model_key](img_np, score_thresh, mask_thresh)
    return img_np, detections