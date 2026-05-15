"""
core/model.py — model registry (Mask R-CNN ResNet-50 FPN v2 · SAM ViT-B · SAM2 Hiera-S)

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
    "Mask R-CNN ResNet-50 v2  (accurate)": "maskrcnn_r50v2",
    "SAM ViT-B  (segment anything)":       "sam_vitb",
    "SAM2 Hiera-S  (better quality)":      "sam2_small",
}

# ─────────────────────────────────────────────────────────────────────────────
# Mask R-CNN ResNet-50 FPN v2
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_maskrcnn_r50v2():
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model   = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    return model, weights.meta["categories"]


def _infer_maskrcnn_r50v2(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,
) -> list[dict]:
    import torch, torchvision

    model, categories = _load_maskrcnn_r50v2()
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
# Three presets trade quality for speed/RAM:
#   fast     : 256 px input,  6² = 36  points  (~2–4 s on CPU, low RAM)
#   balanced : 384 px input, 10² = 100 points  (~6–10 s on CPU)
#   quality  : 512 px input, 12² = 144 points  (original setting)
#
# crop_n_layers=0 stays fixed across all presets — was the main memory killer.
# ─────────────────────────────────────────────────────────────────────────────
_SAM_PRESETS: dict[str, dict] = {
    "fast":     {"input_size": 256, "points_per_side": 6,  "points_per_batch": 8},
    "balanced": {"input_size": 384, "points_per_side": 10, "points_per_batch": 16},
    "quality":  {"input_size": 512, "points_per_side": 12, "points_per_batch": 32},
}


@st.cache_resource
def _load_sam_vitb(quality: str = "fast"):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import urllib.request, os, torch

    cfg = _SAM_PRESETS[quality]

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
        points_per_side=cfg["points_per_side"],
        points_per_batch=cfg["points_per_batch"],
        pred_iou_thresh=0.82,
        stability_score_thresh=0.90,
        box_nms_thresh=0.70,
        min_mask_region_area=200,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
    )
    return generator, cfg["input_size"]


def _resize_for_sam(img_np: np.ndarray, input_size: int) -> tuple[np.ndarray, float]:
    H, W  = img_np.shape[:2]
    scale = min(input_size / max(H, W), 1.0)
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
    quality: str = "fast",
) -> list[dict]:
    import cv2

    generator, input_size  = _load_sam_vitb(quality)
    img_small, scale       = _resize_for_sam(img_np, input_size)
    orig_H, orig_W         = img_np.shape[:2]

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
# SAM2 Hiera-Small  (facebook/sam2.1-hiera-small  ~183 MB via HuggingFace)
#
# Uses the same _SAM_PRESETS and _resize_for_sam as SAM ViT-B.
# SAM2AutomaticMaskGenerator output dict is identical to SAM1's.
# torch.compile is suppressed — avoids Windows build failures.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_sam2(quality: str = "fast"):
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    import torch

    # Suppress torch.compile — can fail on Windows without MSVC
    torch._dynamo.config.suppress_errors = True

    cfg    = _SAM_PRESETS[quality]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with st.spinner("Loading SAM2 weights (~183 MB, once only)…"):
        sam2 = build_sam2_hf("facebook/sam2.1-hiera-small", device=device)

    generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=cfg["points_per_side"],
        points_per_batch=cfg["points_per_batch"],
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        box_nms_thresh=0.70,
        min_mask_region_area=200,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
    )
    return generator, cfg["input_size"]


def _infer_sam2(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,   # not used — SAM2 masks are already binary
    quality: str = "fast",
) -> list[dict]:
    import cv2

    generator, input_size = _load_sam2(quality)
    img_small, scale      = _resize_for_sam(img_np, input_size)
    orig_H, orig_W        = img_np.shape[:2]

    masks_out = generator.generate(img_small)
    masks_out = [m for m in masks_out if m["predicted_iou"] >= score_thresh]

    detections = []
    for i, m in enumerate(masks_out):
        seg  = m["segmentation"].astype(np.uint8)
        x, y, w, h = m["bbox"]
        mask_full = cv2.resize(
            seg, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)
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
    "maskrcnn_r50v2": _load_maskrcnn_r50v2,
    "sam_vitb":       _load_sam_vitb,
    "sam2_small":     _load_sam2,
}

_INFER = {
    "maskrcnn_r50v2": _infer_maskrcnn_r50v2,
    "sam_vitb":       _infer_sam_vitb,
    "sam2_small":     _infer_sam2,
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
    sam_quality: str = "fast",
) -> tuple[np.ndarray, list[dict]]:
    """
    Unified inference entry point. Result is cached by (image, model, thresholds).
    Changing overlay color or chart metric never re-runs inference.
    """
    img_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    if model_key == "sam_vitb":
        detections = _infer_sam_vitb(img_np, score_thresh, mask_thresh, sam_quality)
    elif model_key == "sam2_small":
        detections = _infer_sam2(img_np, score_thresh, mask_thresh, sam_quality)
    else:
        detections = _INFER[model_key](img_np, score_thresh, mask_thresh)
    return img_np, detections