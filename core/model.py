"""
core/model.py — model registry (Mask R-CNN ResNet-50 FPN v2 · SAM ViT-B · SAM2 Hiera-S)

Detection dict schema returned by all inference functions:
    {
        "id":          int,
        "category":    str,
        "score":       float,
        "box":         [x1, y1, x2, y2],
        "mask_raw":    np.ndarray (H, W) — float16 for Mask R-CNN, uint8 {0,1} for SAM/SAM2,
        "mask_thresh": float,
    }
"""

from __future__ import annotations
import io
import os
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import torch

# module-level torch defaults — CPU inference benefits, GPU unaffected
torch.set_grad_enabled(False)
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceError(RuntimeError):
    """User-facing inference failure that can be shown without a traceback."""

# ── registry ──────────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "Mask R-CNN ResNet-50 v2  (accurate)": "maskrcnn_r50v2",
    "SAM ViT-B  (segment anything)":       "sam_vitb",
    "SAM2 Hiera-S  (better quality)":      "sam2_small",
}

# Max input side for Mask R-CNN; larger images get downscaled then masks upscaled
_MASKRCNN_MAX_SIDE = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Mask R-CNN ResNet-50 FPN v2
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_maskrcnn_r50v2():
    import torchvision  # noqa: F401 - fully initialize torchvision before loading detection ops
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model   = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.eval().to(_DEVICE)
    return model, weights.meta["categories"]


def _resize_max_side(img_np: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    H, W  = img_np.shape[:2]
    scale = min(max_side / max(H, W), 1.0)
    if scale == 1.0:
        return img_np, 1.0
    new_W, new_H = int(W * scale), int(H * scale)
    resized = np.array(
        Image.fromarray(img_np).resize((new_W, new_H), Image.BILINEAR)
    )
    return resized, scale


def _infer_maskrcnn_r50v2(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,
) -> list[dict]:
    import cv2

    model, categories = _load_maskrcnn_r50v2()

    img_small, scale = _resize_max_side(img_np, _MASKRCNN_MAX_SIDE)
    orig_H, orig_W   = img_np.shape[:2]

    tensor = torch.from_numpy(img_small).permute(2, 0, 1).to(_DEVICE, dtype=torch.float32) / 255.0

    use_amp = (_DEVICE == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast(device_type="cpu", enabled=False)
    with autocast_ctx:
        out = model([tensor])[0]

    scores = out["scores"]
    keep   = scores >= score_thresh

    boxes_t  = out["boxes"][keep]
    scores_t = scores[keep]
    labels_t = out["labels"][keep]
    masks_t  = out["masks"][keep]   # (K, 1, h, w)

    boxes  = boxes_t.detach().cpu().numpy()
    scores = scores_t.detach().cpu().numpy()
    labels = labels_t.detach().cpu().numpy()
    masks  = masks_t.detach().cpu().numpy()

    inv = 1.0 / scale
    detections = []
    for i, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
        m = mask[0]
        if scale != 1.0:
            m = cv2.resize(m, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        detections.append({
            "id":          i + 1,
            "category":    categories[int(label)],
            "score":       float(score),
            "box":         [box[0] * inv, box[1] * inv, box[2] * inv, box[3] * inv],
            "mask_raw":    m.astype(np.float16),
            "mask_thresh": mask_thresh,
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# SAM ViT-B — optimised for CPU / weak machines
# ─────────────────────────────────────────────────────────────────────────────
_SAM_PRESETS: dict[str, dict] = {
    "fast":     {"input_size": 256, "points_per_side": 6,  "points_per_batch": 8},
    "balanced": {"input_size": 384, "points_per_side": 10, "points_per_batch": 16},
    "quality":  {"input_size": 512, "points_per_side": 12, "points_per_batch": 32},
}


@st.cache_resource
def _load_sam_vitb_backbone():
    """Load SAM ViT-B checkpoint once. Generator is built per quality preset."""
    from segment_anything import sam_model_registry
    import urllib.request

    weights_dir = Path(os.getenv("OBJECTLENS_WEIGHTS_DIR", Path(__file__).resolve().parent.parent / "weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt = weights_dir / "sam_vit_b_01ec64.pth"
    url  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    if not ckpt.exists():
        with st.spinner("Downloading SAM ViT-B weights (~375 MB) — once only…"):
            urllib.request.urlretrieve(url, ckpt)

    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    sam.to(_DEVICE)
    return sam


def _build_sam_generator(sam, cfg: dict, min_area: int):
    from segment_anything import SamAutomaticMaskGenerator
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=cfg["points_per_side"],
        points_per_batch=cfg["points_per_batch"],
        pred_iou_thresh=0.82,
        stability_score_thresh=0.90,
        box_nms_thresh=0.70,
        min_mask_region_area=min_area,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
    )


def _resize_for_sam(img_np: np.ndarray, input_size: int) -> tuple[np.ndarray, float]:
    return _resize_max_side(img_np, input_size)


def _infer_sam_vitb(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,   # not used — SAM masks are already binary
    quality: str = "fast",
    min_area: int = 200,
) -> list[dict]:
    import cv2

    cfg            = _SAM_PRESETS[quality]
    sam            = _load_sam_vitb_backbone()
    generator      = _build_sam_generator(sam, cfg, min_area)
    img_small, sc  = _resize_for_sam(img_np, cfg["input_size"])
    orig_H, orig_W = img_np.shape[:2]

    masks_out = generator.generate(img_small)
    masks_out = [m for m in masks_out if m["predicted_iou"] >= score_thresh]

    detections = []
    inv = 1.0 / sc
    for i, m in enumerate(masks_out):
        seg  = m["segmentation"].astype(np.uint8)
        x, y, w, h = m["bbox"]
        mask_full = cv2.resize(seg, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
        detections.append({
            "id":          i + 1,
            "category":    "object",
            "score":       float(m["predicted_iou"]),
            "box":         [x * inv, y * inv, (x + w) * inv, (y + h) * inv],
            "mask_raw":    mask_full,           # uint8 {0,1}
            "mask_thresh": 0.5,
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 Hiera-Small
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_sam2_backbone():
    import torchvision  # noqa: F401 - sam2 imports torchvision ops during model construction
    from sam2.build_sam import build_sam2_hf

    torch._dynamo.config.suppress_errors = True
    with st.spinner("Loading SAM2 weights (~183 MB, once only)…"):
        sam2 = build_sam2_hf("facebook/sam2.1-hiera-small", device=_DEVICE)
    return sam2


def _build_sam2_generator(sam2, cfg: dict, min_area: int):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    return SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=cfg["points_per_side"],
        points_per_batch=cfg["points_per_batch"],
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        box_nms_thresh=0.70,
        min_mask_region_area=min_area,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
    )


def _infer_sam2(
    img_np: np.ndarray,
    score_thresh: float,
    mask_thresh: float,
    quality: str = "fast",
    min_area: int = 200,
) -> list[dict]:
    import cv2

    cfg            = _SAM_PRESETS[quality]
    sam2           = _load_sam2_backbone()
    generator      = _build_sam2_generator(sam2, cfg, min_area)
    img_small, sc  = _resize_for_sam(img_np, cfg["input_size"])
    orig_H, orig_W = img_np.shape[:2]

    masks_out = generator.generate(img_small)
    masks_out = [m for m in masks_out if m["predicted_iou"] >= score_thresh]

    detections = []
    inv = 1.0 / sc
    for i, m in enumerate(masks_out):
        seg  = m["segmentation"].astype(np.uint8)
        x, y, w, h = m["bbox"]
        mask_full = cv2.resize(seg, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
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
    "sam_vitb":       _load_sam_vitb_backbone,
    "sam2_small":     _load_sam2_backbone,
}


def load_model(key: str):
    """Pre-warm the selected model. Optional — inference also triggers load."""
    if key not in _LOADERS:
        raise InferenceError(f"Unknown model key: {key}")
    return _LOADERS[key]()


@st.cache_data
def run_inference(
    image_bytes: bytes,
    model_key: str,
    score_thresh: float,
    mask_thresh: float,
    sam_quality: str = "fast",
    min_area: int = 200,
) -> tuple[np.ndarray, list[dict]]:
    """
    Unified inference entry point. Result is cached by (image, model, thresholds).
    Changing overlay color or chart metric never re-runs inference.
    """
    try:
        img_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except Exception as exc:
        raise InferenceError("The uploaded file could not be opened as an image.") from exc

    try:
        if model_key == "sam_vitb":
            detections = _infer_sam_vitb(img_np, score_thresh, mask_thresh, sam_quality, min_area)
        elif model_key == "sam2_small":
            detections = _infer_sam2(img_np, score_thresh, mask_thresh, sam_quality, min_area)
        elif model_key == "maskrcnn_r50v2":
            detections = _infer_maskrcnn_r50v2(img_np, score_thresh, mask_thresh)
        else:
            raise InferenceError(f"Unknown model key: {model_key}")
    except InferenceError:
        raise
    except ModuleNotFoundError as exc:
        raise InferenceError(
            f"The selected model backend is not installed: {exc.name}. Check requirements.txt and redeploy."
        ) from exc
    except OSError as exc:
        raise InferenceError(f"The selected model could not load its weights: {exc}") from exc
    except Exception as exc:
        raise InferenceError(f"Inference failed: {exc}") from exc

    return img_np, detections
