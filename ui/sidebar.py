import streamlit as st
from core.model import MODELS

# model-specific tips shown below the selector
_MODEL_INFO = {
    "maskrcnn101": "Mask R-CNN ResNet-50 FPN v2 · torchvision<br>COCO pretrained · class-aware · fast",
    "sam_vitb":    "SAM ViT-B · Meta AI<br>class-agnostic · segments anything<br>weights ~375 MB, downloaded once<br>runs at 512 px input for CPU speed",
}


def render_sidebar() -> dict:
    """Render sidebar controls and return all parameter values as a dict."""
    with st.sidebar:
        st.markdown('<p class="app-title">StreamLENS</p>', unsafe_allow_html=True)
        st.markdown('<p class="app-sub">small-object instance analyzer</p>', unsafe_allow_html=True)

        # ── input ──────────────────────────────────────────────────────────
        st.markdown('<p class="section-label">Input</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"],
            label_visibility="collapsed",
        )

        # ── model selector ─────────────────────────────────────────────────
        st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
        model_display = st.selectbox(
            "Model", options=list(MODELS.keys()),
            label_visibility="collapsed",
        )
        model_key = MODELS[model_display]

        st.markdown(
            f'<div class="info-box">{_MODEL_INFO[model_key]}</div>',
            unsafe_allow_html=True,
        )

        # ── detection thresholds ───────────────────────────────────────────
        st.markdown('<p class="section-label">Detection</p>', unsafe_allow_html=True)
        score_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)
        mask_thresh  = st.slider("Mask threshold",       0.10, 0.90, 0.50, 0.05,
                                 help="Not used by SAM (its masks are already binary).")

        # ── overlay ────────────────────────────────────────────────────────
        st.markdown('<p class="section-label">Overlay</p>', unsafe_allow_html=True)
        color_by = st.selectbox(
            "Color by",
            options=["instance", "roundness", "aspect_ratio", "eccentricity", "area_px"],
            format_func=lambda x: {
                "instance":     "instance (unique colors)",
                "roundness":    "roundness",
                "aspect_ratio": "aspect ratio",
                "eccentricity": "eccentricity",
                "area_px":      "area (px²)",
            }[x],
        )

        # ── chart metric ───────────────────────────────────────────────────
        st.markdown('<p class="section-label">Chart metric</p>', unsafe_allow_html=True)
        chart_metric = st.selectbox(
            "Distribution of",
            options=["roundness", "aspect_ratio", "eccentricity", "area_px"],
            format_func=lambda x: {
                "roundness":    "roundness",
                "aspect_ratio": "aspect ratio",
                "eccentricity": "eccentricity",
                "area_px":      "area (px²)",
            }[x],
            label_visibility="collapsed",
        )

    return {
        "uploaded":     uploaded,
        "model_key":    model_key,
        "score_thresh": score_thresh,
        "mask_thresh":  mask_thresh,
        "color_by":     color_by,
        "chart_metric": chart_metric,
    }