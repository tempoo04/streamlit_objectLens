import streamlit as st


def render_sidebar() -> dict:
    """
    Renders the sidebar and returns a dict of all user-controlled parameters.
    """
    with st.sidebar:
        st.markdown('<p class="app-title">ObjectLens</p>', unsafe_allow_html=True)
        st.markdown('<p class="app-sub">small-object instance analyzer</p>', unsafe_allow_html=True)

        st.markdown('<p class="section-label">Input</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"],
            label_visibility="collapsed",
        )

        st.markdown('<p class="section-label">Detection</p>', unsafe_allow_html=True)
        score_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)
        mask_thresh  = st.slider("Mask threshold",       0.10, 0.90, 0.50, 0.05)

        st.markdown('<p class="section-label">Overlay</p>', unsafe_allow_html=True)
        color_by = st.selectbox(
            "Color by",
            options=["instance", "roundness", "aspect_ratio", "eccentricity", "area_px"],
            format_func=lambda x: {
                "instance":    "instance (unique colors)",
                "roundness":   "roundness",
                "aspect_ratio":"aspect ratio",
                "eccentricity":"eccentricity",
                "area_px":     "area (px²)",
            }[x],
        )

        st.markdown('<p class="section-label">Chart metric</p>', unsafe_allow_html=True)
        chart_metric = st.selectbox(
            "Distribution of",
            options=["roundness", "aspect_ratio", "eccentricity", "area_px"],
            format_func=lambda x: {
                "roundness":   "roundness",
                "aspect_ratio":"aspect ratio",
                "eccentricity":"eccentricity",
                "area_px":     "area (px²)",
            }[x],
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="info-box">Mask R-CNN · ResNet-50 FPN<br>pretrained on COCO (80 classes)</div>',
            unsafe_allow_html=True,
        )

    return {
        "uploaded":     uploaded,
        "score_thresh": score_thresh,
        "mask_thresh":  mask_thresh,
        "color_by":     color_by,
        "chart_metric": chart_metric,
    }
