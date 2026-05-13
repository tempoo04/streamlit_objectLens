import streamlit as st

st.set_page_config(
    page_title="ObjectLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.styles   import inject_styles
from ui.sidebar  import render_sidebar
from ui.results  import render_metrics, render_histogram, render_table_and_download
from core.model    import run_inference
from core.features import compute_features
from core.overlay  import render_overlay

inject_styles()
params   = render_sidebar()
uploaded = params["uploaded"]

if uploaded is None:
    st.markdown("### Upload an image to begin")
    st.markdown(
        "ObjectLens uses instance segmentation models to detect and segment individual objects, "
        "then computes shape descriptors — roundness, aspect ratio, eccentricity — for every instance. "
        "Choose between **Mask R-CNN ResNet-101**, **SAM ViT-B**, or **YOLO11-seg** from the sidebar. "
        "Works best with clearly separated small objects: coins, seeds, pills, fruits, screws, etc."
    )
    st.info("⬅  Upload an image and select a model in the sidebar to get started.")
else:
    image_bytes = uploaded.read()

    with st.spinner(f"Running inference…"):
        img_np, detections = run_inference(
            image_bytes,
            params["model_key"],
            params["score_thresh"],
            params["mask_thresh"],
        )

    df, objects = compute_features(detections)

    if df.empty:
        st.warning("No objects detected above the threshold. Try lowering the confidence slider.")
    else:
        render_metrics(df)
        st.markdown("")

        col_img, col_chart = st.columns(2)

        with col_img:
            st.markdown('<p class="section-label">Annotated image</p>', unsafe_allow_html=True)
            overlay = render_overlay(img_np, objects, params["color_by"])
            st.image(overlay, use_column_width=True)

        with col_chart:
            metric = params["chart_metric"]
            st.markdown(
                f'<p class="section-label">{metric.replace("_", " ")} distribution</p>',
                unsafe_allow_html=True,
            )
            render_histogram(df, metric)

        render_table_and_download(df)