import streamlit as st

from core.features import compute_features
from core.model import InferenceError, run_inference
from core.overlay import render_overlay
from ui.results import render_histogram, render_metrics, render_table_and_download
from ui.sidebar import render_sidebar
from ui.styles import inject_styles


MAX_UPLOAD_MB = 20


st.set_page_config(
    page_title="ObjectLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_empty_state() -> None:
    st.markdown("### Upload an image to begin")
    st.markdown(
        "ObjectLens detects individual objects, segments them, and computes shape "
        "descriptors for each instance: roundness, aspect ratio, eccentricity, area, "
        "and perimeter. It works best with clear photos of separated objects such as "
        "coins, seeds, pills, fruit, screws, or lab samples."
    )
    st.info("Use the sidebar to upload an image and choose an inference model.")


def render_results(img_np, detections, params: dict) -> None:
    df, objects = compute_features(detections)

    if df.empty:
        st.warning("No objects were detected above the current threshold. Try lowering confidence or using a clearer image.")
        return

    render_metrics(df)

    col_img, col_chart = st.columns(2)
    with col_img:
        st.markdown('<p class="section-label">Annotated image</p>', unsafe_allow_html=True)
        overlay = render_overlay(img_np, objects, params["color_by"])
        st.image(overlay, use_container_width=True)

    with col_chart:
        metric = params["chart_metric"]
        label = metric.replace("_", " ")
        st.markdown(f'<p class="section-label">{label} distribution</p>', unsafe_allow_html=True)
        render_histogram(df, metric)

    render_table_and_download(df)


def main() -> None:
    inject_styles()
    params = render_sidebar()
    uploaded = params["uploaded"]

    if uploaded is None:
        render_empty_state()
        return

    image_bytes = uploaded.getvalue()
    if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Uploaded image is larger than {MAX_UPLOAD_MB} MB. Use a smaller file for reliable deployment performance.")
        return

    try:
        with st.spinner("Running inference..."):
            img_np, detections = run_inference(
                image_bytes,
                params["model_key"],
                params["score_thresh"],
                params["mask_thresh"],
                params["sam_quality"],
            )
    except InferenceError as exc:
        st.error(str(exc))
        st.caption("The app is still running. Adjust the model or image and try again.")
        return

    render_results(img_np, detections, params)


if __name__ == "__main__":
    main()
