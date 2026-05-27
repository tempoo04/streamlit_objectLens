import streamlit as st
import plotly.express as px
import pandas as pd


_METRIC_LABELS = {
    "roundness":    "Roundness (0–1)",
    "aspect_ratio": "Aspect ratio",
    "eccentricity": "Eccentricity (0–1)",
    "area_px":      "Area (px²)",
}

_DISPLAY_COLS = [
    "id", "category", "score",
    "area_px", "roundness", "aspect_ratio", "eccentricity", "perimeter_px",
]


def render_metrics(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects detected",  len(df))
    c2.metric("Avg roundness",     f"{df['roundness'].mean():.2f}")
    c3.metric("Avg aspect ratio",  f"{df['aspect_ratio'].mean():.2f}")
    c4.metric("Median area",       f"{int(df['area_px'].median())} px²")


def render_histogram(df: pd.DataFrame, metric: str) -> None:
    fig = px.histogram(
        df, x=metric, nbins=20,
        color_discrete_sequence=["#378ADD"],
        labels={metric: _METRIC_LABELS.get(metric, metric)},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#666"),
        showlegend=False,
        bargap=0.08,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False, title="count"),
        height=260,
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_table_and_download(df: pd.DataFrame) -> None:
    st.markdown('<p class="section-label">Per-object results</p>', unsafe_allow_html=True)
    st.dataframe(df[_DISPLAY_COLS], use_container_width=True, hide_index=True)

    csv = df[_DISPLAY_COLS].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results.csv",
        data=csv,
        file_name="objectlens_results.csv",
        mime="text/csv",
    )
