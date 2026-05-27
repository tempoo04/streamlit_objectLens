import streamlit as st


def inject_styles() -> None:
    """Inject global CSS for the ObjectLens Streamlit app."""
    st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

  [data-testid="metric-container"] {
    background: #f8f9fa;
    border: 0.5px solid #e0e0e0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
  }
  [data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 500 !important;
  }
  [data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 0.5px solid #e8e8e8;
  }
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stFileUploader label {
    font-size: 12px !important;
    color: #666 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .stDownloadButton button {
    border: 1px solid #2563eb !important;
    color: #2563eb !important;
    background: transparent !important;
    font-size: 13px !important;
  }
  .stDownloadButton button:hover { background: #eff6ff !important; }

  .section-label {
    font-size: 11px; font-weight: 500; color: #999;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 6px; margin-top: 4px;
  }
  .app-title {
    font-size: 22px; font-weight: 500; color: #1a1a1a;
    letter-spacing: -0.3px;
  }
  .app-sub { font-size: 13px; color: #999; margin-top: -4px; margin-bottom: 16px; }
  .info-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 8px 12px;
    font-size: 12px; color: #334155; margin-top: 8px;
  }
</style>
""", unsafe_allow_html=True)
