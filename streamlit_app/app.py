import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

from src.model import AttentionLayer
from src.config import FEATURE_COLS, SEQ_LEN, MODEL_SAVE_PATH, DATA_RAW_DIR
from src.data_loading import load_fd001

# -----------------------------
# Streamlit basic config
# -----------------------------
st.set_page_config(
    page_title="Turbofan Engine RUL Predictor",
    layout="wide",
    page_icon="🛩️",
)

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    return model


@st.cache_resource
def load_scaler():
    """
    Try to load the MinMaxScaler saved from preprocessing.
    Expected path: data/processed/scaler.pkl

    If not found, returns None and the app will fall back to
    using raw values (works for demo, but not ideal).
    """
    scaler_path = Path("data/processed/scaler.pkl")
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


model = load_model()
scaler = load_scaler()

# -----------------------------
# Helper: build last window for an engine
# -----------------------------
def prepare_engine_window(df_engine: pd.DataFrame) -> np.ndarray:
    """
    df_engine: dataframe for a single engine_id, must contain FEATURE_COLS.
    Returns: np.array shape (1, SEQ_LEN, num_features)
    """
    df_engine = df_engine.sort_values("cycle").copy()

    # Apply scaling if scaler is available
    features = df_engine[FEATURE_COLS].values
    if scaler is not None:
        features = scaler.transform(df_engine[FEATURE_COLS])
    else:
        # Fallback: use raw values
        features = df_engine[FEATURE_COLS].values

    if len(features) < SEQ_LEN:
        pad_rows = SEQ_LEN - len(features)
        pad = np.zeros((pad_rows, features.shape[1]))
        window = np.vstack([pad, features])
    else:
        window = features[-SEQ_LEN:]

    return window[np.newaxis, ...]  # (1, SEQ_LEN, num_features)


def classify_health(rul_value: float) -> str:
    if rul_value > 100:
        return "🟢 Healthy"
    elif rul_value > 50:
        return "🟡 Moderate Wear"
    elif rul_value > 20:
        return "🟠 Warning"
    else:
        return "🔴 Critical – Immediate Maintenance"


def export_test_engine_csv(engine_id: int) -> pd.DataFrame:
    """
    Load FD001 test data and return CSV dataframe for a single engine.
    """
    _, test_df, _ = load_fd001(DATA_RAW_DIR)
    df_engine = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")
    return df_engine


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "",
    ["Predict RUL", "Project Details", "Samples / Download"]
)

# -----------------------------
# Hero: rotating turbofan SVG animation (instead of protein NGL)
# -----------------------------
engine_html = """
<div style="width:100%; height:320px; border-radius:14px; overflow:hidden; background:radial-gradient(circle at 20% 20%, #283046 0, #050814 45%, #02030a 100%); display:flex; align-items:center; justify-content:center;">
  <svg width="260" height="260" viewBox="0 0 260 260">
    <defs>
      <radialGradient id="fanGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#e0f4ff"/>
        <stop offset="40%" stop-color="#7aa6ff"/>
        <stop offset="100%" stop-color="#1b2540"/>
      </radialGradient>
      <radialGradient id="hubGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#ffffff"/>
        <stop offset="60%" stop-color="#aaaaaa"/>
        <stop offset="100%" stop-color="#555555"/>
      </radialGradient>
    </defs>
    <!-- Outer ring (engine casing) -->
    <circle cx="130" cy="130" r="120" fill="none" stroke="#6c7aa8" stroke-width="6" opacity="0.9"/>
    <circle cx="130" cy="130" r="110" fill="none" stroke="#182034" stroke-width="18" opacity="0.95"/>

    <!-- Rotating fan group -->
    <g class="fan-blades">
      <circle cx="130" cy="130" r="80" fill="url(#fanGrad)" opacity="0.30"/>
      <!-- blades -->
      <g fill="#e3edf9">
        <polygon points="130,35 140,105 120,105"/>
        <polygon points="225,130 155,140 155,120"/>
        <polygon points="130,225 120,155 140,155"/>
        <polygon points="35,130 105,120 105,140"/>

        <polygon points="190,65 155,110 165,90"/>
        <polygon points="195,195 155,150 175,160"/>
        <polygon points="65,195 110,155 90,165"/>
        <polygon points="65,65 110,105 90,95"/>
      </g>
      <!-- inner gradient ring -->
      <circle cx="130" cy="130" r="55" fill="url(#fanGrad)" opacity="0.65"/>
    </g>

    <!-- Center hub -->
    <circle cx="130" cy="130" r="24" fill="url(#hubGrad)" stroke="#cfd5e6" stroke-width="2"/>
  </svg>
</div>
<style>
@keyframes spinFan {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.fan-blades {
  transform-origin: 130px 130px;
  animation: spinFan 2.8s linear infinite;
}
</style>
"""

if page in ["Predict RUL", "Project Details"]:
    try:
        html(engine_html, height=320, scrolling=False)
    except Exception:
        st.info("Engine animation could not be embedded in this environment.")

# -----------------------------
# Page: Predict RUL
# -----------------------------
if page == "Predict RUL":
    st.header("Turbofan Engine Remaining Useful Life (RUL) Predictor")
    st.write(
        "Upload sensor time-series data for a turbofan engine and the model "
        "will predict the **Remaining Useful Life (RUL)** in cycles."
    )

    st.markdown("#### 1. Upload engine data (CSV)")
    st.write(
        "Expected columns:\n"
        "- `cycle`\n"
        "- `op_setting_1`, `op_setting_2`, `op_setting_3`\n"
        "- `s1` ... `s21`\n"
        "Optionally: `engine_id` (if multiple engines in one file)"
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file with sensor readings",
        type=["csv"],
        help="For example, export one engine's data from FD001 test set."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # Basic cleaning/assumptions
        if "cycle" not in df.columns:
            df["cycle"] = np.arange(1, len(df) + 1)

        if "engine_id" not in df.columns:
            df["engine_id"] = 1

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(
                f"Missing required sensor columns: {missing}. "
                "Please ensure your CSV has all required features."
            )
            st.stop()

        st.markdown("#### 2. Select engine (if multiple)")
        engine_ids = sorted(df["engine_id"].unique())
        selected_engine = st.selectbox("Engine ID", engine_ids)
        df_engine = df[df["engine_id"] == selected_engine].sort_values("cycle")

        st.markdown("#### 3. Preview data")
        st.dataframe(df_engine.head())

        # Prediction button
        if st.button("🔮 Predict RUL for this engine"):
            X_window = prepare_engine_window(df_engine)
            pred_rul = float(model.predict(X_window)[0, 0])

            health = classify_health(pred_rul)

            st.subheader("Prediction Results")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(
                    f"### Predicted Remaining Useful Life: **{pred_rul:.1f} cycles**"
                )
                st.markdown(f"### Health Status: {health}")
                if scaler is None:
                    st.warning(
                        "Note: scaler.pkl not found, using raw values for demo. "
                        "For exact alignment with training, save the MinMaxScaler "
                        "in preprocessing and place it at `data/processed/scaler.pkl`."
                    )
            with col2:
                # Simple "health bar"
                max_display_rul = 150
                pct = max(0.0, min(1.0, pred_rul / max_display_rul))
                st.markdown("**Relative Health (0–150 cycles)**")
                st.progress(pct)

            # Trends
            st.markdown("#### 4. Sensor Trends")
            st.write("Selected sensors over cycles for this engine:")

            sensor_cols = [c for c in df_engine.columns if c.startswith("s")]
            # Show first 3–4 sensors as line charts
            for s in sensor_cols[:4]:
                st.line_chart(
                    df_engine[["cycle", s]].set_index("cycle"),
                    height=160
                )

            # RUL progression (approx) over time: we can also predict at multiple cycles
            st.markdown("#### 5. Approximate RUL trajectory (optional)")
            with st.expander("Show RUL vs cycle curve (approximate)", expanded=False):
                # Sample a few points along the timeline
                steps = np.linspace(SEQ_LEN, len(df_engine), num=min(30, len(df_engine))).astype(int)
                rul_preds = []
                for t in steps:
                    sub_df = df_engine.iloc[:t]
                    Xw = prepare_engine_window(sub_df)
                    rul_preds.append(float(model.predict(Xw)[0, 0]))
                traj_df = pd.DataFrame({"cycle": steps, "predicted_RUL": rul_preds})
                st.line_chart(traj_df.set_index("cycle"))

    else:
        st.info("Upload a CSV file to begin RUL prediction.")


# -----------------------------
# Page: Project Details
# -----------------------------
if page == "Project Details":
    st.header("Project Details & Results")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Problem Statement")
        st.write(
            "- **Goal:** Predict the Remaining Useful Life (RUL) of turbofan engines "
            "using multivariate sensor time-series data.\n"
            "- **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Dataset, subset **FD001**."
        )

        st.subheader("Data & Preprocessing")
        st.write(
            "- Each engine has cycles with 21 sensors + 3 operating settings.\n"
            "- Training engines run until failure; RUL is computed as `(max_cycle - current_cycle)` and capped at 125.\n"
            "- Features normalized using MinMaxScaler fitted on training set.\n"
            "- Input to the model is a sliding window of **50 cycles** per engine."
        )

        st.subheader("Model Architecture")
        st.write(
            "- 1D CNN layers for local temporal feature extraction.\n"
            "- Bidirectional LSTM layers for sequence modeling.\n"
            "- Custom Attention layer over time dimension.\n"
            "- Dense regression head predicting scalar RUL.\n"
            "- Optimizer: Adam, Loss: MAE."
        )

    with col2:
        st.subheader("Training & Evaluation")
        st.write("- Epochs: up to 100 with early stopping & LR scheduling.")
        st.write("- Validation MAE ≈ **6.18** cycles.")
        st.write("- Final Test (FD001):")
        st.markdown(
            """
            - **MAE:** ~**11.8** cycles  
            - **RMSE:** ~**15.7**  
            - **R²:** ~**0.86**
            """
        )

    st.markdown("### Training & Evaluation Plots")
    figs = {
        "Training & Validation MAE": "reports/figures/loss_curve.png",
        "Predicted vs True RUL (Test)": "reports/figures/pred_vs_true.png",
        "RUL Error Distribution": "reports/figures/error_histogram.png",
    }
    for title, path in figs.items():
        p = Path(path)
        if p.exists():
            st.subheader(title)
            st.image(str(p), width=1000)
        else:
            st.info(f"{title} — figure not found at `{path}` (generate via training/evaluation scripts).")


# -----------------------------
# Page: Samples / Download
# -----------------------------
if page == "Samples / Download":
    st.header("Sample Engines & Downloads")

    st.markdown("### Download sample engine CSV (FD001 Test Set)")

    engine_id_input = st.number_input(
        "Enter Engine ID (1–100)",
        min_value=1,
        max_value=100,
        value=1,
        step=1
    )

    if st.button("📥 Generate CSV for this Engine"):
        try:
            df_sample = export_test_engine_csv(engine_id_input)

            if df_sample.empty:
                st.error("No data found for this engine ID.")
            else:
                csv_bytes = df_sample.to_csv(index=False).encode("utf-8")

                st.success(f"Engine {engine_id_input} CSV ready!")
                st.download_button(
                    label="⬇️ Download Engine CSV",
                    data=csv_bytes,
                    file_name=f"engine_{engine_id_input}_fd001_test.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Failed to generate CSV: {e}")

    st.write(
        "You can generate sample CSVs from the NASA FD001 test set using your preprocessing "
        "scripts and then reuse them here for quick demos.\n"
        "Below is a simple template for a single-engine CSV:"
    )

    sample_code = """\
cycle,op_setting_1,op_setting_2,op_setting_3,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21
1,0.5,0.1,0.0,489.0,606.0,1400.0,15.0,390.0,555.0,2386.0,9046.0,1.3,47.0,522.0,2388.0,8135.0,8.4,0.03,390.0,2380.0,100.0,39.0,23.0,391.0
2,0.5,0.1,0.0,488.7,606.1,1401.2,15.1,389.5,554.7,2385.5,9048.3,1.3,47.1,522.5,2388.5,8136.2,8.4,0.03,390.2,2380.5,100.1,38.9,22.9,390.8
...
"""
    st.code(sample_code, language="text")

    st.write(
        "You can also expose downloads for your trained model and any supporting "
        "artifacts if you want to share this as a reproducible package."
    )

    model_path = Path(MODEL_SAVE_PATH)
    if model_path.exists():
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        st.download_button(
            "Download trained model (.keras)",
            data=model_bytes,
            file_name="best_model.keras"
        )
    else:
        st.info(f"Trained model not found at `{MODEL_SAVE_PATH}`.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with 🛩️ Turbofan RUL · CNN + BiLSTM + Attention · Streamlit")
