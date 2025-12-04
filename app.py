import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import fastf1
import re
import joblib
import xgboost as xgb
import json
import sys
import types

# -----------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------
st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

ARTIFACTS = Path("artifacts")
MODEL_INPUT = Path("streamlit_input.csv")
PRED_OUT = ARTIFACTS / "streamlit_scored_preds.csv"

# -----------------------------------------------------------
# Session state defaults
# Ensure common keys exist so users who skip the "Race Settings"
# step don't trigger KeyError when reading `st.session_state`.
# -----------------------------------------------------------
if "air_temp" not in st.session_state:
    st.session_state["air_temp"] = 28.0
if "track_temp" not in st.session_state:
    st.session_state["track_temp"] = 33.0
if "humidity" not in st.session_state:
    st.session_state["humidity"] = 55
if "pressure" not in st.session_state:
    st.session_state["pressure"] = 1012
if "wind_speed" not in st.session_state:
    st.session_state["wind_speed"] = 2.5
if "wind_dir" not in st.session_state:
    st.session_state["wind_dir"] = 180
if "races_prior" not in st.session_state:
    st.session_state["races_prior"] = 12
if "tire" not in st.session_state:
    st.session_state["tire"] = "SOFT"
if "rain" not in st.session_state:
    st.session_state["rain"] = False


# -----------------------------------------------------------
# Name Splicer / Cleaner
# -----------------------------------------------------------
def clean_driver_name(name: str) -> str:
    if not isinstance(name, str):
        return name

    s = name.strip()

    # Split camelCase (MaxVerstappen → Max Verstappen)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)

    # Remove trailing all-caps codes (VER, NOR, HAM)
    s = re.sub(r"\b[A-Z]{2,4}$", "", s).strip()

    # Collapse spaces
    s = re.sub(r"\s+", " ", s)

    return s


# -----------------------------------------------------------
# FastF1 Driver Standings
# -----------------------------------------------------------
def get_fastf1_driver_points(season=2025):
    fastf1.Cache.enable_cache("fastf1_cache")

    standings = fastf1.api.driver_standings(season)

    data = []
    for entry in standings:
        fname = entry["Driver"]["givenName"]
        lname = entry["Driver"]["familyName"]
        name = f"{fname} {lname}"

        team = entry["Constructors"][0]["name"]
        pts = float(entry["points"])

        data.append({"Driver": name, "Team": team, "Points": pts})

    return pd.DataFrame(data)


def merge_points(df, season_df):
    """Merge FastF1 standings into uploaded grid."""
    merged = df.merge(season_df, on="Driver", how="left")

    # If Points missing → fallback to neutral
    max_pts = merged["Points"].max()
    if pd.isna(max_pts) or max_pts <= 0:
        merged["PointsProp"] = 0.5
    else:
        merged["PointsProp"] = merged["Points"] / max_pts

    return merged


# -----------------------------------------------------------
# Model Loading Functions
# -----------------------------------------------------------
def load_preprocessor():
    p = ARTIFACTS / "preprocessing_pipeline.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {p}")
    try:
        import sys
        import types
        mod_name = 'sklearn.compose._column_transformer'
        if mod_name not in sys.modules:
            try:
                import importlib
                sys.modules[mod_name] = importlib.import_module(mod_name)
            except Exception:
                sys.modules[mod_name] = types.ModuleType(mod_name)
        mod = sys.modules[mod_name]
        if not hasattr(mod, '_RemainderColsList'):
            class _RemainderColsList(list):
                pass
            setattr(mod, '_RemainderColsList', _RemainderColsList)
    except Exception:
        pass

    return joblib.load(p)


def load_ensemble_models():
    models_dir = ARTIFACTS / "ensemble_fold_models_remove_lapped"
    if not (models_dir.exists() and any(models_dir.iterdir())):
        models_dir = ARTIFACTS / "ensemble_fold_models"
    if models_dir.exists() and any(models_dir.iterdir()):
        models = []
        for f in sorted(models_dir.iterdir()):
            if f.suffix in ('.joblib', '.pkl'):
                models.append(joblib.load(f))
        if models:
            return models
    for fname in ("xgb_minimal_cv_randomized.joblib", "xgb_minimal_cv.joblib", "xgb_minimal.joblib"):
        candidate = ARTIFACTS / fname
        if candidate.exists():
            return [joblib.load(candidate)]
    raise FileNotFoundError("No model artifacts found in artifacts/")


def apply_calibration(preds: np.ndarray) -> np.ndarray:
    calib_path = ARTIFACTS / "linear_calibration_remove_lapped.joblib"
    if not calib_path.exists():
        calib_path = ARTIFACTS / "linear_calibration.joblib"
    if calib_path.exists():
        lr = joblib.load(calib_path)
        return lr.predict(preds.reshape(-1, 1))
    return preds


def score(input_csv: Path, output_csv: Path):
    preproc = load_preprocessor()
    models = load_ensemble_models()

    df = pd.read_csv(input_csv)
    try:
        from artifacts.input_normalize import normalize_minimal_features
        df = normalize_minimal_features(df)
    except Exception:
        pass
    minimal_cols = ['GridPosition','AvgQualiTime','weather_tire_cluster','SOFT','MEDIUM','HARD','INTERMEDIATE','WET','races_prior_this_season','Rain','PointsProp','Status']
    missing = [c for c in minimal_cols if c not in df.columns]
    if missing:
        print(f"Warning: input CSV is missing expected columns: {missing}.")
    initial_n = len(df)
    if 'Status' in df.columns:
        if (df['Status'] == 'Lapped').any():
            df = df[df['Status'] != 'Lapped']
    else:
        for col in ['Status__Lapped', 'Status__Lapped.0']:
            if col in df.columns:
                if (df[col] == 1).any():
                    df = df[df[col] != 1]
                break
    if len(df) == 0:
        raise ValueError('No rows left to score after dropping lapped drivers.')
    idx = df.index

    try:
        X = preproc.transform(df)
    except Exception as e:
        minimal_features = [
            'GridPosition','AvgQualiTime','weather_tire_cluster',
            'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
            'races_prior_this_season','Rain','PointsProp'
        ]
        provided = df.copy()
        missing = [c for c in minimal_features if c not in provided.columns]
        if missing:
            raise ValueError(f"Missing minimal features: {missing}")
        X = provided[minimal_features].values

    preds = []
    for model in models:
        try:
            pred = model.predict(X)
        except TypeError as e:
            if "DMatrix" in str(e):
                pred = model.predict(xgb.DMatrix(X))
            else:
                raise
        preds.append(pred)
    avg_pred = np.mean(preds, axis=0)
    calibrated = apply_calibration(avg_pred)

    result = pd.DataFrame({'index': idx, 'prediction': calibrated})
    result.to_csv(output_csv, index=False)


# -----------------------------------------------------------
# Theme Styling
# -----------------------------------------------------------
st.markdown("""
<style>
    body, .stApp {
        background-color: #0d0d0f !important;
        color: #f5f5f5 !important;
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #111115 !important;
        border-right: 1px solid #222;
    }
    .stButton>button {
        background-color: #e10600 !important;
        color: white !important;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #b10500 !important;
    }
    .result-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #1a1a1d;
        border: 2px solid #333;
        text-align: center;
        font-size: 1.2rem;
    }
    .podium-gold { border-color: #FFD700; }
    .podium-silver { border-color: #C0C0C0; }
    .podium-bronze { border-color: #CD7F32; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------
st.sidebar.title("🏎️ F1 Predictor")
page = st.sidebar.radio("Navigation", ["Upload Grid", "Race Settings", "Run Prediction"])


# -----------------------------------------------------------
# 1 — Upload Grid
# -----------------------------------------------------------
if page == "Upload Grid":
    st.header("📤 Upload Qualifying Grid CSV")

    uploaded = st.file_uploader("Upload qualifying_results.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        # Clean driver names
        df["Driver"] = df["Driver"].astype(str).apply(clean_driver_name)

        st.success("Grid Loaded")
        st.dataframe(df, use_container_width=True)

        df.to_csv("qualifying_results_uploaded.csv", index=False)
        st.caption("Saved as qualifying_results_uploaded.csv")


# -----------------------------------------------------------
# 2 — Race Settings
# -----------------------------------------------------------
elif page == "Race Settings":
    st.header("⚙️ Race-Day Settings")

    colA, colB, colC = st.columns(3)

    with colA:
        st.session_state["air_temp"] = st.number_input("Air Temperature (°C)", 10.0, 50.0, 28.0)
        st.session_state["track_temp"] = st.number_input("Track Temperature (°C)", 0.0, 60.0, 33.0)
        st.session_state["humidity"] = st.slider("Humidity (%)", 0, 100, 55)

    with colB:
        st.session_state["pressure"] = st.number_input("Air Pressure (hPa)", 950, 1100, 1012)
        st.session_state["wind_speed"] = st.number_input("Wind Speed (m/s)", 0.0, 15.0, 2.5)
        st.session_state["wind_dir"] = st.slider("Wind Direction (°)", 0, 360, 180)

    with colC:
        st.session_state["races_prior"] = st.number_input("Completed Races", 0, 25, 12)
        st.session_state["tire"] = st.selectbox("Starting Tire", ["SOFT", "MEDIUM", "HARD"])
        st.session_state["rain"] = st.checkbox("Raining?")

    st.info("Settings saved. Navigate to Run Prediction.")


# -----------------------------------------------------------
# 3 — Run Prediction
# -----------------------------------------------------------
elif page == "Run Prediction":
    st.header("🏁 Run Prediction Engine")

    grid_path = Path("qualifying_results_uploaded.csv")
    xtrain_path = ARTIFACTS / "X_train.csv"
    scorer_path = ARTIFACTS / "score_model.py"

    # Validate required files
    if not grid_path.exists():
        st.error("Upload a qualifying grid CSV first.")
        st.stop()

    df = pd.read_csv(grid_path)
    df["Driver"] = df["Driver"].astype(str).apply(clean_driver_name)

    st.subheader("📋 Loaded Grid")
    st.dataframe(df, use_container_width=True)

    # ----------------------------------------------------
    # ⭐ FastF1 Live Points Integration
    # ----------------------------------------------------
    st.subheader("🔢 Auto-Calculating PointsProportion (via FastF1)")

    try:
        season_pts = get_fastf1_driver_points(2025)
        df = merge_points(df, season_pts)
        st.success("FastF1 points loaded.")
    except Exception as e:
        st.warning(f"FastF1 unavailable → using neutral PointsProp defaults. ({e})")
        df["PointsProp"] = 0.5

    # ----------------------------------------------------
    # UI Slider for Tweaking PointsProp
    # ----------------------------------------------------
    st.subheader("🎛️ Adjust PointsProp")

    new_vals = []
    for i, row in df.iterrows():
        suggested = round(row["PointsProp"], 3)

        st.markdown(
            f"""
            <div style='padding:10px; margin-bottom:8px; background-color:#1a1a1d; 
                         border:1px solid #333; border-radius:8px;'>
                <strong>{row['Driver']}</strong><br>
                <span style='font-size:0.85rem; color:#bbb;'>Suggested: {suggested}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        edited = st.slider(
            f"prop_{i}",
            0.0, 1.0, float(suggested),
            step=0.01,
            key=f"ppslider_{i}"
        )
        new_vals.append(edited)

    df["PointsProp"] = new_vals


    # ----------------------------------------------------
    # Build Model Input
    # ----------------------------------------------------
    if not xtrain_path.exists():
        st.error("Missing artifacts/X_train.csv")
        st.stop()

    X_train = pd.read_csv(xtrain_path)
    expected_cols = X_train.columns.tolist()

    model_input = pd.DataFrame(0, index=range(len(df)), columns=expected_cols)

    # Fill known fields
    settings = {
        "GridPosition": df["Pos."].astype(int),
        "AirTemp_C": st.session_state["air_temp"],
        "TrackTemp_C": st.session_state["track_temp"],
        "Humidity_%": st.session_state["humidity"],
        "Pressure_hPa": st.session_state["pressure"],
        "WindSpeed_mps": st.session_state["wind_speed"],
        "WindDirection_deg": st.session_state["wind_dir"],
        "races_prior_this_season": st.session_state["races_prior"],
        "Rain": 1 if st.session_state["rain"] else 0,

        "Driver": df["Driver"],
        "TeamName": df["Team"],
        "PointsProp": df["PointsProp"],

        "SOFT": 1 if st.session_state["tire"] == "SOFT" else 0,
        "MEDIUM": 1 if st.session_state["tire"] == "MEDIUM" else 0,
        "HARD": 1 if st.session_state["tire"] == "HARD" else 0,
    }

    for k, v in settings.items():
        if k in model_input.columns:
            model_input[k] = v

    # Save model input
    model_input.to_csv(MODEL_INPUT, index=False)


    # ----------------------------------------------------
    # Run Prediction
    # ----------------------------------------------------
    if st.button("Run Prediction"):
        with st.spinner("Predicting outcome…"):
            score(MODEL_INPUT, PRED_OUT)

        scored = pd.read_csv(PRED_OUT)
        preds = scored["prediction"]

        final = pd.DataFrame({
            "Driver": df["Driver"],
            "PredictedDeviation": preds
        }).sort_values("PredictedDeviation").reset_index(drop=True)

        final["PredictedPosition"] = final.index + 1

        st.success("Prediction complete!")

        # Podium Display
        st.subheader("🏆 Podium Predictions")
        gold, silver, bronze = final.iloc[0], final.iloc[1], final.iloc[2]

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='result-card podium-silver'><h3>2nd</h3>{silver['Driver']}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='result-card podium-gold'><h3>1st</h3>{gold['Driver']}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='result-card podium-bronze'><h3>3rd</h3>{bronze['Driver']}</div>", unsafe_allow_html=True)

        st.subheader("📊 Full Prediction Table")
        st.dataframe(final, use_container_width=True)

        st.download_button(
            "Download Predictions CSV",
            final.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
