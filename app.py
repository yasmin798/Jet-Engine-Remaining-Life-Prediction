import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Engine RUL Predictor", page_icon="✈️", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("rul_pipeline.joblib")
    with open("meta.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()
SENSORS, FEATURES = meta["sensor_cols"], meta["feature_cols"]
WINDOW, CAP = meta["window"], meta["rul_cap"]

def add_rolling_features(df, sensor_cols, window):
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")[sensor_cols]
    r_mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    r_std = g.rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    r_mean.columns = [f"{c}_mean" for c in sensor_cols]
    r_std.columns = [f"{c}_std" for c in sensor_cols]
    return pd.concat([df, r_mean, r_std], axis=1)

st.title("✈️ Jet Engine Remaining-Life Predictor")
st.write("Predicts how many flight cycles an engine has left from its sensor history, "
         "and raises a maintenance alert when it is close to failure. "
         "*Trained on NASA C-MAPSS (FD001) simulation data. Demo only, not for real maintenance decisions.*")

source = st.sidebar.radio("Data source", ["Demo engines", "Upload CSV"])
threshold = st.sidebar.slider("Maintenance alert threshold (cycles)", 10, 80, 40)

df = None
if source == "Demo engines":
    df = pd.read_csv("demo_engines.csv")
else:
    up = st.sidebar.file_uploader("Upload sensor history (CSV)", type="csv")
    st.sidebar.caption("Needs columns: unit, cycle, s1 ... s21 (same format as the demo file).")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            st.error("Could not read this file as a CSV.")

if df is not None:
    missing = [c for c in ["unit", "cycle"] + SENSORS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    if df[SENSORS].isna().any().any():
        st.error("The file contains missing sensor values. Please fix them and re-upload.")
        st.stop()

    unit = st.selectbox("Choose an engine", sorted(df["unit"].unique()))
    eng = df[df["unit"] == unit]
    feats = add_rolling_features(eng, SENSORS, WINDOW)
    feats["Predicted RUL"] = np.clip(model.predict(feats[FEATURES]), 0, CAP)

    current = float(feats["Predicted RUL"].iloc[-1])
    c1, c2, c3 = st.columns(3)
    c1.metric("Cycles recorded", int(feats["cycle"].max()))
    c2.metric("Predicted remaining life (cycles)", f"{current:.0f}")
    c3.metric("Status", "🔴 MAINTENANCE NEEDED" if current < threshold else "🟢 OK")

    if current < threshold:
        st.error(f"Predicted RUL ({current:.0f}) is below the alert threshold ({threshold}). Schedule maintenance.")
    else:
        st.success("Engine is predicted to be healthy for now.")

    st.subheader("Degradation chart")
    st.line_chart(feats.set_index("cycle")["Predicted RUL"])

    st.subheader("Sensor trend")
    pick = st.selectbox("Sensor", SENSORS, index=min(5, len(SENSORS) - 1))
    st.line_chart(feats.set_index("cycle")[pick])
else:
    st.info("Pick a data source in the sidebar to begin.")