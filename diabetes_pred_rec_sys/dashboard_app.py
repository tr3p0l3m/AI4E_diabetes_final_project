"""Streamlit dashboard application for diabetes risk predictions."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from ml_utils import BASE_FEATURES, predict_diabetes

DB_PATH = Path(__file__).parent / "data" / "predictions.db"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                gender TEXT NOT NULL,
                age REAL,
                urea REAL,
                cr REAL,
                hba1c REAL,
                chol REAL,
                tg REAL,
                hdl REAL,
                ldl REAL,
                vldl REAL,
                bmi REAL,
                predicted_class TEXT,
                probabilities TEXT,
                model_accuracy REAL,
                model_trained TEXT
            )
            """
        )
        conn.commit()


def log_prediction(inputs: Dict[str, Any], result: Dict[str, Any]) -> None:
    probabilities = json.dumps(result.get("probabilities", {}))
    metadata = result.get("model_metadata", {})

    payload = (
        datetime.utcnow().isoformat(),
        inputs["gender"],
        inputs["age"],
        inputs["urea"],
        inputs["cr"],
        inputs["hba1c"],
        inputs["chol"],
        inputs["tg"],
        inputs["hdl"],
        inputs["ldl"],
        inputs["vldl"],
        inputs["bmi"],
        result.get("predicted_class") or result.get("prediction"),
        probabilities,
        metadata.get("test_accuracy"),
        metadata.get("trained_date"),
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at, gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi,
                predicted_class, probabilities, model_accuracy, model_trained
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            payload,
        )
        conn.commit()


def fetch_predictions() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC", conn)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def render_prediction_section() -> None:
    st.subheader("Interactive Predictor")
    st.markdown(
        "Provide lab measurements below. Once submitted, the prediction is stored and "
        "made available to the analytics dashboard."
    )

    default_values = {
        "Gender": "Male",
        "Age": 45,
        "Urea": 4.5,
        "Cr": 60.0,
        "HbA1c": 5.8,
        "Chol": 4.7,
        "TG": 1.2,
        "HDL": 1.2,
        "LDL": 2.5,
        "VLDL": 0.5,
        "BMI": 25.0,
    }

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female"], index=0)
        age = c2.number_input("Age", min_value=1, max_value=120, value=default_values["Age"])
        bmi = c3.number_input("BMI", min_value=10.0, max_value=60.0, value=default_values["BMI"], step=0.1)

        c4, c5, c6 = st.columns(3)
        urea = c4.number_input("Urea (mmol/L)", min_value=0.1, max_value=25.0, value=default_values["Urea"], step=0.1)
        cr = c5.number_input("Creatinine (µmol/L)", min_value=10.0, max_value=200.0, value=default_values["Cr"], step=1.0)
        hba1c = c6.number_input("HbA1c (%)", min_value=3.0, max_value=18.0, value=default_values["HbA1c"], step=0.1)

        c7, c8, c9 = st.columns(3)
        chol = c7.number_input("Cholesterol (mmol/L)", min_value=2.0, max_value=12.0, value=default_values["Chol"], step=0.1)
        tg = c8.number_input("Triglycerides (mmol/L)", min_value=0.2, max_value=10.0, value=default_values["TG"], step=0.1)
        hdl = c9.number_input("HDL (mmol/L)", min_value=0.2, max_value=5.0, value=default_values["HDL"], step=0.1)

        c10, c11 = st.columns(2)
        ldl = c10.number_input("LDL (mmol/L)", min_value=0.1, max_value=8.0, value=default_values["LDL"], step=0.1)
        vldl = c11.number_input("VLDL (mmol/L)", min_value=0.1, max_value=4.0, value=default_values["VLDL"], step=0.1)

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        inputs = {
            "gender": gender,
            "age": age,
            "urea": urea,
            "cr": cr,
            "hba1c": hba1c,
            "chol": chol,
            "tg": tg,
            "hdl": hdl,
            "ldl": ldl,
            "vldl": vldl,
            "bmi": bmi,
        }
        result = predict_diabetes(**inputs)
        log_prediction(inputs, result)

        st.success(f"Predicted class: {result.get('predicted_class', result.get('prediction'))}")
        st.json(result)


def _probabilities_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "probabilities" not in df:
        return pd.DataFrame()
    try:
        probs = df["probabilities"].apply(lambda x: pd.Series(json.loads(x)))
    except Exception:
        return pd.DataFrame()
    probs.index = df.index
    return probs


def render_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Live Analytics Dashboard")
    if df.empty:
        st.info("No predictions logged yet. Submit a prediction to unlock analytics.")
        return

    total_predictions = int(df.shape[0])
    latest_prediction = df.iloc[0]["predicted_class"]
    avg_bmi = df["bmi"].mean()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Predictions", total_predictions)
    metric_cols[1].metric("Most Recent Class", latest_prediction)
    metric_cols[2].metric("Avg BMI", f"{avg_bmi:.1f}")

    class_counts = df["predicted_class"].value_counts().sort_index()
    st.markdown("### Class Distribution")
    st.bar_chart(class_counts)

    df["created_date"] = df["created_at"].dt.date
    timeline = df.groupby("created_date").size()
    st.markdown("### Predictions Over Time")
    st.line_chart(timeline)

    st.markdown("### Feature Averages")
    numeric_feature_cols = [
        col for col in BASE_FEATURES if col in df.columns and col not in {"gender"}
    ]
    feature_means = df[numeric_feature_cols].mean(numeric_only=True)
    st.dataframe(feature_means.rename("mean"))

    probs = _probabilities_frame(df)
    if not probs.empty:
        st.markdown("### Average Probability by Class")
        st.bar_chart(probs.mean())

    st.markdown("### Recent Predictions")
    st.dataframe(df.head(20)[
        [
            "created_at",
            "gender",
            "age",
            "hba1c",
            "chol",
            "bmi",
            "predicted_class",
        ]
    ])


def main() -> None:
    st.set_page_config(page_title="Diabetes Insights Studio", layout="wide")
    st.title("🩺 Diabetes Insights Studio")
    st.caption(
        "A sleek workspace to run predictions, persist patient records, and visualize trends in real time."
    )

    tab_predict, tab_dashboard = st.tabs(["Predict", "Dashboard"])

    with tab_predict:
        render_prediction_section()

    with tab_dashboard:
        df = fetch_predictions()
        render_dashboard(df)


if __name__ == "__main__":
    init_db()
    main()
