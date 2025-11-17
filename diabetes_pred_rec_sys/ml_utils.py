"""Shared utilities for diabetes risk inference."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

BASE_FEATURES: Sequence[str] = [
    "gender",
    "age",
    "urea",
    "cr",
    "hba1c",
    "chol",
    "tg",
    "hdl",
    "ldl",
    "vldl",
    "bmi",
]
MODEL_PATH = Path(__file__).parent / "models" / "best_model_random_forest.pkl"
CLASS_LABELS_DEFAULT = ["Class 0", "Class 1", "Class 2"]


def _gender_to_numeric(value: str) -> int:
    mapping = {"Male": 0, "Female": 1}
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported gender label: {value}") from exc


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model artifact at {MODEL_PATH}. "
            "Train a model and place the serialized pipeline at this path."
        )
    return joblib.load(MODEL_PATH)


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    engineered = features.copy()

    if {"hdl", "chol"}.issubset(engineered.columns):
        hdl_safe = engineered["hdl"].replace(0, np.nan)
        cr_series = engineered.get(
            "cr", pd.Series(1.0, index=engineered.index, dtype=float)
        ).replace(0, np.nan)
        chol_safe = engineered["chol"].replace(0, np.nan)

        engineered["chol_hdl_ratio"] = engineered["chol"] / hdl_safe

        if "ldl" in engineered.columns:
            engineered["ldl_hdl_ratio"] = engineered["ldl"] / hdl_safe
        if "tg" in engineered.columns:
            engineered["tg_hdl_ratio"] = engineered["tg"] / hdl_safe
        if "urea" in engineered.columns:
            engineered["urea_creatinine_ratio"] = engineered["urea"] / cr_series
        if {"ldl", "vldl"}.issubset(engineered.columns):
            engineered["lipid_density"] = (
                engineered["ldl"] + engineered["vldl"]
            ) / chol_safe

        metabolic_cols = [c for c in ["hba1c", "chol", "tg", "bmi"] if c in engineered.columns]
        if len(metabolic_cols) >= 2:
            engineered["metabolic_score"] = (
                engineered[metabolic_cols].rank(pct=True).mean(axis=1)
            )

    if {"age", "bmi"}.issubset(engineered.columns):
        engineered["age_bmi_interaction"] = engineered["age"] * engineered["bmi"]
        engineered["is_obese"] = (engineered["bmi"] >= 30).astype(int)

        engineered["age_band"] = pd.cut(
            engineered["age"],
            bins=[0, 29, 39, 49, 59, 69, np.inf],
            labels=["<30", "30-39", "40-49", "50-59", "60-69", "70+"],
            right=True,
            include_lowest=True,
        ).astype("string")

    ratio_columns = [c for c in engineered.columns if "ratio" in c or "density" in c]
    if ratio_columns:
        engineered[ratio_columns] = engineered[ratio_columns].replace([np.inf, -np.inf], np.nan)

    return engineered


def _prepare_input_dataframe(user_values: Mapping[str, float]) -> pd.DataFrame:
    base_series = {feature: float(user_values[feature]) for feature in BASE_FEATURES}
    df = pd.DataFrame([base_series])
    return engineer_features(df)


def _align_columns(
    df: pd.DataFrame, numeric_cols: Sequence[str] | None, categorical_cols: Sequence[str] | None
) -> pd.DataFrame:
    required: list[str] = []
    if numeric_cols:
        required.extend(list(numeric_cols))
    if categorical_cols:
        required.extend(list(categorical_cols))
    if not required:
        return df
    return df.reindex(columns=required, fill_value=np.nan)


def _attach_feature_names(matrix: Any, feature_names: Sequence[str] | None) -> Any:
    if isinstance(matrix, np.ndarray) and feature_names is not None:
        try:
            return pd.DataFrame(matrix, columns=feature_names)
        except ValueError:
            return matrix
    return matrix


def predict_diabetes(
    *,
    gender: str,
    age: float,
    urea: float,
    cr: float,
    hba1c: float,
    chol: float,
    tg: float,
    hdl: float,
    ldl: float,
    vldl: float,
    bmi: float,
) -> Dict[str, Any]:
    """Run inference with the serialized model and return structured output."""
    artifact = load_model()
    feature_names: Sequence[str] | None = None

    if isinstance(artifact, dict):
        estimator = artifact["model"]
        preprocessor = artifact.get("preprocessor")
        label_encoder = artifact.get("label_encoder")
        numeric_cols = artifact.get("numeric_columns")
        categorical_cols = artifact.get("categorical_columns")
        feature_names = artifact.get("feature_names")
        meta = {
            "test_accuracy": artifact.get("test_accuracy"),
            "trained_date": artifact.get("trained_date"),
        }
        class_names = artifact.get("class_names")
    else:
        estimator = artifact
        preprocessor = None
        label_encoder = None
        numeric_cols = None
        categorical_cols = None
        meta = {}
        class_names = None

    gender_numeric = _gender_to_numeric(gender)
    user_values = {
        "gender": gender_numeric,
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
    features = _prepare_input_dataframe(user_values)
    features = _align_columns(features, numeric_cols, categorical_cols)

    model_input = preprocessor.transform(features) if preprocessor is not None else features
    model_input = _attach_feature_names(model_input, feature_names)

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(model_input)[0]
        labels = class_names or getattr(estimator, "classes_", CLASS_LABELS_DEFAULT)
        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            try:
                label_array = np.asarray(labels)
                labels = label_encoder.inverse_transform(label_array)
            except Exception:
                labels = [str(lbl) for lbl in labels]
        probabilities = {str(label): float(score) for label, score in zip(labels, proba)}
        top_label = max(probabilities, key=probabilities.get)
        response: Dict[str, Any] = {
            "predicted_class": top_label,
            "probabilities": probabilities,
        }
    else:
        prediction = estimator.predict(model_input)[0]
        response = {"prediction": float(prediction)}

    response["model_metadata"] = {k: v for k, v in meta.items() if v is not None}
    return response
