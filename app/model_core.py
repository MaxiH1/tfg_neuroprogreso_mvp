# app/model_core.py

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Carga el dataset procesado que usa el prototipo.
    """
    df = pd.read_csv("data/processed/mmasd_features.csv")
    return df


@st.cache_resource
def train_model_and_explainer(df: pd.DataFrame):
    """
    Entrena el modelo XGBoost y construye el explainer SHAP.

    Devuelve:
      - model: modelo XGBClassifier entrenado
      - explainer: TreeExplainer de SHAP
      - feature_cols: lista de columnas utilizadas como entrada del modelo
    """
    feature_cols = ["edad_meses", "affect_total", "RRB", "overall_total", "severity"]
    target_col = "y"

    X = df[feature_cols]
    y = df[target_col]

    model = xgb.XGBClassifier(
        max_depth=4,
        eta=0.1,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    return model, explainer, feature_cols