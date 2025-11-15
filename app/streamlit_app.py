import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from recommendations import get_recommendation  # mismo folder

# -------------------------------------------------------------------
# Configuración general de la página
# -------------------------------------------------------------------

st.set_page_config(
    page_title="PLATAFORMA CLÍNICA EDUCATIVA",
    page_icon="🧠",
    layout="wide"
)

# 🔹 Estilos sencillos tipo dashboard
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
        padding: 1.5rem 2rem;
    }
    .block-title {
        background-color: #1d4ed8;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 4px;
        display: inline-block;
        font-weight: 700;
        margin: 0.8rem 0 0.5rem 0;
        font-size: 0.95rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1rem 1.4rem;
        border-radius: 10px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 PLATAFORMA CLÍNICA EDUCATIVA")
st.write(
    "Este sistema ilustra cómo un modelo de IA puede integrar información clínica, "
    "educativa y familiar para estimar el progreso de un niño y generar "
    "recomendaciones orientativas según el rol del usuario."
)

# -------------------------------------------------------------------
# Cargar datos y entrenar el modelo (cacheado)
# -------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/mmasd_features.csv")
    return df


@st.cache_resource
def train_model_and_explainer(df):
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
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    return model, explainer, feature_cols


df = load_data()
model, explainer, feature_cols = train_model_and_explainer(df)

# -------------------------------------------------------------------
# Sidebar: selección de rol y de caso
# -------------------------------------------------------------------

st.sidebar.header("Configuración")

rol_humano = st.sidebar.selectbox(
    "Seleccioná tu rol",
    options=["Familia", "Docente", "Terapeuta"],
)

rol_map = {
    "Familia": "familia",
    "Docente": "docente",
    "Terapeuta": "terapeuta",
}
rol_interno = rol_map[rol_humano]

# Elegir un caso del dataset (por índice)
indice_caso = st.sidebar.number_input(
    "Seleccioná un caso del dataset (índice)",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1,
)

st.sidebar.info(
    "En esta versión de la plataforma se utilizan casos reales del dataset procesado. "
    "Más adelante se puede habilitar la carga manual de datos."
)

# -------------------------------------------------------------------
# BLOQUE 1: Mostrar datos del caso seleccionado
# -------------------------------------------------------------------

st.markdown(
    '<div class="block-title">1  Datos del caso seleccionado</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="card">', unsafe_allow_html=True)

features_row = df.loc[indice_caso, feature_cols]
target_real = df.loc[indice_caso, "y"]

col1, col2 = st.columns(2)
with col1:
    st.write("**Características del niño**")
    st.table(features_row.to_frame(name="valor"))

with col2:
    st.write("**Valor objetivo en el dataset (y)**")
    st.write(int(target_real))

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# BLOQUE 2: Predicción del modelo
# -------------------------------------------------------------------

st.markdown(
    '<div class="block-title">2  Predicción del modelo</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="card">', unsafe_allow_html=True)

X_case = features_row.to_frame().T  # DataFrame de 1 fila
proba = model.predict_proba(X_case)[0, 1]
pred = int(proba >= 0.5)

st.markdown(f"**Probabilidad estimada (clase 1):** `{proba:.2f}`")
st.markdown(f"**Predicción binaria del modelo:** `{pred}`")

st.caption(
    "La clase 1 representa un estado de mayor necesidad de apoyo según la definición "
    "utilizada en el dataset."
)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# BLOQUE 3: Explicación de la predicción (sin gráfico SHAP)
# -------------------------------------------------------------------

st.markdown(
    '<div class="block-title">3  Explicación de la predicción</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="card">', unsafe_allow_html=True)

# Calculamos valores SHAP (se usan internamente y para las recomendaciones)
shap_values_case = explainer.shap_values(X_case)[0]

st.write(
    "La predicción se basa en la combinación de la edad del niño, las puntuaciones de "
    "afecto, las conductas repetitivas y restringidas (RRB), el funcionamiento global "
    "y el nivel de severidad. "
)
st.write(
    "Los detalles técnicos de la explicabilidad del modelo (SHAP, importancia de "
    "variables y análisis cuantitativo) se documentan en el informe del prototipo "
    "tecnológico presentado en el trabajo final."
)

# (El gráfico SHAP local se omite en la interfaz para mantenerla simple.)
# Si en el futuro quisieras mostrarlo, aquí iría el force_plot.

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# BLOQUE 4: Recomendación personalizada según rol
# -------------------------------------------------------------------

st.markdown(
    '<div class="block-title">4  Recomendación personalizada según tu rol</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="card">', unsafe_allow_html=True)

reco = get_recommendation(
    rol=rol_interno,
    prob=float(proba),
    features_row=features_row,
    shap_values_row=shap_values_case,
)

st.markdown(f"**Rol seleccionado:** {rol_humano}")
st.markdown("**Perfil estimado de apoyo:**")
st.caption(
    "El perfil se calcula internamente combinando la probabilidad del modelo con "
    "la información más influyente de las variables de entrada."
)

st.markdown(f"**Contexto:** {reco['intro']}")
st.markdown(f"**Recomendación principal:** {reco['recomendacion']}")
st.markdown(f"**Nota importante:** {reco['disclaimer']}")

st.info(
    "Las recomendaciones generadas son orientativas y no reemplazan la evaluación ni las "
    "decisiones de los profesionales de la salud o de la educación."
)

st.markdown("</div>", unsafe_allow_html=True)