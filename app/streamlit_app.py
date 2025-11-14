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
    page_title="Prototipo Neuroprogreso",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Prototipo de plataforma - Predicción y recomendaciones")
st.write(
    "Este prototipo ilustra cómo un modelo de IA puede integrar información "
    "para estimar el progreso de un niño y generar recomendaciones orientativas "
    "según el rol del usuario."
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
        eval_metric="logloss"
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
    options=["Familia", "Docente", "Terapeuta"]
)

rol_map = {
    "Familia": "familia",
    "Docente": "docente",
    "Terapeuta": "terapeuta"
}
rol_interno = rol_map[rol_humano]

# Elegir un caso del dataset (por índice)
indice_caso = st.sidebar.number_input(
    "Seleccioná un caso del dataset (índice)",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

st.sidebar.info(
    "En esta versión del prototipo se utilizan casos reales del dataset procesado. "
    "Más adelante se puede habilitar la carga manual de datos."
)

# -------------------------------------------------------------------
# Mostrar datos del caso seleccionado
# -------------------------------------------------------------------

st.subheader("1️⃣ Datos del caso seleccionado")

features_row = df.loc[indice_caso, feature_cols]
target_real = df.loc[indice_caso, "y"]

col1, col2 = st.columns(2)
with col1:
    st.write("**Características del niño**")
    st.table(features_row.to_frame(name="valor"))

with col2:
    st.write("**Valor objetivo en el dataset (y)**")
    st.write(int(target_real))

# -------------------------------------------------------------------
# Predicción del modelo
# -------------------------------------------------------------------

st.subheader("2️⃣ Predicción del modelo")

X_case = features_row.to_frame().T  # DataFrame de 1 fila
proba = model.predict_proba(X_case)[0, 1]
pred = int(proba >= 0.5)

st.write(f"**Probabilidad estimada (clase 1):** {proba:.2f}")
st.write(f"**Predicción binaria del modelo:** {pred}")

st.caption(
    "La clase 1 representa un estado de mayor necesidad de apoyo según la definición "
    "utilizada en el dataset."
)

# -------------------------------------------------------------------
# Explicabilidad local con SHAP
# -------------------------------------------------------------------

st.subheader("3️⃣ Explicación de la predicción (SHAP local)")

shap_values_case = explainer.shap_values(X_case)[0]

st.write(
    "El siguiente gráfico muestra cómo cada variable contribuye a la predicción "
    "para este caso en particular."
)

# Force plot local en modo matplotlib
shap.initjs()  # aunque estemos usando matplotlib, no molesta

fig, ax = plt.subplots(figsize=(8, 2.5))
shap.force_plot(
    explainer.expected_value,
    shap_values_case,
    X_case,
    matplotlib=True,
    show=False
)
st.pyplot(fig)

# -------------------------------------------------------------------
# Recomendación personalizada según rol
# -------------------------------------------------------------------

st.subheader("4️⃣ Recomendación personalizada según tu rol")

reco = get_recommendation(
    rol=rol_interno,
    prob=float(proba),
    features_row=features_row,
    shap_values_row=shap_values_case
)

st.write(f"**Rol seleccionado:** {rol_humano}")
st.write(f"**Perfil estimado de apoyo:**")
st.caption(
    "El perfil se calcula internamente combinando la probabilidad del modelo con "
    "las variables más influyentes."
)

st.markdown(f"**Contexto:** {reco['intro']}")
st.markdown(f"**Recomendación principal:** {reco['recomendacion']}")
st.markdown(f"**Nota importante:** {reco['disclaimer']}")

st.info(
    "Las recomendaciones generadas son orientativas y no reemplazan la evaluación ni las "
    "decisiones de los profesionales de la salud o de la educación."
)