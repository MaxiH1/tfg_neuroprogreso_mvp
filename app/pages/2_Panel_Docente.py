# pages/2_Panel_Docente.py

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
from recommendations import get_recommendation


# ===============================
# Estilos simplificados
# ===============================
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}

.card-risk-low    { border-left: 6px solid #22c55e; background: #f0fdf4; }
.card-risk-medium { border-left: 6px solid #f59e0b; background: #fffbeb; }
.card-risk-high   { border-left: 6px solid #ef4444; background: #fef2f2; }

.card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.3rem; }
.card-text  { font-size: 0.92rem; color:#374151; }

.insight-icon { font-size: 1.7rem; margin-right: 0.6rem; }

.mini-tag {
    background: #fff7ed;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #b45309;
    display:inline-block;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ===============================
# Cargar datos y modelo
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/mmasd_features.csv")


@st.cache_resource
def load_model(df):
    features = ["edad_meses", "affect_total", "RRB", "overall_total", "severity"]
    X = df[features]
    y = df["y"]

    model = xgb.XGBClassifier(
        max_depth=4, eta=0.1, n_estimators=300,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss"
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer, features


df = load_data()
model, explainer, feature_cols = load_model(df)


# ===============================
# Nombres ficticios
# ===============================
FICTICIOUS_NAMES = [
    "Luna","Tomás","Mía","Mateo","Emma","Lucas",
    "Valentina","Benjamín","Sofía","Julián","Isabella",
    "Thiago","Camila","Santino","Olivia","Felipe",
]

def student_options(df):
    opts = {}
    for idx in df.index:
        name = FICTICIOUS_NAMES[idx % len(FICTICIOUS_NAMES)]
        edad = df.loc[idx, "edad_meses"]//12
        opts[f"{name} – {edad} años (Caso {idx})"] = idx
    return opts


# ===============================
# Clasificación de riesgo pedagógico
# ===============================
def classify_risk(prob):
    if prob < 0.20:
        return ("low", "📘", "Riesgo bajo",
            "El estudiante mantiene una trayectoria acorde.",
            "Continuar reforzando hábitos y consignas claras.")
    elif prob < 0.60:
        return ("medium", "⚠️", "Riesgo moderado",
            "Puede beneficiarse de apoyos adicionales.",
            "Ajustar duración de tareas y monitorear conductas.")
    else:
        return ("high", "🚨", "Riesgo alto",
            "Requiere un seguimiento cercano.",
            "Coordinar con familia y profesionales para ajustar apoyos.")


# ===============================
# Indicadores en lenguaje simple
# ===============================
def participation_text(val):
    if val <= 15:
        return "Participación limitada: se beneficia de consignas cortas y apoyo visual."
    elif val <= 25:
        return "Participación intermedia: responde bien con instrucciones claras."
    return "Buena participación en actividades estructuradas."


def regulation_text(val):
    if val <= 2:
        return "Regulación adecuada en la mayoría de las situaciones."
    elif val <= 4:
        return "Aparecen algunas conductas repetitivas; anticipar cambios ayuda."
    return "Las conductas repetitivas pueden afectar la clase; coordinar estrategias."


def demand_text(val):
    if val <= 2:
        return "Las tareas actuales parecen manejables."
    elif val <= 4:
        return "Algunas tareas pueden resultar exigentes; ajustar dificultad ayuda."
    return "Las demandas son altas; revisar adaptaciones y apoyos disponibles."


# ===============================
# LAYOUT
# ===============================
st.markdown("## 🏫 Panel Docente")

# Selector
options = student_options(df)
sel_label = st.selectbox("Seleccioná al estudiante", list(options.keys()))
idx = options[sel_label]

row = df.loc[idx, feature_cols]
X_case = row.to_frame().T
proba = float(model.predict_proba(X_case)[0, 1])

shap_case = explainer.shap_values(X_case)[0]

# ------------------------
# 1. Tarjeta de riesgo
# ------------------------
nivel, emoji, titulo_riesgo, texto1, texto2 = classify_risk(proba)

st.markdown(f"""
<div class="card card-risk-{nivel}">
    <div class="card-title">{emoji} {titulo_riesgo}</div>
    <div class="card-text">{texto1}</div>
    <div class="card-text" style="margin-top:0.3rem;">{texto2}</div>
</div>
""", unsafe_allow_html=True)


# ------------------------
# 2. Indicadores (3 tarjetas limpias)
# ------------------------
st.markdown("### 2️⃣ Indicadores para el aula")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">📝 Participación en clase</div>
        <div class="card-text">%s</div>
    </div>
    """ % participation_text(row["affect_total"]), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🔄 Regulación emocional</div>
        <div class="card-text">%s</div>
    </div>
    """ % regulation_text(row["RRB"]), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">📚 Demandas académicas</div>
        <div class="card-text">%s</div>
    </div>
    """ % demand_text(row["severity"]), unsafe_allow_html=True)


# ------------------------
# 3. Sugerencias pedagógicas
# ------------------------
st.markdown("### 3️⃣ Sugerencias pedagógicas")

reco = get_recommendation("docente", proba, row, shap_case)

st.markdown("""
<div class="card">
    <div class="mini-tag">Sugerencia generada con IA</div>
    <b>Contexto:</b> %s<br>
    <b>Recomendación principal:</b> %s<br>
    <b>Nota:</b> %s
</div>
""" % (reco["intro"], reco["recomendacion"], reco["disclaimer"]),
unsafe_allow_html=True)

st.caption("Las recomendaciones no reemplazan el juicio profesional docente.")