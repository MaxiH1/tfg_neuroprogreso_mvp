# pages/1_Panel_Familia.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from recommendations import get_recommendation


# -------------------------------------------------------------------
# Estilos específicos para el panel de Familia
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .family-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
    }

    .semaforo-card {
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        display: flex;
        gap: 1.2rem;
        align-items: flex-start;
    }
    .semaforo-low {
        background-color: #ecfdf3;
        border-left: 6px solid #22c55e;
    }
    .semaforo-medium {
        background-color: #fffbeb;
        border-left: 6px solid #f59e0b;
    }
    .semaforo-high {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
    }

    .semaforo-icon {
        font-size: 2.3rem;
        line-height: 1;
        margin-top: 0.1rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .chip-green {
        background-color: #dcfce7;
        color: #166534;
    }
    .chip-amber {
        background-color: #fef3c7;
        color: #92400e;
    }
    .chip-red {
        background-color: #fee2e2;
        color: #b91c1c;
    }

    .indicator-row {
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        padding: 0.7rem 0.4rem;
        border-bottom: 1px dashed #e5e7eb;
    }
    .indicator-icon {
        font-size: 1.4rem;
        width: 1.7rem;
        text-align: center;
    }
    .indicator-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.1rem;
    }
    .indicator-text {
        font-size: 0.9rem;
        color: #4b5563;
    }

    .mini-tag {
        display: inline-flex;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background-color: #eff6ff;
        color: #1d4ed8;
        font-weight: 500;
        margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Carga de datos y modelo (misma lógica que el prototipo original)
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/mmasd_features.csv")
    return df


@st.cache_resource
def train_model_and_explainer(df: pd.DataFrame):
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
# Utilidades: alias de niño/niña y cálculo de riesgo
# -------------------------------------------------------------------
FICTICIOUS_NAMES = [
    "Luna", "Tomás", "Mía", "Mateo", "Emma", "Lucas",
    "Valentina", "Benjamín", "Sofía", "Julián", "Isabella",
    "Thiago", "Camila", "Santino", "Olivia", "Felipe",
]


def build_child_options(df: pd.DataFrame):
    options = {}
    for idx in df.index:
        name = FICTICIOUS_NAMES[idx % len(FICTICIOUS_NAMES)]
        edad_meses = int(df.loc[idx, "edad_meses"])
        edad_anos = edad_meses // 12
        alias = f"{name} – {edad_anos} años (Caso {idx})"
        options[alias] = idx
    return options


def classify_risk(prob: float):
    """
    Devuelve:
    - nivel: 'low' | 'medium' | 'high'
    - etiqueta visible
    - texto principal empático
    - texto adicional
    """

    if prob < 0.20:
        nivel = "low"
        etiqueta = "Tu hijo/a está transitando un buen momento"
        texto = (
            "Tu hijo/a está atravesando un momento favorable. "
            "Con tu acompañamiento cotidiano y pequeñas rutinas claras, "
            "puede seguir progresando con tranquilidad."
        )
        extra = (
            "Es útil seguir celebrando los avances, mantener hábitos que funcionan "
            "y estar atento/a a cualquier cambio en el sueño, el ánimo o la concentración."
        )

    elif prob < 0.65:
        nivel = "medium"
        etiqueta = "Tu hijo/a podría necesitar un poco más de apoyo"
        texto = (
            "Tu hijo/a podría necesitar un poco más de apoyo en este tiempo. "
            "Pequeños cambios en casa y una buena comunicación con la escuela "
            "pueden ayudarlo/a a sostener su progreso."
        )
        extra = (
            "Podés anotar situaciones que se repiten y conversarlas con la escuela o terapeutas. "
            "Detectar cambios temprano siempre ayuda."
        )

    else:
        nivel = "high"
        etiqueta = "Tu hijo/a está necesitando mayor acompañamiento"
        texto = (
            "Tu hijo/a está atravesando un momento que requiere más apoyo. "
            "Acompañarlo/a de cerca y coordinar acciones entre familia, escuela "
            "y profesionales puede marcar una gran diferencia."
        )
        extra = (
            "Unir esfuerzos, compartir lo que observan en casa y ajustar rutinas clave "
            "puede ayudarlo/a a recuperar estabilidad y bienestar."
        )

    return nivel, etiqueta, texto, extra



def risk_to_emoji_and_chip(nivel: str):
    if nivel == "low":
        return "😊", "chip chip-green", "Nivel verde"
    elif nivel == "medium":
        return "🙂", "chip chip-amber", "Nivel amarillo"
    else:
        return "😟", "chip chip-red", "Nivel rojo"


def indicator_level_from_prob(prob: float):
    # Usamos la misma probabilidad para armar tres mensajes complementarios
    if prob < 0.33:
        return "ok"
    elif prob < 0.66:
        return "watch"
    else:
        return "alert"


def indicator_chip_class(level: str):
    if level == "ok":
        return "chip chip-green", "Fortaleza"
    elif level == "watch":
        return "chip chip-amber", "En observación"
    else:
        return "chip chip-red", "A conversar con el equipo"


# -------------------------------------------------------------------
# Layout principal
# -------------------------------------------------------------------
st.markdown("## 👨‍👩‍👧 Panel de Familia")

st.caption(
    "Visualizá el progreso general del niño o niña y recibí sugerencias cotidianas para acompañarlo en casa. "
    "Esta vista está pensada para familias, con lenguaje simple y sin gráficos técnicos."
)

# --- Selector de niño/niña -------------------------------------------------
options = build_child_options(df)
selected_label = st.selectbox("Seleccioná a quién querés ver", list(options.keys()))
selected_index = options[selected_label]

features_row = df.loc[selected_index, feature_cols]
y_real = int(df.loc[selected_index, "y"])

X_case = features_row.to_frame().T
proba = float(model.predict_proba(X_case)[0, 1])
pred = int(proba >= 0.5)

# Cálculo de SHAP solo para alimentar el motor de recomendaciones
shap_values_case = explainer.shap_values(X_case)[0]


# -------------------------------------------------------------------
# 1. Bloque de Predicción IA / Riesgo detectado
# -------------------------------------------------------------------
nivel, etiqueta, texto, extra = classify_risk(proba)
emoji, chip_class, chip_text = risk_to_emoji_and_chip(nivel)

st.markdown("### 1. Predicción IA y riesgo detectado")

semaforo_class = {
    "low": "semaforo-card semaforo-low",
    "medium": "semaforo-card semaforo-medium",
    "high": "semaforo-card semaforo-high",
}[nivel]

st.markdown(
    f"""
    <div class="{semaforo_class}">
        <div class="semaforo-icon">{emoji}</div>
        <div>
            <div style="margin-bottom:0.2rem;">
                <span class="{chip_class}">{chip_text}</span>
                <span style="margin-left:0.4rem; font-weight:600; font-size:0.95rem; color:#111827;">
                    {etiqueta}
                </span>
            </div>
            <div style="font-size:0.95rem; color:#111827; margin-bottom:0.35rem;">
                {texto}
            </div>
            <div style="font-size:0.9rem; color:#4b5563;">
                {extra}
            </div>
            <div style="font-size:0.8rem; color:#6b7280; margin-top:0.5rem;">
                Probabilidad estimada por el modelo para un nivel de apoyo alto (clase 1): <b>{proba:.2f}</b>.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# 2. Indicadores cotidianos en lenguaje simple (sin gráficos)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Bloque B – Indicadores del día a día (vista amigable)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 3. Recomendación personalizada usando tu motor de recomendaciones
# -------------------------------------------------------------------
st.markdown("### 3. Recomendación personalizada para la familia")

reco = get_recommendation(
    rol="familia",
    prob=proba,
    features_row=features_row,
    shap_values_row=shap_values_case,
)

with st.container():
    st.markdown('<div class="family-card">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="mini-tag">Recomendación generada con IA</div>
        """,
        unsafe_allow_html=True,
    )

    st.write(f"**Contexto:** {reco['intro']}")
    st.write(f"**Recomendación principal:** {reco['recomendacion']}")
    st.write(f"**Nota importante:** {reco['disclaimer']}")

    st.info(
        "Las recomendaciones generadas son orientativas y no reemplazan la evaluación ni las "
        "decisiones de los profesionales de la salud o de la educación."
    )

    st.markdown("</div>", unsafe_allow_html=True)