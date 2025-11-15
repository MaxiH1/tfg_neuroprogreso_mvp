# pages/3_Panel_Profesional.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from recommendations import get_recommendation

# -------------------------------------------------------------------
# Estilos específicos para el panel Profesional
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .pro-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
    }

    .risk-banner {
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        display: flex;
        gap: 1.1rem;
        align-items: flex-start;
    }
    .risk-low {
        background-color: #ecfdf5;
        border-left: 6px solid #10b981;
    }
    .risk-medium {
        background-color: #fffbeb;
        border-left: 6px solid #f59e0b;
    }
    .risk-high {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
    }

    .risk-icon {
        font-size: 2.1rem;
        line-height: 1;
        margin-top: 0.15rem;
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
        background-color: #d1fae5;
        color: #047857;
    }
    .chip-amber {
        background-color: #fef3c7;
        color: #92400e;
    }
    .chip-red {
        background-color: #fee2e2;
        color: #b91c1c;
    }

    .indicator-col-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .indicator-tag {
        display: inline-flex;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    .tag-soft-green {
        background-color: #dcfce7;
        color: #166534;
    }
    .tag-soft-amber {
        background-color: #fef3c7;
        color: #92400e;
    }
    .tag-soft-red {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .small-muted {
        font-size: 0.8rem;
        color: #6b7280;
    }

    .mini-tag {
        display: inline-flex;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background-color: #eef2ff;
        color: #4f46e5;
        font-weight: 500;
        margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Carga de datos y modelo (misma lógica que en Familia / Docente)
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
# Utilidades para alias de niño/niña y clasificación de riesgo
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


def classify_global_risk(prob: float):
    """Umbrales ajustados para no saturar en riesgo alto."""
    if prob < 0.30:
        nivel = "low"
        titulo = "Riesgo clínico bajo según el modelo"
        texto = (
            "El modelo estima una probabilidad baja de requerir apoyos clínicos intensivos. "
            "Se recomienda mantener el seguimiento habitual y registrar cambios relevantes "
            "en funcionamiento diario, sueño y participación escolar."
        )
    elif prob < 0.70:
        nivel = "medium"
        titulo = "Riesgo clínico moderado – sugerido seguimiento cercano"
        texto = (
            "El modelo estima un riesgo clínico moderado. Puede ser útil un seguimiento más "
            "frecuente, ajustando intervenciones específicas y coordinando información con la "
            "familia y la escuela para detectar cambios tempranamente."
        )
    else:
        nivel = "high"
        titulo = "Riesgo clínico alto – sugerida coordinación interdisciplinaria"
        texto = (
            "El modelo estima un riesgo clínico alto. Resulta pertinente revisar el plan "
            "de intervención, priorizar objetivos y coordinar acciones entre el equipo "
            "interdisciplinario, la escuela y la familia."
        )

    return nivel, titulo, texto


def risk_chip(nivel: str):
    if nivel == "low":
        return "✅", "chip chip-green", "Perfil de riesgo bajo"
    elif nivel == "medium":
        return "⚠️", "chip chip-amber", "Perfil de riesgo moderado"
    else:
        return "🚨", "chip chip-red", "Perfil de riesgo alto"


def level_from_value(value, q_low, q_high):
    if value <= q_low:
        return "low"
    elif value <= q_high:
        return "medium"
    else:
        return "high"


def level_tag(level: str):
    if level == "low":
        return "tag-soft-green", "Perfil leve"
    elif level == "medium":
        return "tag-soft-amber", "Perfil intermedio"
    else:
        return "tag-soft-red", "Perfil marcado"


FRIENDLY_FEATURE_NAMES = {
    "edad_meses": "Edad cronológica",
    "affect_total": "Área socio-comunicativa",
    "RRB": "Conductas repetitivas / intereses restringidos",
    "overall_total": "Funcionamiento adaptativo global",
    "severity": "Índice global de severidad",
}


# -------------------------------------------------------------------
# Layout principal
# -------------------------------------------------------------------
st.markdown("## 🩺 Panel de Profesional")

st.caption(
    "Este panel ofrece un resumen clínico global, algunos indicadores derivados del modelo "
    "y una recomendación orientativa para apoyar la toma de decisiones profesionales."
)

# --- Selector de caso -------------------------------------------------------
options = build_child_options(df)
selected_label = st.selectbox("Seleccioná el caso a visualizar", list(options.keys()))
selected_index = options[selected_label]

features_row = df.loc[selected_index, feature_cols]
y_real = int(df.loc[selected_index, "y"])

X_case = features_row.to_frame().T
proba = float(model.predict_proba(X_case)[0, 1])
pred = int(proba >= 0.5)

# SHAP para factores influyentes
shap_values_case = explainer.shap_values(X_case)[0]


# -------------------------------------------------------------------
# 1. Perfil de riesgo global
# -------------------------------------------------------------------
nivel, titulo_riesgo, texto_riesgo = classify_global_risk(proba)
emoji, chip_class, chip_text = risk_chip(nivel)

risk_css = {
    "low": "risk-banner risk-low",
    "medium": "risk-banner risk-medium",
    "high": "risk-banner risk-high",
}[nivel]

st.markdown("### 1. Perfil de riesgo global")

st.markdown(
    f"""
    <div class="{risk_css}">
        <div class="risk-icon">{emoji}</div>
        <div>
            <div style="margin-bottom:0.25rem;">
                <span class="{chip_class}">{chip_text}</span>
            </div>
            <div style="font-weight:600; font-size:0.98rem; margin-bottom:0.25rem; color:#111827;">
                {titulo_riesgo}
            </div>
            <div style="font-size:0.92rem; color:#374151; margin-bottom:0.35rem;">
                {texto_riesgo}
            </div>
            <div class="small-muted">
                Probabilidad estimada por el modelo para la clase 1 (mayor necesidad de apoyo): 
                <b>{proba:.2f}</b>.  
                Valor objetivo registrado en el dataset (y): <b>{y_real}</b>.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# 2. Indicadores clínicos sintetizados
# -------------------------------------------------------------------
st.markdown("### 2. Indicadores clínicos sintetizados")

# Umbrales basados en cuantiles de la muestra
q_aff_low, q_aff_high = df["affect_total"].quantile([0.33, 0.66])
q_rrb_low, q_rrb_high = df["RRB"].quantile([0.33, 0.66])
q_sev_low, q_sev_high = df["severity"].quantile([0.33, 0.66])

aff_val = float(features_row["affect_total"])
rrb_val = float(features_row["RRB"])
sev_val = float(features_row["severity"])

aff_level = level_from_value(aff_val, q_aff_low, q_aff_high)
rrb_level = level_from_value(rrb_val, q_rrb_low, q_rrb_high)
sev_level = level_from_value(sev_val, q_sev_low, q_sev_high)

aff_tag_class, aff_tag_text = level_tag(aff_level)
rrb_tag_class, rrb_tag_text = level_tag(rrb_level)
sev_tag_class, sev_tag_text = level_tag(sev_level)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<div class="indicator-col-title">Escala socio-comunicativa</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="indicator-tag {aff_tag_class}">{aff_tag_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:0.9rem; color:#4b5563; margin-top:0.3rem;">
            Resume la puntuación total en el área socio-comunicativa. Permite estimar
            el nivel de dificultades en interacción social, comunicación y juego.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<div class="indicator-col-title">Conductas repetitivas</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="indicator-tag {rrb_tag_class}">{rrb_tag_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:0.9rem; color:#4b5563; margin-top:0.3rem;">
            Integra la presencia de conductas repetitivas, intereses restringidos
            o rigidez conductual, relevantes para ajustar apoyos en la vida diaria.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<div class="indicator-col-title">Índice global de severidad</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="indicator-tag {sev_tag_class}">{sev_tag_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:0.9rem; color:#4b5563; margin-top:0.3rem;">
            Integra la información global del caso en una escala de severidad relativa
            dentro de la muestra, facilitando la priorización de objetivos clínicos.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Estos indicadores son aproximaciones derivadas de la información disponible en el dataset y del modelo. "
    "No reemplazan la valoración clínica directa ni la aplicación de escalas específicas."
)

# -------------------------------------------------------------------
# 3. Factores que más influyeron en la predicción (según el modelo)
# -------------------------------------------------------------------
st.markdown("### 3. Factores más influyentes en la predicción")

# top 3 absolutas
abs_shap = np.abs(shap_values_case[0])
idx_sorted = np.argsort(-abs_shap)  # descendente
top_idx = idx_sorted[:3]

items = []
for i in top_idx:
    feat = feature_cols[i]
    name = FRIENDLY_FEATURE_NAMES.get(feat, feat)
    value = float(features_row[feat])
    contrib = shap_values_case[0][i]
    direction = "aumentó" if contrib > 0 else "disminuyó"
    items.append(
        f"- **{name}** (valor observado: `{value}`) {direction} la probabilidad estimada de requerir mayor apoyo."
    )

st.markdown(
    """
    El modelo identifica, para este caso, las siguientes variables como las de mayor
    peso relativo en la predicción. La interpretación debe hacerse siempre en el marco
    de la historia clínica y de la evaluación integral.
    """
)

st.markdown("\n".join(items))

st.caption(
    "Los valores SHAP describen contribuciones relativas del modelo, no equivalen a coeficientes clínicos "
    "ni a significancia estadística en estudios poblacionales."
)

# -------------------------------------------------------------------
# 4. Recomendación orientativa para la intervención
# -------------------------------------------------------------------
st.markdown("### 4. Recomendación orientativa para la intervención")

reco = get_recommendation(
    rol="terapeuta",   # rol interno para profesionales de la salud
    prob=proba,
    features_row=features_row,
    shap_values_row=shap_values_case,
)

with st.container():
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="mini-tag">Recomendación basada en IA (uso profesional)</div>',
        unsafe_allow_html=True,
    )

    st.write(f"**Contexto clínico sintetizado:** {reco['intro']}")
    st.write(f"**Sugerencia principal de intervención:** {reco['recomendacion']}")
    st.write(f"**Nota de uso responsable:** {reco['disclaimer']}")

    st.info(
        "Estas sugerencias son un apoyo orientativo para la práctica profesional y no sustituyen "
        "la evaluación clínica, el juicio experto ni las guías de práctica basadas en evidencia."
    )

    st.markdown("</div>", unsafe_allow_html=True)