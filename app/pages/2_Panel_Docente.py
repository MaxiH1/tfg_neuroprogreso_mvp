# pages/2_Panel_Docente.py

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap

from recommendations import get_recommendation

# -------------------------------------------------------------------
# Estilos específicos para el panel de Docente
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .teacher-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #fbbf24;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
    }

    .risk-card {
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        display: flex;
        gap: 1.2rem;
        align-items: flex-start;
    }
    .risk-low {
        background-color: #fffbeb;
        border-left: 6px solid #22c55e;
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

    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.03);
        margin-bottom: 0.7rem;
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .metric-help {
        font-size: 0.8rem;
        color: #4b5563;
    }

    .mini-tag {
        display: inline-flex;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background-color: #fffbeb;
        color: #92400e;
        font-weight: 500;
        margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Carga de datos y modelo (misma lógica base que en Familia)
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
# Utilidades: alias de estudiante y cálculo de riesgo pedagógico
# -------------------------------------------------------------------
FICTICIOUS_NAMES = [
    "Luna", "Tomás", "Mía", "Mateo", "Emma", "Lucas",
    "Valentina", "Benjamín", "Sofía", "Julián", "Isabella",
    "Thiago", "Camila", "Santino", "Olivia", "Felipe",
]


def build_student_options(df: pd.DataFrame):
    options = {}
    for idx in df.index:
        name = FICTICIOUS_NAMES[idx % len(FICTICIOUS_NAMES)]
        edad_meses = int(df.loc[idx, "edad_meses"])
        edad_anos = edad_meses // 12
        alias = f"{name} – {edad_anos} años (Caso {idx})"
        options[alias] = idx
    return options


def classify_pedagogical_risk(prob: float):
    """
    Pensado para lenguaje docente:
    - nivel: 'low' | 'medium' | 'high'
    - etiqueta visible
    - texto principal
    - texto adicional
    """
    if prob < 0.33:
        nivel = "low"
        etiqueta = "El estudiante mantiene una trayectoria académica esperable"
        texto = (
            "Según el modelo, el estudiante se encuentra en una situación de **bajo riesgo pedagógico**. "
            "Podría sostener su progreso actual con las estrategias habituales del aula."
        )
        extra = (
            "Es un buen momento para reforzar logros, ofrecer pequeños desafíos graduados y seguir "
            "observando si aparecen cambios en la atención, el ánimo o la participación."
        )
    elif prob < 0.66:
        nivel = "medium"
        etiqueta = "El estudiante podría beneficiarse de un apoyo adicional"
        texto = (
            "El modelo sugiere un **riesgo pedagógico moderado**. Puede ser útil monitorear con más detalle "
            "algunas conductas en clase y realizar ajustes livianos en la enseñanza."
        )
        extra = (
            "Registrar brevemente situaciones de mayor dificultad, ajustar consignas o tiempos de trabajo, "
            "y mantener una comunicación abierta con la familia puede ayudar a prevenir mayores desfasajes."
        )
    else:
        nivel = "high"
        etiqueta = "El estudiante requiere un seguimiento cercano"
        texto = (
            "El modelo indica un **riesgo pedagógico alto**. Es un buen momento para coordinar acciones "
            "entre escuela, familia y profesionales, priorizando ajustes razonables en el entorno escolar."
        )
        extra = (
            "Conversar en equipo, revisar apoyos disponibles en el aula y documentar observaciones clave "
            "puede contribuir a una intervención más ajustada a las necesidades del estudiante."
        )

    return nivel, etiqueta, texto, extra


def risk_to_emoji_and_chip(nivel: str):
    if nivel == "low":
        return "📘", "chip chip-green", "Riesgo bajo"
    elif nivel == "medium":
        return "⚠️", "chip chip-amber", "Riesgo moderado"
    else:
        return "🚨", "chip chip-red", "Riesgo alto"


def describe_participation(affect_total: float) -> str:
    if affect_total <= 15:
        return "Participación limitada; puede necesitar más andamiaje y apoyos visuales."
    elif affect_total <= 25:
        return "Participación intermedia; responde bien cuando la consigna es clara y acotada."
    else:
        return "Buena participación en general, especialmente cuando se priorizan actividades estructuradas."


def describe_regulation(rrb: float) -> str:
    if rrb <= 2:
        return "Pocas conductas repetitivas observadas; la regulación parece adecuada en la mayoría de las situaciones."
    elif rrb <= 4:
        return "Aparecen algunas conductas repetitivas; puede ayudar anticipar cambios y ofrecer pausas breves."
    else:
        return "Las conductas repetitivas pueden interferir con el aprendizaje; conviene coordinar estrategias con la familia y terapeutas."


def describe_global_demand(severity: float) -> str:
    if severity <= 2:
        return "Las demandas académicas actuales parecen estar dentro de un rango manejable."
    elif severity <= 4:
        return "Es posible que algunas tareas resulten exigentes; graduar la dificultad puede marcar una diferencia."
    else:
        return "Las demandas actuales pueden estar siendo altas; revisar adaptaciones curriculares y apoyos disponibles."


# -------------------------------------------------------------------
# Layout principal
# -------------------------------------------------------------------
st.markdown("## 🏫 Panel de Docente (en construcción funcional)")

st.caption(
    "Este panel está pensado para ofrecer una lectura rápida del riesgo pedagógico, "
    "algunos indicadores del funcionamiento en el aula y una sugerencia práctica para ajustar la enseñanza."
)

# --- Selector de estudiante ------------------------------------------------
options = build_student_options(df)
selected_label = st.selectbox("Seleccioná al estudiante", list(options.keys()))
selected_index = options[selected_label]

features_row = df.loc[selected_index, feature_cols]
X_case = features_row.to_frame().T
proba = float(model.predict_proba(X_case)[0, 1])
pred = int(proba >= 0.5)

# SHAP solo para alimentar la recomendación
shap_values_case = explainer.shap_values(X_case)[0]

# -------------------------------------------------------------------
# 1. Predicción IA y riesgo pedagógico
# -------------------------------------------------------------------
nivel, etiqueta, texto, extra = classify_pedagogical_risk(proba)
emoji, chip_class, chip_text = risk_to_emoji_and_chip(nivel)

st.markdown("### 1️⃣ Predicción IA y riesgo pedagógico")

risk_class = {
    "low": "risk-card risk-low",
    "medium": "risk-card risk-medium",
    "high": "risk-card risk-high",
}[nivel]

st.markdown(
    f"""
    <div class="{risk_class}">
        <div class="risk-icon">{emoji}</div>
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
                Este valor no reemplaza la observación docente, sino que la complementa.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# 2. Indicadores para el aula (lectura rápida)
# -------------------------------------------------------------------
st.markdown("### 2️⃣ Indicadores para el aula")

st.write(
    "Estos indicadores se derivan de la información disponible en el modelo y buscan "
    "aportar una **lectura rápida**, no un diagnóstico. Podés usarlos como punto de partida "
    "para pensar apoyos y ajustes razonables."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Participación en clase</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{features_row["affect_total"]:.0f} (escala interna)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-help">{describe_participation(features_row["affect_total"])}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Regulación / conductas repetitivas</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{features_row["RRB"]:.0f} (escala interna)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-help">{describe_regulation(features_row["RRB"])}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Demandas globales percibidas</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{features_row["severity"]:.0f} (escala interna)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-help">{describe_global_demand(features_row["severity"])}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Si estos indicadores no coinciden con lo que observás en el aula, tu criterio docente "
    "es el principal. Podés usar esta información como insumo para reuniones de equipo, "
    "acuerdos con la familia o coordinación con otros profesionales."
)

# -------------------------------------------------------------------
# 3. Sugerencias pedagógicas personalizadas
# -------------------------------------------------------------------
st.markdown("### 3️⃣ Sugerencias pedagógicas para el aula")

reco = get_recommendation(
    rol="docente",
    prob=proba,
    features_row=features_row,
    shap_values_row=shap_values_case,
)

with st.container():
    st.markdown('<div class="teacher-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="mini-tag">Sugerencia generada con IA</div>',
        unsafe_allow_html=True,
    )

    st.write(f"**Contexto:** {reco['intro']}")
    st.write(f"**Recomendación principal:** {reco['recomendacion']}")
    st.write(f"**Nota importante:** {reco['disclaimer']}")

    st.info(
        "Las sugerencias generadas son orientativas y no reemplazan el juicio profesional ni las "
        "decisiones pedagógicas de la escuela."
    )

    st.markdown("</div>", unsafe_allow_html=True)