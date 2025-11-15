import streamlit as st
import pandas as pd
import xgboost as xgb
import shap

from recommendations import get_recommendation  # mismo folder

# -------------------------------------------------------------------
# Configuración general de la página
# -------------------------------------------------------------------

st.set_page_config(
    page_title="PLATAFORMA CLÍNICA EDUCATIVA",
    page_icon="🧠",
    layout="wide",
)

# 🔹 Estilos tipo dashboard + cards de rol
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
    .role-card {
        background-color: #eef2ff;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
        text-align: center;
        border: 1px solid #e0e7ff;
    }
    .role-card h3 {
        margin-bottom: 0.4rem;
    }
    .role-card p {
        font-size: 0.9rem;
        color: #4b5563;
        min-height: 3.2rem;
    }
    .role-card-familia {
        background-color: #e0f2fe;
        border-color: #7dd3fc;
    }
    .role-card-docente {
        background-color: #fef9c3;
        border-color: #facc15;
    }
    .role-card-terapeuta {
        background-color: #dcfce7;
        border-color: #4ade80;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Datos y modelo (cacheado)
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


@st.cache_data
def generar_nombres_ficticios(n):
    """
    Genera una lista de nombres ficticios (uno por cada fila).
    Pensado para 31 casos, pero dejamos margen.
    """
    base_names = [
        "Lucía", "Mateo", "Valentina", "Thiago", "Sofía", "Benjamín",
        "Martina", "Julián", "Camila", "Santino", "Agustina", "Bruno",
        "Isabella", "Tomás", "Emma", "Joaquín", "Mía", "Lautaro",
        "Olivia", "Franco", "Catalina", "Nicolás", "Luna", "Ramiro",
        "Abril", "Felipe", "Renata", "Simón", "Victoria", "Bautista",
        "Zoe", "Iván", "Clara", "Dante", "Malena", "Gael", "Josefina",
        "Ian", "Mora", "Lucas"
    ]
    if n <= len(base_names):
        return base_names[:n]
    else:
        extra = [f"Caso {i+1}" for i in range(n - len(base_names))]
        return base_names + extra


df = load_data()
model, explainer, feature_cols = train_model_and_explainer(df)
nombres_ninos = generar_nombres_ficticios(len(df))


# -------------------------------------------------------------------
# Helpers de navegación
# -------------------------------------------------------------------

def set_rol(rol_humano: str):
    st.session_state["rol_humano"] = rol_humano


def reset_rol():
    if "rol_humano" in st.session_state:
        del st.session_state["rol_humano"]
    st.rerun()


# -------------------------------------------------------------------
# Pantalla inicial: selección de rol
# -------------------------------------------------------------------

def pantalla_inicio():
    st.title("🧠 PLATAFORMA CLÍNICA EDUCATIVA")
    st.subheader("Bienvenido/a")
    st.write(
        "Por favor, seleccioná tu rol para ver un panel personalizado. "
        "La plataforma integra información clínica, educativa y familiar para estimar "
        "el progreso del niño y ofrecer recomendaciones orientativas."
    )

    st.markdown("")  # pequeño espacio

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="role-card role-card-familia">
                <h3>Familia</h3>
                <p>Visualizá el progreso general del niño y recibí sugerencias
                cotidianas para acompañarlo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ver panel de Familia", key="btn_familia"):
            set_rol("Familia")
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="role-card role-card-docente">
                <h3>Docente</h3>
                <p>Explorá indicadores escolares, riesgo pedagógico
                y sugerencias para el aula.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ver panel de Docente", key="btn_docente"):
            set_rol("Docente")
            st.rerun()

    with col3:
        st.markdown(
            """
            <div class="role-card role-card-terapeuta">
                <h3>Profesional</h3>
                <p>Observá el perfil clínico global del niño y una
                recomendación orientativa para la intervención.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ver panel de Profesional", key="btn_terapeuta"):
            set_rol("Terapeuta")
            st.rerun()


# -------------------------------------------------------------------
# Panel clínico–educativo según rol
# -------------------------------------------------------------------

def panel_rol(rol_humano: str):
    # Map interno para el módulo de recomendaciones
    rol_map = {
        "Familia": "familia",
        "Docente": "docente",
        "Terapeuta": "terapeuta",
    }
    rol_interno = rol_map[rol_humano]

    # Encabezado
    col_titulo, col_boton = st.columns([4, 1])
    with col_titulo:
        st.title(f"🧠 Plataforma Clínica Educativa – Panel de {rol_humano}")
        st.write(
            "Este panel muestra una vista simplificada del progreso del niño y una "
            "recomendación personalizada según tu rol."
        )
    with col_boton:
        st.button("Cambiar rol", on_click=reset_rol)

    # ---------------- Sidebar: selección de niño ----------------
    st.sidebar.header("Configuración del caso")

    opciones_ninos = [
        f"{nombres_ninos[i]}"
        for i in range(len(df))
    ]
    # Diccionario para mapear etiqueta -> índice
    label_to_idx = {label: i for i, label in enumerate(opciones_ninos)}

    nombre_seleccionado = st.sidebar.selectbox(
        "Seleccioná un niño",
        options=opciones_ninos,
        index=0,
    )
    indice_caso = label_to_idx[nombre_seleccionado]

    st.sidebar.info(
        "En esta versión de la plataforma se utilizan casos reales del dataset "
        "procesado, presentados con nombres ficticios. Más adelante se puede "
        "habilitar la carga manual de datos."
    )

    # Datos del caso
    features_row = df.loc[indice_caso, feature_cols]
    target_real = int(df.loc[indice_caso, "y"])
    edad = int(features_row["edad_meses"])
    affect = int(features_row["affect_total"])
    rrb = int(features_row["RRB"])
    overall = int(features_row["overall_total"])
    severity = int(features_row["severity"])

    if target_real == 1:
        descripcion_y = "Mayor necesidad de apoyo (clase 1 en el dataset)."
    else:
        descripcion_y = "Necesidad de apoyo relativamente menor (clase 0 en el dataset)."

    # ---------------- BLOQUE 1: Minihistorial ----------------
    st.markdown(
        '<div class="block-title">1  Datos del caso seleccionado</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Nombre ficticio:** {nombre_seleccionado}")
        st.markdown("**Minihistorial del niño (resumen interno):**")
        st.markdown(
            f"""
            - Edad aproximada: **{edad} meses**  
            - Puntaje de afecto / interacción socioemocional: **{affect}**  
            - Conductas repetitivas y restringidas (RRB): **{rrb}**  
            - Funcionamiento global (overall_total): **{overall}**  
            - Nivel de severidad: **{severity}**
            """
        )

    with col2:
        st.markdown("**Estado objetivo en el dataset (y):**")
        st.markdown(f"Valor interno: **{target_real}**")
        st.caption(descripcion_y)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- BLOQUE 2: Predicción del modelo ----------------
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
        "La clase 1 representa un estado de mayor necesidad de apoyo según la "
        "definición utilizada en el dataset."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- BLOQUE 3: Explicación (sin gráfico) ----------------
    st.markdown(
        '<div class="block-title">3  Explicación de la predicción</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="card">', unsafe_allow_html=True)

    shap_values_case = explainer.shap_values(X_case)[0]

    st.write(
        "La predicción se basa en la combinación de la edad, las puntuaciones de "
        "afecto, las conductas repetitivas (RRB), el funcionamiento global y el "
        "nivel de severidad. "
    )
    st.write(
        "Los detalles técnicos de la explicabilidad del modelo (valores SHAP, "
        "importancia de variables, etc.) se desarrollan en el informe del prototipo "
        "tecnológico del trabajo final."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- BLOQUE 4: Recomendación según rol ----------------
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
        "El perfil se calcula internamente combinando la probabilidad del modelo "
        "con la información más influyente de las variables de entrada."
    )

    st.markdown(f"**Contexto:** {reco['intro']}")
    st.markdown(f"**Recomendación principal:** {reco['recomendacion']}")
    st.markdown(f"**Nota importante:** {reco['disclaimer']}")

    st.info(
        "Las recomendaciones generadas son orientativas y no reemplazan la evaluación "
        "ni las decisiones de los profesionales de la salud o de la educación."
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Enrutamiento principal
# -------------------------------------------------------------------

if "rol_humano" not in st.session_state:
    pantalla_inicio()
else:
    panel_rol(st.session_state["rol_humano"])
