# app/streamlit_app.py

import streamlit as st

st.set_page_config(
    page_title="Plataforma Clínica Educativa",
    page_icon="🧠",
    layout="wide",
)

# -------------------------------------------------------------------
# Estilos generales (cards grandes para roles)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .role-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease-in-out;
        height: 100%;
    }
    .role-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.09);
    }
    .role-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .role-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .role-badge-familia {
        background-color: #dbeafe;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        color: #1d4ed8;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .role-badge-docente {
        background-color: #fef3c7;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        color: #b45309;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .role-badge-profesional {
        background-color: #d1fae5;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        color: #047857;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Header principal
# -------------------------------------------------------------------
st.markdown(
    """
    <h1 style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
        <span style="font-size:2.3rem;">🧠</span>
        <span>PLATAFORMA CLÍNICA EDUCATIVA</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.write(
    "La plataforma integra información clínica, educativa y familiar para estimar el "
    "progreso de un niño o niña y ofrecer recomendaciones orientativas según el rol del usuario."
)

st.markdown("### Bienvenido/a")

st.write(
    "Por favor, seleccioná tu rol para ver un panel personalizado. "
    "Cada panel presenta la información con un lenguaje y nivel de detalle adaptado "
    "a las necesidades de **familias**, **docentes** y **profesionales de la salud**."
)

st.markdown("---")

# -------------------------------------------------------------------
# Cards de selección de rol
# ⚠️ IMPORTANTE: las rutas de page_link son SOLO el nombre del archivo
#     que está dentro de app/pages/
# -------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="role-card">
            <div class="role-badge-familia">👨‍👩‍👧 Familia</div>
            <div class="role-title">Familia</div>
            <div class="role-subtitle">
                Visualizá el progreso general del niño y recibí sugerencias cotidianas para acompañarlo en casa.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "1_Panel_Familia.py",      # ✅ solo nombre de archivo
        label="Ver panel de Familia",
        icon="👨‍👩‍👧",
    )

with col2:
    st.markdown(
        """
        <div class="role-card">
            <div class="role-badge-docente">📘 Docente</div>
            <div class="role-title">Docente</div>
            <div class="role-subtitle">
                Explorá indicadores escolares, riesgo pedagógico y sugerencias para el aula.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "2_Panel_Docente.py",      # ✅ placeholder, ya creado como archivo vacío
        label="Ver panel de Docente",
        icon="📘",
    )

with col3:
    st.markdown(
        """
        <div class="role-card">
            <div class="role-badge-profesional">🩺 Profesional</div>
            <div class="role-title">Profesional</div>
            <div class="role-subtitle">
                Observá el perfil clínico global y una recomendación orientativa para la intervención.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "3_Panel_Profesional.py",  # ✅ placeholder
        label="Ver panel de Profesional",
        icon="🩺",
    )

