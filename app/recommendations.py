# app/recommendations.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------------------
# Configuración de la base de conocimiento
# -------------------------------------------------------------------

BASE_KB_DIR = Path("knowledge_base")

# Ojo con los nombres de carpeta que tenés en el proyecto
ROLE_DIRS: Dict[str, str] = {
    "familia": "family_guidelines",
    "docente": "educational_guidelines",
    "terapeuta": "clinical_guidelines",
}


@dataclass
class RetrievedDoc:
    title: str
    text: str
    score: float


class SimpleRAG:
    """
    RAG-light: carga documentos .md por rol, arma un TF-IDF simple
    y devuelve los más parecidos a una consulta de texto.
    """

    def __init__(self, role: str):
        self.role = role
        self.docs: List[str] = []
        self.doc_titles: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None
        self._load_docs()

    def _load_docs(self) -> None:
        subdir = ROLE_DIRS.get(self.role)
        if not subdir:
            return

        kb_dir = BASE_KB_DIR / subdir
        if not kb_dir.exists():
            # No hay carpeta para este rol
            return

        for path in sorted(kb_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue

            text = text.strip()
            if not text:
                continue

            self.docs.append(text)
            self.doc_titles.append(path.stem)

        if self.docs:
            self.vectorizer = TfidfVectorizer(stop_words="spanish")
            self.doc_matrix = self.vectorizer.fit_transform(self.docs)

    def has_docs(self) -> bool:
        return bool(self.docs) and self.vectorizer is not None

    def retrieve(self, query: str, top_k: int = 2) -> List[RetrievedDoc]:
        if not self.has_docs():
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]

        idxs = np.argsort(sims)[::-1][:top_k]
        results: List[RetrievedDoc] = []
        for i in idxs:
            if sims[i] <= 0:
                continue
            results.append(
                RetrievedDoc(
                    title=self.doc_titles[i],
                    text=self.docs[i],
                    score=float(sims[i]),
                )
            )
        return results


# -------------------------------------------------------------------
# Helpers para armar consultas y extraer fragmentos de texto
# -------------------------------------------------------------------


def build_query(role: str, prob: float, features_row, shap_values_row=None) -> str:
    """
    Construye una consulta simple en texto según rol + probabilidad.
    No es un prompt para LLM, solo una query para TF-IDF.
    """

    riesgo = "bajo"
    if prob >= 0.7:
        riesgo = "alto"
    elif prob >= 0.4:
        riesgo = "moderado"

    if role == "familia":
        return (
            f"recomendaciones para familias, riesgo {riesgo}, hábitos de sueño, "
            f"rutinas en casa, comunicación con escuela"
        )
    elif role == "docente":
        return (
            f"recomendaciones pedagógicas para docentes, riesgo {riesgo}, "
            f"participación en clase, ajustes razonables, regulación de conducta, "
            f"ausencias y seguimiento escolar"
        )
    else:  # terapeuta / profesional
        return (
            f"orientaciones clínicas para profesionales de la salud, riesgo {riesgo}, "
            f"priorización de objetivos, coordinación interdisciplinaria, seguimiento"
        )


def extract_highlights(text: str, max_sentences: int = 3) -> str:
    """
    Extrae las primeras frases o líneas no vacías de un documento .md
    para usarlas como 'corazón' de la recomendación.
    """
    # Cortamos por saltos de línea
    lines = [l.strip("- •").strip() for l in text.splitlines()]
    lines = [l for l in lines if l]

    if not lines:
        return ""

    # Tomamos las primeras 3 líneas/frases informativas
    highlights = lines[:max_sentences]
    return " ".join(highlights)


# -------------------------------------------------------------------
# Reglas base (fallback) por rol – por si no hay docs o similitud baja
# -------------------------------------------------------------------


def _fallback_for_familia(prob: float) -> Dict[str, str]:
    if prob < 0.33:
        intro = (
            "Según los datos recientes, tu hijo/a se encuentra en un momento relativamente estable."
        )
        recomendacion = (
            "Mantené las rutinas que vienen funcionando (sueño, horarios, anticipación de cambios) "
            "y seguí observando pequeños avances en la vida diaria."
        )
    elif prob < 0.66:
        intro = (
            "Los datos sugieren que tu hijo/a podría necesitar un poco más de apoyo en este tiempo."
        )
        recomendacion = (
            "Puede ayudar reforzar las rutinas de sueño, anticipar cambios en el día y conversar "
            "con la escuela sobre lo que ven en el aula, buscando acuerdos simples y sostenibles."
        )
    else:
        intro = (
            "Los datos indican que tu hijo/a está atravesando un momento de mayor necesidad de apoyo."
        )
        recomendacion = (
            "Es recomendable acompañarlo/a de cerca, revisar las rutinas en casa y coordinar acciones "
            "con escuela y profesionales para sostener su bienestar y su participación cotidiana."
        )

    disclaimer = (
        "Esta sugerencia es orientativa y debe complementarse con el criterio de los profesionales "
        "que acompañan al niño o niña."
    )
    return {
        "intro": intro,
        "recomendacion": recomendacion,
        "disclaimer": disclaimer,
    }


def _fallback_for_docente(prob: float) -> Dict[str, str]:
    if prob < 0.33:
        intro = (
            "El modelo sugiere un riesgo pedagógico bajo en este momento."
        )
        recomendacion = (
            "Puede ser útil sostener las estrategias que ya funcionan en el aula, ofrecer consignas "
            "claras y breves, y reforzar los logros para consolidar la participación del estudiante."
        )
    elif prob < 0.66:
        intro = (
            "El modelo indica un riesgo pedagógico moderado que merece seguimiento cercano."
        )
        recomendacion = (
            "Se sugiere ajustar la dificultad de las actividades, fragmentar tareas extensas, "
            "combinar apoyos visuales y verbales, y registrar situaciones que se repiten para "
            "compartirlas con la familia y el equipo de apoyo."
        )
    else:
        intro = (
            "El modelo señala un riesgo pedagógico alto en este período."
        )
        recomendacion = (
            "Conviene revisar en equipo los apoyos disponibles en el aula, definir adaptaciones "
            "prioritarias (tiempos, consignas, apoyos visuales, espacios de regulación) y coordinar "
            "acuerdos claros con la familia y otros profesionales."
        )

    disclaimer = (
        "Esta sugerencia es orientativa y debe complementarse con el criterio pedagógico y las "
        "normativas institucionales vigentes."
    )
    return {
        "intro": intro,
        "recomendacion": recomendacion,
        "disclaimer": disclaimer,
    }


def _fallback_for_terapeuta(prob: float) -> Dict[str, str]:
    if prob < 0.33:
        intro = "El perfil actual se alinea con una necesidad de apoyo clínico baja."
        recomendacion = (
            "Puede resultar suficiente mantener el plan de intervención vigente, monitoreando "
            "funcionamiento adaptativo, participación escolar y bienestar familiar."
        )
    elif prob < 0.66:
        intro = "El perfil sugiere una necesidad de apoyo clínico moderada."
        recomendacion = (
            "Se recomienda revisar objetivos específicos, ajustar la intensidad de las intervenciones "
            "y fortalecer canales de comunicación con escuela y familia para detectar cambios tempranos."
        )
    else:
        intro = "El perfil indica una necesidad de apoyo clínico alta."
        recomendacion = (
            "Resulta pertinente priorizar objetivos críticos, coordinar acciones interdisciplinarias "
            "y documentar de manera sistemática la evolución en diferentes contextos (hogar, escuela, "
            "espacios terapéuticos)."
        )

    disclaimer = (
        "Esta sugerencia es un apoyo orientativo para la práctica profesional y no sustituye la "
        "evaluación clínica ni las guías de práctica basadas en evidencia."
    )
    return {
        "intro": intro,
        "recomendacion": recomendacion,
        "disclaimer": disclaimer,
    }


# -------------------------------------------------------------------
# Función principal que usan los paneles
# -------------------------------------------------------------------


def get_recommendation(
    rol: str,
    prob: float,
    features_row,
    shap_values_row=None,
) -> Dict[str, str]:
    """
    Punto de entrada único desde los paneles.
    1) Construye una query según rol + prob
    2) Recupera fragmentos de la base de conocimiento (RAG-light)
    3) Si no hay docs o similitud baja, usa un fallback basado en reglas
    """

    rag = SimpleRAG(rol)
    query = build_query(rol, prob, features_row, shap_values_row)
    retrieved_docs = rag.retrieve(query, top_k=2) if rag.has_docs() else []

    # Fallback base según rol
    if rol == "familia":
        base = _fallback_for_familia(prob)
    elif rol == "docente":
        base = _fallback_for_docente(prob)
    else:
        base = _fallback_for_terapeuta(prob)

    # Si no encontramos docs relevantes, devolvemos solo el fallback
    if not retrieved_docs:
        return base

    # Tomamos los mejores documentos y extraemos sus "ideas fuerza"
    highlights = []
    for doc in retrieved_docs:
        h = extract_highlights(doc.text, max_sentences=2)
        if h:
            highlights.append(h)

    # Si por algún motivo no pudimos extraer nada, devolvemos el fallback
    if not highlights:
        return base

    # Integramos el RAG-light con el texto base
    joined_highlights = " ".join(highlights)

    # Ajuste de mensaje según rol (solo cambiamos la parte de recomendación)
    base_intro = base["intro"]
    base_disclaimer = base["disclaimer"]

    if rol == "familia":
        recomendacion = (
            "A partir de las orientaciones disponibles para familias, podrían ser útiles acciones como: "
            f"{joined_highlights}"
        )
    elif rol == "docente":
        recomendacion = (
            "Considerando las guías pedagógicas disponibles, se sugiere priorizar estrategias tales como: "
            f"{joined_highlights}"
        )
    else:
        recomendacion = (
            "De acuerdo con las guías clínicas relacionadas, podrían priorizarse intervenciones como: "
            f"{joined_highlights}"
        )

    return {
        "intro": base_intro,
        "recomendacion": recomendacion,
        "disclaimer": base_disclaimer,
    }