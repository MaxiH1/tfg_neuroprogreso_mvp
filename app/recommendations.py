# app/recommendations.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------------------
# Config: ubicación de la base de conocimiento
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # raíz del repo
KB_DIR = BASE_DIR / "knowledge_base"

# rol interno -> subcarpeta dentro de knowledge_base
ROLE_DIRS = {
    "familia": "family_guidelines",
    "docente": "docente",        # los usaremos más adelante
    "terapeuta": "profesional",  # idem
}


# -------------------------------------------------------------------
# SimpleRetriever: TF-IDF + coseno
# -------------------------------------------------------------------
class SimpleRetriever:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.docs: List[str] = []
        self.titles: List[str] = []

        # Cargamos .md y .txt
        if self.base_dir.exists():
            for path in self.base_dir.rglob("*.md"):
                self.titles.append(path.stem)
                self.docs.append(path.read_text(encoding="utf-8"))
            for path in self.base_dir.rglob("*.txt"):
                self.titles.append(path.stem)
                self.docs.append(path.read_text(encoding="utf-8"))

        # Si no hay docs, dejamos el vectorizador vacío
        if self.docs:
            self.vectorizer = TfidfVectorizer()
            self.doc_matrix = self.vectorizer.fit_transform(self.docs)
        else:
            self.vectorizer = None
            self.doc_matrix = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Devuelve lista de (título, texto) ordenada por similitud."""
        if not self.docs or self.vectorizer is None or self.doc_matrix is None:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [(self.titles[i], self.docs[i]) for i in idxs]


# Caché simple en memoria para no recalcular TF-IDF todo el tiempo
_RETRIEVERS: Dict[str, SimpleRetriever | None] = {}


def _get_retriever_for_role(rol: str) -> SimpleRetriever | None:
    """Devuelve un retriever para el rol interno ('familia', 'docente', 'terapeuta')."""
    if rol not in ROLE_DIRS:
        return None

    if rol in _RETRIEVERS:
        return _RETRIEVERS[rol]

    subdir = ROLE_DIRS[rol]
    base = KB_DIR / subdir
    if base.exists():
        _RETRIEVERS[rol] = SimpleRetriever(base)
    else:
        _RETRIEVERS[rol] = None

    return _RETRIEVERS[rol]


# -------------------------------------------------------------------
# Constructores de queries según rol (por ahora, solo familia)
# -------------------------------------------------------------------
def _build_query_familia(prob: float, features_row) -> str:
    """
    Armamos una consulta simple en texto para buscar en los documentos de familia.
    No hace falta que sea perfecta, solo razonable.
    """
    partes = ["apoyos en casa", "rutinas", "comunicación con la escuela"]

    # Ajustamos según nivel de riesgo del modelo
    if prob >= 0.7:
        partes.append("mayor necesidad de apoyo")
        partes.append("coordinación con escuela y terapeutas")
    elif prob >= 0.4:
        partes.append("apoyo moderado")
        partes.append("seguimiento cercano")

    # Usamos algunas features para matizar la consulta (best effort)
    try:
        rrb_val = float(features_row.get("RRB", 0.0))
        if rrb_val > 5:
            partes.append("conductas repetitivas")
    except Exception:
        pass

    try:
        aff_val = float(features_row.get("affect_total", 0.0))
        if aff_val > 15:
            partes.append("interacción social y comunicación")
    except Exception:
        pass

    return " ".join(partes)


# -------------------------------------------------------------------
# Recomendación para FAMILIA (RAG-light)
# -------------------------------------------------------------------
def _recommend_familia(prob: float, features_row, shap_values_row) -> Dict[str, Any]:
    retriever = _get_retriever_for_role("familia")

    # 1) Si no hay retriever o docs, usamos el fallback clásico
    if retriever is None:
        return _fallback_familia(prob)

    query = _build_query_familia(prob, features_row)
    docs = retriever.retrieve(query, top_k=3)

    if not docs:
        # Si por alguna razón no se recupera nada, también fallback
        return _fallback_familia(prob)

    # 2) Tomamos la primera línea de cada documento como “idea clave”
    bullets: List[str] = []
    for title, body in docs:
        first_line = body.strip().splitlines()[0]
        bullets.append(f"- {first_line}")

    intro = (
        "Según el perfil de riesgo estimado por el modelo y la información disponible, "
        "se seleccionaron orientaciones de la biblioteca para familias relacionadas con "
        "rutinas, organización cotidiana y comunicación con la escuela."
    )

    recomendacion = (
        "Podría ser útil considerar las siguientes acciones en el hogar:\n\n"
        + "\n".join(bullets)
    )

    disclaimer = (
        "Esta sugerencia se apoya en materiales de referencia para familias y en la salida del modelo, "
        "pero siempre debe revisarse y adaptarse junto con los profesionales que acompañan al niño o la niña."
    )

    return {
        "intro": intro,
        "recomendacion": recomendacion,
        "disclaimer": disclaimer,
    }


def _fallback_familia(prob: float) -> Dict[str, str]:
    """Versión de respaldo si todavía no hay docs o algo falla."""

    if prob < 0.33:
        intro = (
            "Tu hijo/a está atravesando un momento relativamente favorable. "
            "Mantener las rutinas que están funcionando ayuda a sostener este progreso."
        )
    elif prob < 0.66:
        intro = (
            "Tu hijo/a podría necesitar un poco más de apoyo en este tiempo. "
            "Pequeños cambios en casa y una buena comunicación con la escuela pueden marcar diferencia."
        )
    else:
        intro = (
            "Tu hijo/a está atravesando un momento que requiere un acompañamiento más cercano. "
            "Resulta útil coordinar acciones entre la familia, la escuela y los profesionales."
        )

    recomendacion = (
        "Podés observar qué situaciones resultan más difíciles en el día a día, anotar ejemplos concretos "
        "y compartirlos con la escuela o el equipo terapéutico para pensar estrategias en conjunto."
    )

    disclaimer = (
        "Esta recomendación es orientativa y no reemplaza la evaluación ni las decisiones de los "
        "profesionales de la salud o de la educación."
    )

    return {
        "intro": intro,
        "recomendacion": recomendacion,
        "disclaimer": disclaimer,
    }


# -------------------------------------------------------------------
# Recomendaciones para DOCENTE y TERAPEUTA (por ahora, plantillas)
# -------------------------------------------------------------------
def _recommend_docente(prob: float, features_row, shap_values_row) -> Dict[str, str]:
    """
    De momento dejamos una plantilla “clásica”.
    Más adelante podemos hacer otro RAG-light con knowledge_base/docente.
    """
    intro = (
        "Según los datos recientes, podría ser útil tener en cuenta apoyos pedagógicos adicionales "
        "y revisar la organización del aula para este estudiante."
    )

    recomendacion = (
        "Observar qué consignas generan mayor dificultad, graduar la complejidad de las tareas y "
        "acordar con la familia pequeñas metas alcanzables, manteniendo una comunicación regular "
        "sobre avances y dificultades."
    )

    disclaimer = (
        "Esta sugerencia es orientativa y debe complementarse con el criterio docente y las políticas "
        "institucionales de la escuela."
    )

    return {
        "intro": intro,
        "recomendacion": recomendacion,
        "disclaimer": disclaimer,
    }


def _recommend_terapeuta(prob: float, features_row, shap_values_row) -> Dict[str, str]:
    intro = (
        "A partir del perfil de riesgo estimado por el modelo y las características observadas, "
        "podría ser pertinente revisar la formulación clínica breve y priorizar objetivos específicos."
    )

    recomendacion = (
        "Se sugiere identificar una o dos metas centrales a corto plazo, coordinar con la escuela y la familia "
        "las estrategias principales, y documentar cambios en funcionamiento adaptativo, sueño y conducta."
    )

    disclaimer = (
        "Estas sugerencias son un apoyo orientativo para la práctica profesional y no sustituyen "
        "la evaluación clínica ni las guías de práctica basadas en evidencia."
    )

    return {
        "intro": intro,
        "recomendacion": recomendacion,
        "disclaimer": disclaimer,
    }


# -------------------------------------------------------------------
# API pública usada por los paneles
# -------------------------------------------------------------------
def get_recommendation(
    rol: str,
    prob: float,
    features_row,
    shap_values_row,
) -> Dict[str, Any]:
    """
    rol: 'familia', 'docente' o 'terapeuta' (profesional).
    """
    rol = (rol or "").lower()

    if rol == "familia":
        return _recommend_familia(prob, features_row, shap_values_row)

    if rol == "docente":
        return _recommend_docente(prob, features_row, shap_values_row)

    # Cualquier otro lo tratamos como profesional de la salud
    return _recommend_terapeuta(prob, features_row, shap_values_row)