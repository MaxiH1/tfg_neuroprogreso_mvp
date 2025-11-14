import yaml
import numpy as np
import os

# ---------------------------------------------------------
# Cargar YAML con reglas
# ---------------------------------------------------------

def load_rules():
    yaml_path = "knowledge_base/rules/rule_based_recommendations.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


# ---------------------------------------------------------
# Clasificador de perfiles según probabilidad y variables
# ---------------------------------------------------------

def clasificar_perfil(prob, features_row):
    """
    Devuelve un perfil simple: apoyo_leve / apoyo_moderado / apoyo_focalizado / apoyo_intensivo
    """
    RRB = features_row["RRB"]
    overall = features_row["overall_total"]

    # Reglas simples basadas en probabilidad
    if prob < 0.40:
        return "apoyo_leve"
    elif 0.40 <= prob < 0.70:
        return "apoyo_moderado"
    elif 0.70 <= prob < 0.90:
        return "apoyo_focalizado"
    else:
        return "apoyo_intensivo"


# ---------------------------------------------------------
# Variable clave según SHAP (tomamos el feature con mayor contribución positiva)
# ---------------------------------------------------------

def variable_clave_por_shap(features_row, shap_values_row):
    """
    Retorna la variable con impacto positivo más fuerte en la predicción.
    """
    # shap_values_row es un array con un valor por cada feature
    idx_max = np.argmax(shap_values_row)
    variable = features_row.index[idx_max]
    return variable


# ---------------------------------------------------------
# Clasificar nivel de una variable (bajo, medio, alto)
# ---------------------------------------------------------

def clasificar_nivel_variable(variable, valor):
    """
    Clasificación genérica basada en terciles simples.
    Se puede ajustar según distribución real.
    """
    if variable == "RRB":
        if valor <= 4:
            return "bajo"
        elif valor <= 10:
            return "medio"
        else:
            return "alto"

    if variable == "overall_total":
        if valor <= 10:
            return "bajo"
        elif valor <= 20:
            return "medio"
        else:
            return "alto"

    if variable == "affect_total":
        if valor <= 6:
            return "bajo"
        elif valor <= 12:
            return "medio"
        else:
            return "alto"

    # fallback genérico
    if valor < np.median(valor):
        return "bajo"
    else:
        return "alto"


# ---------------------------------------------------------
# Buscar regla adecuada en YAML
# ---------------------------------------------------------

def buscar_regla(rules_data, perfil, rol, variable_clave, nivel):
    """
    Busca la regla que coincide con perfil + rol + variable_clave + nivel.
    Si no encuentra coincidencia exacta, devuelve None.
    """
    for regla in rules_data["reglas"]:
        if (
            regla["perfil"] == perfil and
            regla["rol"] == rol and
            regla["condicion"]["variable_clave"] == variable_clave and
            regla["condicion"]["nivel"] == nivel
        ):
            return regla
    return None


# ---------------------------------------------------------
# Función principal: generar UNA recomendación personalizada
# ---------------------------------------------------------

def get_recommendation(rol, prob, features_row, shap_values_row):
    """
    Devuelve un diccionario con:
    - intro
    - recomendacion
    - disclaimer
    """

    # 1) Cargar reglas
    rules_data = load_rules()

    # 2) Determinar el perfil
    perfil = clasificar_perfil(prob, features_row)

    # 3) Determinar variable clave según SHAP
    variable_clave = variable_clave_por_shap(features_row, shap_values_row)

    # 4) Determinar nivel (bajo, medio, alto)
    valor = features_row[variable_clave]
    nivel = clasificar_nivel_variable(variable_clave, valor)

    # 5) Buscar regla
    regla = buscar_regla(rules_data, perfil, rol, variable_clave, nivel)

    # Si no hay regla exacta → fallback
    if regla is None:
        return {
            "intro": "Según los datos recientes, podría ser útil tener en cuenta apoyos adicionales.",
            "recomendacion": "Observar qué situaciones generan mayor dificultad y ajustar el ambiente para facilitar la participación del niño.",
            "disclaimer": "Esta sugerencia es orientativa y debe complementarse con el criterio de los profesionales que acompañan al niño."
        }

    # 6) Construir salida final desde la regla
    return {
        "intro": regla["intro_base"],
        "recomendacion": regla["recomendacion_base"],
        "disclaimer": "Esta sugerencia es orientativa y debe complementarse con el criterio de los profesionales que acompañan al niño."
    }