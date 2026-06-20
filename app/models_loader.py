"""
Carga los modelos serializados y el preprocesamiento real (Act9) UNA SOLA
VEZ al iniciar el proceso.

A diferencia del patrón del ejemplo de referencia del curso (donde el modelo
se reentrenaba en cada request GET), aquí los .joblib se cargan en el evento
de startup de FastAPI (ver app/main.py) y se guardan en memoria durante toda
la vida del proceso. Cada request de predicción solo hace .predict(), nunca
relee disco ni reentrena.

A partir de esta versión, cada rama carga DOS objetos por separado:
  - el modelo entrenado (XGBoost), en model/modelo_*.joblib
  - el preprocesamiento real de Act9 (RobustScaler + esquema de columnas),
    en artefactos_preprocesamiento/preprocesamiento_real_*.joblib
En versiones anteriores ambos vivían juntos dentro de un único Pipeline de
sklearn; ahora están separados porque el preprocesamiento real (get_dummies +
RobustScaler fit-eado en Act9) no es un objeto sklearn componible en un
Pipeline estándar.
"""
import glob
import os
import joblib

from app.preprocessing import cargar_preprocesamiento_real

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")


def _ultimo_archivo(patron: str) -> str:
    """Si hay varios .joblib que matchean el patrón (ej. tras reemplazar por
    modelos reentrenados), toma el más reciente por fecha de modificación."""
    candidatos = glob.glob(os.path.join(MODEL_DIR, patron))
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró ningún archivo que matchee '{patron}' en {MODEL_DIR}"
        )
    return max(candidatos, key=os.path.getmtime)


def cargar_modelos() -> dict:
    """Devuelve un dict con el modelo y el preprocesamiento real de cada rama,
    listo para guardar en app.state."""
    path_reg = _ultimo_archivo("modelo_regresion_tarifa_*.joblib")
    path_clf = _ultimo_archivo("modelo_clasificacion_cancelacion_*.joblib")

    data_reg = joblib.load(path_reg)
    data_clf = joblib.load(path_clf)

    prep_reg = cargar_preprocesamiento_real("regresion")
    prep_clf = cargar_preprocesamiento_real("clasificacion")

    return {
        "regresion": {
            "modelo": data_reg["modelo"],
            "preprocesamiento": prep_reg,
            "algoritmo": data_reg["algoritmo"],
            "metrica": data_reg["metrica"],
            "valor_metrica": data_reg["valor_metrica"],
            "archivo": os.path.basename(path_reg),
        },
        "clasificacion": {
            "modelo": data_clf["modelo"],
            "preprocesamiento": prep_clf,
            "algoritmo": data_clf["algoritmo"],
            "metrica": data_clf["metrica"],
            "valor_metrica": data_clf["valor_metrica"],
            "umbral_decision": data_clf["umbral_decision"],
            "archivo": os.path.basename(path_clf),
        },
    }
