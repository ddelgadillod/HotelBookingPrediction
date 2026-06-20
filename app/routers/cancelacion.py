from fastapi import APIRouter, Request

from app.schemas import ReservaHotelInput, PrediccionCancelacionOutput
from app.preprocessing import transformar_real

router = APIRouter(tags=["Clasificación — Cancelación"])


@router.post("/predict/cancelacion", response_model=PrediccionCancelacionOutput)
def predecir_cancelacion(payload: ReservaHotelInput, request: Request):
    """
    Predice si una reserva de hotel será cancelada.

    Recibe los  20 campos crudos de la reserva. Internamente se
    aplica el preprocesamiento. La
    decisión final usa el umbral óptimo encontrado 0.43.
    """
    modelo_info = request.app.state.modelos["clasificacion"]
    modelo = modelo_info["modelo"]
    prep = modelo_info["preprocesamiento"]
    umbral = modelo_info["umbral_decision"]

    X_input = transformar_real(payload.model_dump(), prep)
    proba = float(modelo.predict_proba(X_input)[0, 1])
    pred = int(proba >= umbral)

    return PrediccionCancelacionOutput(
        cancelado_predicho=pred,
        probabilidad_cancelacion=round(proba, 4),
        modelo=modelo_info["algoritmo"],
        metrica_validacion={modelo_info["metrica"]: modelo_info["valor_metrica"]},
    )
