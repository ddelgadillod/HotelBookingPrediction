from fastapi import APIRouter, Request

from app.schemas import ReservaHotelInput, PrediccionTarifaOutput
from app.preprocessing import transformar_real

router = APIRouter(tags=["Regresión — Tarifa diaria"])


@router.post("/predict/tarifa", response_model=PrediccionTarifaOutput)
def predecir_tarifa(payload: ReservaHotelInput, request: Request):
    """
    Predice la tarifa diaria (tarifa_diaria / ADR) de una reserva de hotel.

    Recibe los 20 campos crudos de la reserva sin preprocesar. Internamente
    se aplica el preprocesamiento antes de invocar al modelo.
    """
    modelo_info = request.app.state.modelos["regresion"]
    modelo = modelo_info["modelo"]
    prep = modelo_info["preprocesamiento"]

    X_input = transformar_real(payload.model_dump(), prep)
    prediccion = float(modelo.predict(X_input)[0])

    return PrediccionTarifaOutput(
        tarifa_diaria_predicha=round(prediccion, 2),
        modelo=modelo_info["algoritmo"],
        metrica_validacion={modelo_info["metrica"]: modelo_info["valor_metrica"]},
    )
