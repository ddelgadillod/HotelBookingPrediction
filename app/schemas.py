"""
Esquemas Pydantic — traducción directa del contrato de datos generado en
Taller3_ModelosPrueba_Contrato_DiegoDelgadillo.ipynb (contrato_datos_schema.json).

Los valores de los Literal son exactamente los observados en el dataset
histórico (hotel_bookings.csv). Si el cliente envía un valor fuera de estos
literales, FastAPI devuelve un 422 automáticamente, sin validación manual.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ReservaHotelInput(BaseModel):
    """Payload crudo de una reserva de hotel — sin ningún preprocesamiento."""

    # ── Numéricas ────────────────────────────────────────────────────────
    dias_anticipacion: int = Field(..., ge=0, le=750,
        description="Días entre la reserva y la llegada. Observado: [0, 737]")
    semana_llegada: int = Field(..., ge=1, le=53,
        description="Semana del año de llegada. Observado: [1, 53]")
    dia_llegada: int = Field(..., ge=1, le=31,
        description="Día del mes de llegada. Observado: [1, 31]")
    noches_fin_semana: int = Field(..., ge=0, le=20,
        description="Noches de fin de semana reservadas. Observado: [0, 19]")
    noches_semana: int = Field(..., ge=0, le=60,
        description="Noches de semana reservadas. Observado: [0, 50]")
    adultos: int = Field(..., ge=0, le=60,
        description="Número de adultos. Observado: [0, 55] (0 se imputa a 1)")
    ninos: Optional[int] = Field(None, ge=0, le=12,
        description="Número de niños. Acepta null (se imputa a 0). Observado: [0, 10]")
    cancelaciones_previas: int = Field(..., ge=0, le=30,
        description="Cancelaciones previas del cliente. Observado: [0, 26]")
    cambios_reserva: int = Field(..., ge=0, le=25,
        description="Cambios realizados sobre la reserva. Observado: [0, 21]")
    dias_lista_espera: int = Field(..., ge=0, le=400,
        description="Días en lista de espera. Observado: [0, 391]")
    solicitudes_especiales: int = Field(..., ge=0, le=6,
        description="Número de solicitudes especiales. Observado: [0, 5]")

    # ── Categóricas ──────────────────────────────────────────────────────
    # tipo_hotel y mes_llegada se reciben (parte del contexto de la reserva)
    # pero el modelo entrenado NO las usa como predictoras.
    tipo_hotel: Literal["City Hotel", "Resort Hotel"]
    mes_llegada: Literal[
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    regimen_alimenticio: Literal["BB", "FB", "HB", "SC", "Undefined"]
    segmento_mercado: Literal[
        "Aviation", "Complementary", "Corporate", "Direct",
        "Groups", "Offline TA/TO", "Online TA", "Undefined"
    ]
    canal_distribucion: Literal["Corporate", "Direct", "GDS", "TA/TO", "Undefined"]
    tipo_deposito: Literal["No Deposit", "Non Refund", "Refundable"]
    tipo_cliente: Literal["Contract", "Group", "Transient", "Transient-Party"]
    tipo_hab_reservada: Literal["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    tipo_hab_asignada: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "dias_anticipacion": 192, "semana_llegada": 39, "dia_llegada": 19,
                "noches_fin_semana": 4, "noches_semana": 10, "adultos": 2, "ninos": 0,
                "cancelaciones_previas": 0, "cambios_reserva": 0, "dias_lista_espera": 0,
                "solicitudes_especiales": 2, "tipo_hotel": "Resort Hotel",
                "mes_llegada": "September", "regimen_alimenticio": "BB",
                "segmento_mercado": "Offline TA/TO", "canal_distribucion": "TA/TO",
                "tipo_deposito": "No Deposit", "tipo_cliente": "Contract",
                "tipo_hab_reservada": "D", "tipo_hab_asignada": "D",
            }
        }
    }


class PrediccionTarifaOutput(BaseModel):
    tarifa_diaria_predicha: float = Field(..., description="Tarifa diaria predicha en USD")
    moneda: str = "USD"
    modelo: str
    metrica_validacion: dict


class PrediccionCancelacionOutput(BaseModel):
    cancelado_predicho: int = Field(..., description="0 = no cancela, 1 = cancela")
    probabilidad_cancelacion: float = Field(..., ge=0, le=1)
    modelo: str
    metrica_validacion: dict
