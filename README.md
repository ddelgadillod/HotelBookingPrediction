# API de Predicción Hotel Booking

API REST construida con **FastAPI** que expone dos modelos de aprendizaje
supervisado entrenados sobre el dataset *Hotel Booking Demand*:

- **`POST /predict/tarifa`** — Regresión. Predice la tarifa diaria (`tarifa_diaria` / ADR) de una reserva.
- **`POST /predict/cancelacion`** — Clasificación. Predice si una reserva será cancelada, junto con la probabilidad.

Este proyecto corresponde a la Actividad 25 del curso de Aprendizaje
de Máquina Aplicado (Maestría CDAI, Universidad del Valle). Los modelos son
los finales (hiperparámetros óptimos vía RandomizedSearchCV +
GridSearchCV de refinamiento).




## Ejecutar localmente

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Documentación interactiva (Swagger UI): http://127.0.0.1:8000/docs

### Ejemplo de payload — `POST /predict/tarifa`

```json
{
  "dias_anticipacion": 192, "semana_llegada": 39, "dia_llegada": 19,
  "noches_fin_semana": 4, "noches_semana": 10, "adultos": 2, "ninos": 0,
  "cancelaciones_previas": 0, "cambios_reserva": 0, "dias_lista_espera": 0,
  "solicitudes_especiales": 2, "tipo_hotel": "Resort Hotel",
  "mes_llegada": "September", "regimen_alimenticio": "BB",
  "segmento_mercado": "Offline TA/TO", "canal_distribucion": "TA/TO",
  "tipo_deposito": "No Deposit", "tipo_cliente": "Contract",
  "tipo_hab_reservada": "D", "tipo_hab_asignada": "D"
}
```

Respuesta:

```json
{
  "tarifa_diaria_predicha": 68.25,
  "moneda": "USD",
  "modelo": "XGBoost",
  "metrica_validacion": {"R2": 0.7451}
}
```

El mismo payload contra `/predict/cancelacion` devuelve:

```json
{
  "cancelado_predicho": 0,
  "probabilidad_cancelacion": 0.0057,
  "modelo": "XGBoost",
  "metrica_validacion": {"F1": 0.6119}
}
```

## Pruebas de funcionamiento

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_api.py -v
```

Las pruebas:
1. Verifican que `/` y `/health` respondan.
2. Envían los 20 casos de `tests/dataset_prueba_api_regresion.csv` y
   `tests/dataset_prueba_api_clasificacion.csv` al API y comparan la
   respuesta contra la predicción calculada con el mismo preprocesamiento
   real + modelo final (deben coincidir casi exacto, mismo proceso).
3. Confirman que payloads con categorías inválidas o campos faltantes
   sean rechazados con HTTP 422 (validación automática de Pydantic).
4. Confirman que categorías válidas pero poco frecuentes no rompen el
   preprocesamiento manual (`pd.get_dummies` + alineación de esquema).
