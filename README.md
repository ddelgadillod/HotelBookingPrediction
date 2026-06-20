# API de Predicción Hotel Booking

API REST construida con **FastAPI** que expone dos modelos de aprendizaje
supervisado entrenados sobre el dataset *Hotel Booking Demand*:

- **`POST /predict/tarifa`** — Regresión. Predice la tarifa diaria (`tarifa_diaria` / ADR) de una reserva.
- **`POST /predict/cancelacion`** — Clasificación. Predice si una reserva será cancelada, junto con la probabilidad.

Este proyecto corresponde al **Taller 4 / Actividad 25** del curso de Aprendizaje
de Máquina Aplicado (Maestría CDAI, Universidad del Valle). Los modelos son
los **finales** (hiperparámetros óptimos encontrados en
`Act20_ML_DiegoDelgadillo_HotelBooking_final.ipynb` vía RandomizedSearchCV +
GridSearchCV de refinamiento), entrenados sobre el **preprocesamiento real**
de `Act9_ML_DiegoDelgadillo_HotelBooking_final.ipynb` (Taller 1).

## Arquitectura

```
app/
├── main.py              # FastAPI app + lifespan (carga modelos al iniciar)
├── schemas.py            # Modelos Pydantic (contrato de datos)
├── models_loader.py       # Carga modelo + preprocesamiento real de cada rama
├── preprocessing.py       # Replica el preprocesamiento REAL de Act9
│                           # (pd.get_dummies + RobustScaler ya ajustados)
└── routers/
    ├── tarifa.py          # Endpoint de regresión
    └── cancelacion.py     # Endpoint de clasificación

model/                          # Modelos entrenados (.joblib) — solo el estimador
artefactos_preprocesamiento/    # RobustScaler real + esquema de columnas (Act9)
tests/                          # Pruebas de funcionamiento + datasets de prueba
```

### Por qué el modelo y el preprocesamiento están separados

A diferencia de una primera versión de este proyecto (que envolvía todo en un
único `Pipeline` de scikit-learn), aquí el preprocesamiento real de Act9 usa
`pd.get_dummies()` y un `RobustScaler` ajustado **sobre todo el dataset antes
del split train/test** — no son objetos componibles dentro de un `Pipeline`
estándar de sklearn de la misma forma que un `ColumnTransformer`. Por eso:

- `model/*.joblib` contiene solo el estimador entrenado (XGBoost) + metadata.
- `artefactos_preprocesamiento/*.joblib` contiene el `RobustScaler` ya
  fit-eado en Act9 y el esquema exacto de columnas que produjo
  `pd.get_dummies()`, necesario para alinear cualquier payload nuevo al
  mismo orden y mismas columnas que vio el modelo en entrenamiento.
- `app/preprocessing.py::transformar_real()` aplica ambos en el orden
  correcto: imputación → `habitacion_cambiada` → `get_dummies` alineado al
  esquema fijo (categorías no vistas quedan con todas sus dummies en 0,
  equivalente a `handle_unknown='ignore'`) → `RobustScaler.transform()`.

Los modelos se cargan **una sola vez** al iniciar el proceso (evento
`lifespan` de FastAPI) y se mantienen en `app.state.modelos` durante toda
la vida del servicio — nunca se relee disco ni se reentrena en cada request.

### Nota sobre versiones de librerías

`requirements.txt` fija `pandas==2.2.2` y `scikit-learn==1.6.1` porque son
las versiones exactas usadas en la sesión de Google Colab donde se generó
`artefactos_preprocesamiento/*.joblib` (el `RobustScaler` real de Act9). Una
versión distinta de scikit-learn puede cargar el `.joblib` con un
`InconsistentVersionWarning` (no rompe, pero sklearn no garantiza el
resultado) — se evita fijando la misma versión. `imbalanced-learn==0.14.0`
es la primera versión con soporte confirmado para scikit-learn 1.6.

### Nota sobre el leakage de escala heredado de Act9

El `RobustScaler` de Act9 se ajustó sobre **todo** `X_r`/`X_c` antes de
separar train/test (no solo sobre train). Esto es un leakage menor de
escala, heredado deliberadamente para mantener trazabilidad exacta con las
entregas previas del curso (Talleres 1, 2 y 3). No se corrigió en esta
versión por decisión explícita de seguir la línea de las entregas.

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

## Despliegue en Render (free tier)

1. Subir este repositorio a GitHub — **incluir** la carpeta
   `artefactos_preprocesamiento/` junto con `model/`, ambas son necesarias
   para que el API funcione.
2. En Render: **New > Web Service**, conectar el repo.
3. Render detecta `render.yaml` automáticamente, o configurar manualmente:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. El servicio queda disponible en una URL pública HTTPS.

**Nota sobre cold starts:** el free tier de Render duerme el servicio tras
15 minutos sin tráfico. La primera petición tras inactividad puede tardar
30-60 segundos en responder mientras el contenedor arranca de nuevo.

## Reentrenar desde cero

```bash
python3 reentrenar_y_serializar.py /ruta/a/hotel_bookings.csv
```

Este script replica el preprocesamiento de Act9 desde el CSV crudo, entrena
ambos modelos con los hiperparámetros óptimos de Act20, y serializa tanto
los modelos (`model/`) como los objetos de preprocesamiento
(`artefactos_preprocesamiento/`). `models_loader.py` toma automáticamente
los archivos más recientes por fecha de modificación si hay varios.

## Reemplazo de los modelos finales (si se vuelve a optimizar)

1. Re-ejecutar `reentrenar_y_serializar.py` con nuevos hiperparámetros, **o**
2. Si el preprocesamiento de Act9 cambia (nuevas variables, otro split),
   regenerar también `artefactos_preprocesamiento/*.joblib` — el esquema de
   columnas y el scaler deben corresponder exactamente al modelo con el que
   se empareja.
3. El contrato de datos (`app/schemas.py`) no debería requerir cambios
   mientras las categorías observadas en el dataset no cambien.

