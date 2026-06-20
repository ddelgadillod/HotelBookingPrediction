"""
Servicio FastAPI — API de acceso a los modelos de aprendizaje supervisado.

Expone dos endpoints:
  POST /predict/tarifa       — regresión, predice tarifa_diaria (USD)
  POST /predict/cancelacion  — clasificación, predice cancelado (0/1) + probabilidad

Modelos finales (hiperparámetros óptimos de Act20) entrenados sobre el
preprocesamiento REAL de Act9 (RobustScaler + pd.get_dummies ya ajustados,
serializados en artefactos_preprocesamiento/). Modelo y preprocesamiento se
cargan UNA SOLA VEZ al iniciar el proceso (evento lifespan), no en cada request.

Ejecutar localmente:
    uvicorn app.main:app --reload --port 8000

Documentación interactiva (Swagger UI):
    http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.models_loader import cargar_modelos
from app.routers import tarifa, cancelacion


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: cargar modelos + preprocesamiento real en memoria
    app.state.modelos = cargar_modelos()
    print("Modelos cargados:")
    for clave, info in app.state.modelos.items():
        extra = f", umbral={info['umbral_decision']}" if "umbral_decision" in info else ""
        print(f"  {clave}: {info['algoritmo']} ({info['metrica']}={info['valor_metrica']}{extra}) "
              f"<- {info['archivo']}")
    yield
    # Shutdown: nada que liberar explícitamente (joblib libera con el proceso)


app = FastAPI(
    title="API de Predicción Hotel Booking",
    description=(
        "API de acceso a modelos de aprendizaje supervisado entrenados sobre "
        "el dataset Hotel Booking Demand. Modelos finales (XGBoost, "
        "hiperparámetros óptimos)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(tarifa.router)
app.include_router(cancelacion.router)


@app.get("/", tags=["Salud"])
def root():
    """Endpoint de salud — confirma que el servicio está activo."""
    return {"status": "ok", "servicio": "Hotel API", "docs": "/docs"}


@app.get("/health", tags=["Salud"])
def health():
    """Verifica que ambos modelos estén cargados en memoria."""
    return {"status": "ok"}
