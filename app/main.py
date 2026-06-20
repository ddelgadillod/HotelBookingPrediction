"""
Servicio FastAPI — API de acceso a los modelos de aprendizaje supervisado.

Expone dos endpoints:
  POST /predict/tarifa       — regresión, predice tarifa_diaria (USD)
  POST /predict/cancelacion  — clasificación, predice cancelado (0/1) + probabilidad

Y una interfaz de usuario opcional:
  GET  /app                  — panel HTML que consulta ambos endpoints a la vez

Modelos finales (hiperparámetros óptimos de Act20) entrenados sobre el
preprocesamiento REAL de Act9 (RobustScaler + pd.get_dummies ya ajustados,
serializados en artefactos_preprocesamiento/). Modelo y preprocesamiento se
cargan UNA SOLA VEZ al iniciar el proceso (evento lifespan), no en cada request.

Ejecutar localmente:
    uvicorn app.main:app --reload --port 8000

Interfaz visual:
    http://127.0.0.1:8000/app

Documentación interactiva (Swagger UI):
    http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models_loader import cargar_modelos
from app.routers import tarifa, cancelacion

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


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
    version="2.1.0",
    lifespan=lifespan,
)

app.include_router(tarifa.router)
app.include_router(cancelacion.router)

# Sirve cualquier asset adicional (CSS/JS/imágenes) que se agregue a static/
# bajo /static/*. La página principal de la interfaz se expone en /app
# (ver más abajo), no aquí, para no chocar con el endpoint de salud en "/".
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["Salud"])
def root():
    """Endpoint de salud — confirma que el servicio está activo."""
    return {"status": "ok", "servicio": "Hotel API", "docs": "/docs", "interfaz": "/app"}


@app.get("/health", tags=["Salud"])
def health():
    """Verifica que ambos modelos estén cargados en memoria."""
    return {"status": "ok"}


@app.get("/app", tags=["Interfaz"])
def interfaz():
    """Panel HTML — formulario único que consulta /predict/tarifa y
    /predict/cancelacion a la vez y muestra ambos resultados juntos."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))