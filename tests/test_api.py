"""
Pruebas de funcionamiento del API (cubre el punto 7 del Taller 4 / Act. 25).

Levanta el servicio FastAPI con TestClient (no necesita un servidor corriendo
aparte), envía cada fila de los datasets de prueba (20 casos por rama,
tomados del split de TEST, nunca visto en entrenamiento), y compara la
respuesta del API contra la predicción ya calculada con el preprocesamiento
REAL de Act9 + el modelo final (hiperparámetros de Act20).

Ejecutar:
    ./venv_deploy/bin/python -m pytest tests/test_api.py -v
o directamente:
    ./venv_deploy/bin/python tests/test_api.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

VARS_INCLUIDAS_NUM = [
    'dias_anticipacion', 'semana_llegada', 'dia_llegada',
    'noches_fin_semana', 'noches_semana', 'adultos', 'ninos',
    'cancelaciones_previas', 'cambios_reserva',
    'dias_lista_espera', 'solicitudes_especiales'
]
VARS_INCLUIDAS_CAT = [
    'tipo_hotel', 'mes_llegada', 'regimen_alimenticio',
    'segmento_mercado', 'canal_distribucion', 'tipo_deposito',
    'tipo_cliente', 'tipo_hab_reservada', 'tipo_hab_asignada'
]
CAMPOS_PAYLOAD = VARS_INCLUIDAS_NUM + VARS_INCLUIDAS_CAT

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _fila_a_payload(fila: pd.Series) -> dict:
    payload = {}
    for col in CAMPOS_PAYLOAD:
        valor = fila[col]
        if col in VARS_INCLUIDAS_NUM:
            payload[col] = None if pd.isna(valor) else int(valor)
        else:
            payload[col] = str(valor)
    return payload


def test_endpoint_raiz():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200


def test_predict_tarifa_contra_preprocesamiento_real():
    """
    Compara la predicción del API contra tarifa_diaria_pred_pipeline,
    calculada con el mismo preprocesamiento real (Act9) + modelo final
    (Act20) usados por el API. Debe coincidir exacto (mismo proceso,
    mismas versiones de librerías dentro de este venv).
    """
    ruta = os.path.join(TESTS_DIR, "dataset_prueba_api_regresion.csv")
    df = pd.read_csv(ruta)

    with TestClient(app) as client:
        diffs = []
        for _, fila in df.iterrows():
            payload = _fila_a_payload(fila)
            r = client.post("/predict/tarifa", json=payload)
            assert r.status_code == 200, f"Fallo HTTP {r.status_code}: {r.text}"
            pred_api = r.json()["tarifa_diaria_predicha"]
            pred_esperada = fila["tarifa_diaria_pred_pipeline"]
            diffs.append(abs(pred_api - pred_esperada))

        max_diff = max(diffs)
        avg_diff = sum(diffs) / len(diffs)
        print(f"\n[tarifa] {len(df)} casos — diff promedio: {avg_diff:.4f} USD, "
              f"diff máxima: {max_diff:.4f} USD")
        # Tolerancia mínima: mismo preprocesamiento, mismo modelo, mismo proceso
        # de cálculo que generó el CSV de referencia -> debe coincidir casi exacto.
        assert max_diff < 0.5, f"Diferencia inesperadamente alta: {max_diff:.4f} USD"


def test_predict_cancelacion_contra_preprocesamiento_real():
    ruta = os.path.join(TESTS_DIR, "dataset_prueba_api_clasificacion.csv")
    df = pd.read_csv(ruta)

    with TestClient(app) as client:
        aciertos = 0
        diffs_proba = []
        for _, fila in df.iterrows():
            payload = _fila_a_payload(fila)
            r = client.post("/predict/cancelacion", json=payload)
            assert r.status_code == 200, f"Fallo HTTP {r.status_code}: {r.text}"
            resp = r.json()
            pred_api = resp["cancelado_predicho"]
            pred_esperada = int(fila["cancelado_pred_pipeline"])
            proba_api = resp["probabilidad_cancelacion"]
            proba_esperada = fila["cancelado_proba_pipeline"]

            if pred_api == pred_esperada:
                aciertos += 1
            diffs_proba.append(abs(proba_api - proba_esperada))

        avg_diff_proba = sum(diffs_proba) / len(diffs_proba)
        print(f"\n[cancelacion] {len(df)} casos — coincidencia de clase: "
              f"{aciertos}/{len(df)}, diff promedio de probabilidad: {avg_diff_proba:.5f}")
        assert aciertos == len(df), "La clase predicha debería coincidir exacto (mismo proceso)"
        assert avg_diff_proba < 0.001, f"Diferencia de probabilidad inesperadamente alta: {avg_diff_proba:.5f}"


def test_validacion_categoria_invalida():
    """Confirma que Pydantic rechaza categorías fuera del enum con 422."""
    payload_invalido = {
        "dias_anticipacion": 10, "semana_llegada": 10, "dia_llegada": 10,
        "noches_fin_semana": 1, "noches_semana": 2, "adultos": 2, "ninos": 0,
        "cancelaciones_previas": 0, "cambios_reserva": 0, "dias_lista_espera": 0,
        "solicitudes_especiales": 1, "tipo_hotel": "Hotel Que No Existe",
        "mes_llegada": "September", "regimen_alimenticio": "BB",
        "segmento_mercado": "Direct", "canal_distribucion": "Direct",
        "tipo_deposito": "No Deposit", "tipo_cliente": "Transient",
        "tipo_hab_reservada": "A", "tipo_hab_asignada": "A",
    }
    with TestClient(app) as client:
        r = client.post("/predict/tarifa", json=payload_invalido)
        assert r.status_code == 422


def test_validacion_campo_faltante():
    """Confirma que falta un campo requerido también produce 422."""
    payload_incompleto = {"dias_anticipacion": 10}
    with TestClient(app) as client:
        r = client.post("/predict/cancelacion", json=payload_incompleto)
        assert r.status_code == 422


def test_categoria_no_vista_no_falla():
    """
    pd.get_dummies no maneja categorías nuevas de forma nativa, a diferencia
    de OneHotEncoder(handle_unknown='ignore'). transformar_real() debe
    replicar ese comportamiento sin lanzar error: una categoría no presente
    en el esquema de entrenamiento simplemente no activa ninguna dummie.
    Aquí se prueba con una categoría VÁLIDA según el contrato (Pydantic no
    la rechaza) pero poco frecuente, para confirmar que el pipeline interno
    no se rompe con categorías de baja frecuencia.
    """
    payload = {
        "dias_anticipacion": 10, "semana_llegada": 10, "dia_llegada": 10,
        "noches_fin_semana": 1, "noches_semana": 2, "adultos": 2, "ninos": 0,
        "cancelaciones_previas": 0, "cambios_reserva": 0, "dias_lista_espera": 0,
        "solicitudes_especiales": 1, "tipo_hotel": "Resort Hotel",
        "mes_llegada": "January", "regimen_alimenticio": "Undefined",
        "segmento_mercado": "Aviation", "canal_distribucion": "GDS",
        "tipo_deposito": "Refundable", "tipo_cliente": "Group",
        "tipo_hab_reservada": "P", "tipo_hab_asignada": "K",
    }
    with TestClient(app) as client:
        r = client.post("/predict/tarifa", json=payload)
        assert r.status_code == 200, f"No debería fallar con categorías poco frecuentes: {r.text}"


if __name__ == "__main__":
    print("Ejecutando pruebas de funcionamiento del API...\n")
    test_endpoint_raiz()
    print("✓ Endpoint raíz responde correctamente")
    test_health()
    print("✓ Endpoint /health responde correctamente")
    test_predict_tarifa_contra_preprocesamiento_real()
    print("✓ /predict/tarifa consistente con el preprocesamiento real + modelo final")
    test_predict_cancelacion_contra_preprocesamiento_real()
    print("✓ /predict/cancelacion consistente con el preprocesamiento real + modelo final")
    test_validacion_categoria_invalida()
    print("✓ Validación de categoría inválida (422)")
    test_validacion_campo_faltante()
    print("✓ Validación de campo faltante (422)")
    test_categoria_no_vista_no_falla()
    print("✓ Categorías poco frecuentes no rompen el preprocesamiento")
    print("\nTODAS LAS PRUEBAS PASARON.")
