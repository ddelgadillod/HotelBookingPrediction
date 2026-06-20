"""
Preprocesamiento REAL — replica exacta de Act9_ML_DiegoDelgadillo_HotelBooking_final.ipynb
(celdas 111-120 para Rama A / Regresión, celdas 138-142 para Rama B / Clasificación).

A diferencia de una versión anterior de este módulo (que usaba un Pipeline de
sklearn con OneHotEncoder + RobustScaler ajustados por este proyecto), esta
versión usa los objetos de preprocesamiento REALES de Act9 — el mismo
RobustScaler ya fit-eado y el mismo esquema de columnas de pd.get_dummies que
produjeron los datasets sobre los que se buscaron los hiperparámetros óptimos
en Act20. Esto garantiza trazabilidad exacta entre Taller 1 -> Taller 2/3 -> API.

IMPORTANTE — diferencias de diseño respecto a la versión anterior:
1. El RobustScaler de Act9 se ajustó sobre TODO el dataset antes del split
   train/test (leakage menor de escala). Se mantiene así deliberadamente
   para honrar la línea de las entregas académicas previas.
2. pd.get_dummies no tiene un equivalente nativo a OneHotEncoder(handle_unknown=
   'ignore'); aquí se replica ese comportamiento manualmente con DataFrame.reindex:
   una categoría no vista en entrenamiento simplemente no activa ninguna
   dummie de esa variable (todas en 0), en vez de lanzar un error.
3. 'tipo_hotel' y 'mes_llegada' se reciben en el payload pero se descartan
   antes de pd.get_dummies (igual que en Act9): nunca fueron predictoras.

Los objetos reales (scaler, esquema de columnas) se cargan desde
artefactos_preprocesamiento/preprocesamiento_real_*.joblib, generados al
ejecutar la celda de exportación añadida a Act9 en Google Colab.
"""
import os
import joblib
import numpy as np
import pandas as pd

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

PREP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artefactos_preprocesamiento")


def cargar_preprocesamiento_real(rama: str) -> dict:
    """
    rama: 'regresion' o 'clasificacion'.
    Devuelve el dict serializado desde Act9 con: scaler, num_cols_escaladas,
    columnas_X_finales (orden exacto post get_dummies), cols_ohe, origen.
    """
    nombre_archivo = f"preprocesamiento_real_{rama}.joblib"
    ruta = os.path.join(PREP_DIR, nombre_archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró '{ruta}'. Genera este archivo ejecutando la celda de "
            f"exportación de preprocesamiento en Act9 (Google Colab) y cópialo a "
            f"{PREP_DIR}/."
        )
    return joblib.load(ruta)


def transformar_real(payload: dict, prep: dict) -> pd.DataFrame:
    """
    Aplica el preprocesamiento real de Act9 a un payload crudo (1 fila):
      1. Imputación (ninos -> 0, adultos 0 -> 1)
      2. Feature derivada 'habitacion_cambiada'
      3. One-Hot Encoding manual (pd.get_dummies), alineado al esquema fijo de
         columnas capturado en entrenamiento — categorías no vistas quedan con
         todas sus dummies en 0 (equivalente a handle_unknown='ignore')
      4. RobustScaler real (fit en Act9) sobre las columnas numéricas
    Devuelve un DataFrame de 1 fila con las 58 columnas en el orden exacto
    que espera el modelo.
    """
    df = pd.DataFrame([payload])

    df['ninos'] = pd.to_numeric(df['ninos'], errors='coerce').fillna(0).astype(int)
    df.loc[df['adultos'] == 0, 'adultos'] = np.nan
    df['adultos'] = df['adultos'].fillna(1).astype(int)
    df['habitacion_cambiada'] = (
        df['tipo_hab_reservada'] != df['tipo_hab_asignada']
    ).astype(int)

    cols_ohe = prep['cols_ohe']
    df_sin_descartadas = df.drop(columns=['tipo_hotel', 'mes_llegada'])
    df_enc = pd.get_dummies(df_sin_descartadas, columns=cols_ohe, dtype=int)

    # Alinear al esquema fijo de Act9: agrega columnas faltantes (categoría no
    # vista) en 0, descarta columnas fuera de esquema, fija el ORDEN exacto.
    columnas_finales = prep['columnas_X_finales']
    df_alineado = df_enc.reindex(columns=columnas_finales, fill_value=0)

    num_cols = prep['num_cols_escaladas']
    scaler = prep['scaler']
    df_alineado[num_cols] = scaler.transform(df_alineado[num_cols])

    return df_alineado
