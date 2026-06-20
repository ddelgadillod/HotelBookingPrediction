"""
Entrena y serializa los modelos FINALES de producción:
  - Regresión:     XGBoost, hiperparámetros óptimos de Act20 (Grid Search)
  - Clasificación: XGBoost, hiperparámetros óptimos de Act20 (Grid scoring=AUC) + umbral 0.43

Este script NO regenera el preprocesamiento desde cero: CARGA los objetos
reales exportados directamente desde la sesión de Google Colab donde corrió
Act9_ML_DiegoDelgadillo_HotelBooking_final.ipynb
(artefactos_preprocesamiento/preprocesamiento_real_*.joblib) y los usa para
transformar el split train/test antes de entrenar. Esto garantiza que el
modelo entrenado y el objeto de preprocesamiento que usará el API en
producción sean exactamente coherentes entre sí — mismo scaler, mismo
esquema de columnas, mismo proceso de validación, usando
app.preprocessing.transformar_real() (el mismo código que ejecuta el API
en cada request).

IMPORTANTE: si necesitas regenerar el preprocesamiento (porque cambió el
CSV fuente, o porque Act9 cambió), debes volver a exportar los .joblib
desde Colab (ver celda de exportación añadida a Act9) y reemplazar los
archivos en artefactos_preprocesamiento/ ANTES de correr este script.

Uso:
    python3 reentrenar_y_serializar.py /ruta/a/hotel_bookings.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              roc_auc_score, f1_score, recall_score, cohen_kappa_score)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from app.preprocessing import VARS_INCLUIDAS_NUM, VARS_INCLUIDAS_CAT, cargar_preprocesamiento_real, transformar_real

RANDOM_STATE = 42
RUTA_CSV = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/hotel_bookings.csv'

RENAME_COLS = {
    'hotel': 'tipo_hotel', 'is_canceled': 'cancelado', 'lead_time': 'dias_anticipacion',
    'arrival_date_year': 'anio_llegada', 'arrival_date_month': 'mes_llegada',
    'arrival_date_week_number': 'semana_llegada', 'arrival_date_day_of_month': 'dia_llegada',
    'stays_in_weekend_nights': 'noches_fin_semana', 'stays_in_week_nights': 'noches_semana',
    'adults': 'adultos', 'children': 'ninos', 'babies': 'bebes',
    'meal': 'regimen_alimenticio', 'country': 'pais_origen',
    'market_segment': 'segmento_mercado', 'distribution_channel': 'canal_distribucion',
    'is_repeated_guest': 'cliente_recurrente', 'previous_cancellations': 'cancelaciones_previas',
    'previous_bookings_not_canceled': 'reservas_previas_cumplidas',
    'reserved_room_type': 'tipo_hab_reservada', 'assigned_room_type': 'tipo_hab_asignada',
    'booking_changes': 'cambios_reserva', 'deposit_type': 'tipo_deposito',
    'agent': 'id_agente', 'company': 'id_empresa', 'days_in_waiting_list': 'dias_lista_espera',
    'customer_type': 'tipo_cliente', 'adr': 'tarifa_diaria',
    'required_car_parking_spaces': 'plazas_parking', 'total_of_special_requests': 'solicitudes_especiales',
    'reservation_status': 'estado_reserva', 'reservation_status_date': 'fecha_estado_reserva',
}

df_raw = pd.read_csv(RUTA_CSV).rename(columns=RENAME_COLS)
print(f"df_raw: {df_raw.shape}")

COLS_SELECCIONADAS = VARS_INCLUIDAS_NUM + VARS_INCLUIDAS_CAT + ['tarifa_diaria', 'cancelado']
df_clean = df_raw[COLS_SELECCIONADAS].copy()
n_antes = len(df_clean)
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
print(f"Duplicados eliminados: {n_antes - len(df_clean):,} ({(n_antes-len(df_clean))/n_antes*100:.1f}%)")
print(f"df_clean tras deduplicar: {df_clean.shape}")

# Cargar los objetos de preprocesamiento REALES exportados desde Colab
print("\nCargando preprocesamiento real exportado desde Act9 (Colab)...")
prep_r = cargar_preprocesamiento_real("regresion")
prep_c = cargar_preprocesamiento_real("clasificacion")
print(f"  Regresión:     {prep_r.get('origen')}")
print(f"  Clasificación: {prep_c.get('origen')}")

# ════════════════════════════════════════════════════════════════════════
# RAMA A — REGRESIÓN
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*70); print("RAMA A — REGRESIÓN"); print("="*70)

sss_sub = StratifiedShuffleSplit(n_splits=1, test_size=1 - 25000/len(df_clean), random_state=RANDOM_STATE)
for idx_keep, _ in sss_sub.split(df_clean, df_clean['tipo_hotel']):
    df_r = df_clean.iloc[idx_keep].copy()

df_r.loc[df_r['tarifa_diaria'] < 0, 'tarifa_diaria'] = np.nan
df_r.loc[df_r['tarifa_diaria'] > 800, 'tarifa_diaria'] = np.nan
med_hotel_r = df_r.groupby('tipo_hotel')['tarifa_diaria'].transform('median')
df_r['tarifa_diaria'] = df_r['tarifa_diaria'].fillna(med_hotel_r)

X_r_raw = df_r[VARS_INCLUIDAS_NUM + VARS_INCLUIDAS_CAT].copy()
y_r = df_r['tarifa_diaria'].copy()

X_train_r_raw, X_test_r_raw, y_train_r, y_test_r = train_test_split(
    X_r_raw, y_r, test_size=0.30, random_state=RANDOM_STATE, shuffle=True)
print(f"X_train (crudo): {X_train_r_raw.shape}  |  X_test (crudo): {X_test_r_raw.shape}")

# Transformar train y test con el preprocesamiento REAL.
# Se usa una versión por-lote (no fila por fila) por performance: aplica la
# misma lógica de transformar_real() pero sobre el DataFrame completo de una
# sola vez, ya que aquí no hay restricción de "una sola fila por request"
# como en el API.
def transformar_real_batch(df_crudo: pd.DataFrame, prep: dict) -> pd.DataFrame:
    df = df_crudo.copy()
    df['ninos'] = pd.to_numeric(df['ninos'], errors='coerce').fillna(0).astype(int)
    df.loc[df['adultos'] == 0, 'adultos'] = np.nan
    df['adultos'] = df['adultos'].fillna(1).astype(int)
    df['habitacion_cambiada'] = (df['tipo_hab_reservada'] != df['tipo_hab_asignada']).astype(int)

    cols_ohe = prep['cols_ohe']
    df_sin_descartadas = df.drop(columns=['tipo_hotel', 'mes_llegada'])
    df_enc = pd.get_dummies(df_sin_descartadas, columns=cols_ohe, dtype=int)

    columnas_finales = prep['columnas_X_finales']
    df_alineado = df_enc.reindex(columns=columnas_finales, fill_value=0)

    num_cols = prep['num_cols_escaladas']
    scaler = prep['scaler']
    df_alineado[num_cols] = scaler.transform(df_alineado[num_cols])
    return df_alineado


X_train_r = transformar_real_batch(X_train_r_raw, prep_r).reset_index(drop=True)
X_test_r = transformar_real_batch(X_test_r_raw, prep_r).reset_index(drop=True)
print(f"X_train (transformado): {X_train_r.shape}  |  X_test (transformado): {X_test_r.shape}")

HP_REG_XGB = {
    'n_estimators': 429, 'max_depth': 9, 'learning_rate': 0.05758774553506099,
    'subsample': 0.8524554503989051, 'colsample_bytree': 0.8545330472743582,
    'reg_alpha': 1.965488623333802, 'reg_lambda': 4.460232775885567,
    'tree_method': 'hist', 'random_state': RANDOM_STATE, 'verbosity': 0,
}
modelo_reg_final = xgb.XGBRegressor(**HP_REG_XGB)
modelo_reg_final.fit(X_train_r, y_train_r.reset_index(drop=True))

pred_r = modelo_reg_final.predict(X_test_r)
r2_final = r2_score(y_test_r, pred_r)
mae_final = mean_absolute_error(y_test_r, pred_r)
rmse_final = np.sqrt(mean_squared_error(y_test_r, pred_r))
print(f"R² test: {r2_final:.4f}  MAE: {mae_final:.2f}  RMSE: {rmse_final:.2f}")

# ════════════════════════════════════════════════════════════════════════
# RAMA B — CLASIFICACIÓN
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*70); print("RAMA B — CLASIFICACIÓN"); print("="*70)

sss_sub_c = StratifiedShuffleSplit(n_splits=1, test_size=1 - 25000/len(df_clean), random_state=RANDOM_STATE)
for idx_keep_c, _ in sss_sub_c.split(df_clean, df_clean['tipo_hotel']):
    df_c = df_clean.iloc[idx_keep_c].copy()

df_c.loc[df_c['tarifa_diaria'] < 0, 'tarifa_diaria'] = np.nan
df_c.loc[df_c['tarifa_diaria'] > 800, 'tarifa_diaria'] = np.nan
mediana_por_hotel = df_c.groupby('tipo_hotel')['tarifa_diaria'].transform('median')
df_c['tarifa_diaria'] = df_c['tarifa_diaria'].fillna(mediana_por_hotel)

X_c_raw = df_c[VARS_INCLUIDAS_NUM + VARS_INCLUIDAS_CAT].copy()
y_c = df_c['cancelado'].copy()

X_train_c_raw, X_test_c_raw, y_train_c, y_test_c = train_test_split(
    X_c_raw, y_c, test_size=0.30, random_state=RANDOM_STATE, stratify=y_c)
print(f"X_train (crudo): {X_train_c_raw.shape}  |  X_test (crudo): {X_test_c_raw.shape}")

X_train_c = transformar_real_batch(X_train_c_raw, prep_c).reset_index(drop=True)
X_test_c = transformar_real_batch(X_test_c_raw, prep_c).reset_index(drop=True)
print(f"X_train (transformado): {X_train_c.shape}  |  X_test (transformado): {X_test_c.shape}")

smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_bal_c, y_bal_c = smote.fit_resample(X_train_c, y_train_c.reset_index(drop=True))
print(f"X_bal_c (post-SMOTE): {X_bal_c.shape}")

HP_CLF_XGB = {
    'n_estimators': 339, 'max_depth': 10, 'learning_rate': 0.05750312535105989,
    'subsample': 0.8247466768575095, 'colsample_bytree': 0.6510758917689862,
    'eval_metric': 'logloss', 'tree_method': 'hist',
    'random_state': RANDOM_STATE, 'verbosity': 0,
}
UMBRAL_DECISION = 0.43

modelo_clf_final = xgb.XGBClassifier(**HP_CLF_XGB)
modelo_clf_final.fit(X_bal_c, y_bal_c)

proba_c = modelo_clf_final.predict_proba(X_test_c)[:, 1]
pred_c = (proba_c >= UMBRAL_DECISION).astype(int)

f1_final = f1_score(y_test_c, pred_c)
auc_final = roc_auc_score(y_test_c, proba_c)
recall_final = recall_score(y_test_c, pred_c)
kappa_final = cohen_kappa_score(y_test_c, pred_c)
print(f"F1 test: {f1_final:.4f}  AUC: {auc_final:.4f}  Recall: {recall_final:.4f}  Kappa: {kappa_final:.4f}")

# ════════════════════════════════════════════════════════════════════════
# SERIALIZACIÓN — solo los modelos (el preprocesamiento ya está serializado
# y NO se regenera: es el real de Colab, importado al inicio de este script)
# ════════════════════════════════════════════════════════════════════════
TS = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs('model', exist_ok=True)

path_reg = f'model/modelo_regresion_tarifa_xgboost_{TS}.joblib'
path_clf = f'model/modelo_clasificacion_cancelacion_xgboost_{TS}.joblib'

joblib.dump({
    'modelo': modelo_reg_final, 'algoritmo': 'XGBoost',
    'hiperparametros': HP_REG_XGB,
    'metrica': 'R2', 'valor_metrica': round(r2_final, 4),
    'mae': round(mae_final, 2), 'rmse': round(rmse_final, 2),
    'origen': 'Act20_ML_DiegoDelgadillo_HotelBooking_final.ipynb (Grid Search), '
              'entrenado sobre preprocesamiento REAL exportado desde Colab (Act9)',
}, path_reg)

joblib.dump({
    'modelo': modelo_clf_final, 'algoritmo': 'XGBoost',
    'hiperparametros': HP_CLF_XGB, 'umbral_decision': UMBRAL_DECISION,
    'metrica': 'F1', 'valor_metrica': round(f1_final, 4),
    'auc': round(auc_final, 4), 'recall': round(recall_final, 4), 'kappa': round(kappa_final, 4),
    'origen': 'Act20_ML_DiegoDelgadillo_HotelBooking_final.ipynb (Grid Search, scoring=AUC), '
              'entrenado sobre preprocesamiento REAL exportado desde Colab (Act9)',
    'justificacion_seleccion': (
        'Se priorizó frente a Random Forest (F1=0.6142, Recall=0.6889) por restricción de '
        'infraestructura: el modelo con Random Forest serializado pesa ~82 MB, arriesgando '
        'el límite de RAM (512 MB) del free tier de Render.'
    ),
}, path_clf)

print(f"\nGuardado: {path_reg}")
print(f"Guardado: {path_clf}")
print("(El preprocesamiento real NO se regeneró — sigue siendo el exportado desde Colab)")

# Limpiar modelos anteriores (entrenados sobre una versión distinta del
# preprocesamiento, ya no coherentes con los artefactos actuales)
import glob
for f in glob.glob('model/*.joblib'):
    if f not in (path_reg, path_clf):
        os.remove(f)
        print(f"Eliminado modelo anterior incompatible: {f}")

print("\nListo — modelos finales reentrenados sobre el preprocesamiento real de Colab.")
