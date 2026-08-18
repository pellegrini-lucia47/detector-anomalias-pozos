import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Detector de Anomalías en Pozos Petroleros",
    page_icon="🛢️",
    layout="wide"
)

SENSORES = ["P-ANULAR", "P-PDG", "P-TPT", "T-TPT", "P-JUS-CKGL", "P-MON-CKP",
            "T-JUS-CKP", "T-PDG", "P-JUS-BS", "P-MON-CKGL", "P-JUS-CKP", "PT-P", "QBS", "QGL"]

UMBRAL_FISICO = 1e8  # descarta valores de sensor físicamente imposibles (errores de medición)


# ============================================================
# CARGA DEL MODELO (con cache para no recargar en cada interacción)
# ============================================================
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_final.pkl")
    imputer = joblib.load("imputer.pkl")
    columnas_features = joblib.load("columnas_features.pkl")
    return modelo, imputer, columnas_features


def extraer_features(df_serie, sensores, umbral_fisico=UMBRAL_FISICO):
    """Calcula las mismas features estadísticas usadas en el entrenamiento."""
    feats = {}
    if "state" in df_serie.columns:
        df_valid = df_serie[df_serie["state"].notna()]
    else:
        df_valid = df_serie
    n = len(df_valid)
    x = np.arange(n)

    for sensor in sensores:
        if sensor not in df_valid.columns:
            for suf in ["mean", "std", "min", "max", "range", "slope", "R2"]:
                feats[f"{sensor}_{suf}"] = np.nan
            continue

        y = df_valid[sensor].values.astype(float)
        mask = ~pd.isna(y) & (np.abs(y) < umbral_fisico)

        if mask.sum() < 2:
            for suf in ["mean", "std", "min", "max", "range", "slope", "R2"]:
                feats[f"{sensor}_{suf}"] = np.nan
            continue

        y_valid = y[mask]
        x_valid = x[mask]
        media = np.mean(y_valid)
        desvio = np.std(y_valid)

        feats[f"{sensor}_mean"] = media
        feats[f"{sensor}_std"] = desvio
        feats[f"{sensor}_min"] = np.min(y_valid)
        feats[f"{sensor}_max"] = np.max(y_valid)
        feats[f"{sensor}_range"] = np.max(y_valid) - np.min(y_valid)

        if len(x_valid) >= 2 and desvio > 0:
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_valid, y_valid)
            feats[f"{sensor}_slope"] = slope
            feats[f"{sensor}_R2"] = r_value ** 2
        else:
            feats[f"{sensor}_slope"] = 0.0
            feats[f"{sensor}_R2"] = np.nan

    return feats


def preparar_para_prediccion(feats_dict, columnas_features, imputer):
    """Arma un DataFrame de una fila con exactamente las columnas que espera el modelo."""
    fila = pd.DataFrame([feats_dict])

    # Agregar columnas faltantes (que el modelo espera pero no se pudieron calcular) como NaN
    for col in columnas_features:
        if col not in fila.columns:
            fila[col] = np.nan

    fila = fila[columnas_features]  # mismo orden que en el entrenamiento
    fila = fila.replace([np.inf, -np.inf], np.nan)

    fila_imputada = pd.DataFrame(imputer.transform(fila), columns=columnas_features)
    return fila_imputada


# ============================================================
# INTERFAZ
# ============================================================
st.title("🛢️ Detector de Anomalías en Pozos Petroleros")
st.markdown(
    "Subí un archivo con la serie temporal de sensores de un pozo (formato del dataset 3W) "
    "y el modelo va a indicar si corresponde a **operación normal** o a una **anomalía "
    "(inestabilidad de flujo)**."
)

with st.sidebar:
    st.header("ℹ️ Sobre el modelo")
    st.markdown("""
    - Entrenado con datos **reales** del dataset 3W (Petrobras)
    - Clasificación binaria: Normal vs Anomalía
    - Prioriza **no dejar pasar anomalías reales** (alto recall), aunque
      eso implique algunas falsas alarmas a revisar
    - Evaluado con pozos completamente distintos entre entrenamiento y prueba
    """)

archivo = st.file_uploader(
    "Subí el archivo de la instancia (.parquet o .csv)",
    type=["parquet", "csv"]
)

if archivo is not None:
    try:
        if archivo.name.endswith(".parquet"):
            df_serie = pd.read_parquet(archivo)
        else:
            df_serie = pd.read_csv(archivo)

        st.success(f"Archivo cargado: {df_serie.shape[0]} filas, {df_serie.shape[1]} columnas")

        with st.spinner("Calculando features y prediciendo..."):
            modelo, imputer, columnas_features = cargar_modelo()
            feats = extraer_features(df_serie, SENSORES)
            fila_lista = preparar_para_prediccion(feats, columnas_features, imputer)

            prediccion = modelo.predict(fila_lista)[0]
            probabilidades = modelo.predict_proba(fila_lista)[0]
            clases = modelo.classes_
            prob_dict = dict(zip(clases, probabilidades))

        # ============================================================
        # RESULTADO
        # ============================================================
        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            if prediccion == "ANOMALIA":
                st.error("⚠️ ANOMALÍA DETECTADA")
            else:
                st.success("✅ OPERACIÓN NORMAL")

            st.metric("Confianza", f"{max(probabilidades)*100:.1f}%")
            st.write("Probabilidades:")
            for clase, prob in prob_dict.items():
                st.write(f"- {clase}: {prob*100:.1f}%")

        with col2:
            sensores_disponibles = [s for s in SENSORES if s in df_serie.columns]
            if sensores_disponibles:
                sensor_elegido = st.selectbox("Sensor a visualizar", sensores_disponibles)
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(df_serie[sensor_elegido].values, linewidth=0.8)
                ax.set_title(f"Serie temporal — {sensor_elegido}")
                ax.set_xlabel("Muestra")
                ax.set_ylabel(sensor_elegido)
                st.pyplot(fig)

        with st.expander("Ver features calculadas"):
            st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "valor"}))

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        st.info(
            "Verificá que el archivo tenga el formato esperado (columnas de sensores del dataset 3W, "
            "por ejemplo: P-TPT, T-TPT, P-PDG, etc.)"
        )
else:
    st.info("👆 Subí un archivo para empezar.")
