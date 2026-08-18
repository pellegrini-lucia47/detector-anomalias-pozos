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
    page_title="ECOPOZO — Monitoreo Inteligente de Pozos",
    page_icon="🛢️",
    layout="wide"
)

SENSORES = ["P-ANULAR", "P-PDG", "P-TPT", "T-TPT", "P-JUS-CKGL", "P-MON-CKP",
            "T-JUS-CKP", "T-PDG", "P-JUS-BS", "P-MON-CKGL", "P-JUS-CKP", "PT-P", "QBS", "QGL"]

UMBRAL_FISICO = 1e8  # descarta valores de sensor físicamente imposibles (errores de medición)

# Umbrales del ECO SCORE, calibrados con datos reales de test (ver análisis en Colab)
UMBRAL_VERDE = 30
UMBRAL_AMARILLO = 65

# Umbrales de riesgo ambiental (reglas simples, basadas en variación de sensores clave)
UMBRAL_CAIDA_PRESION_ANULAR = -50   # slope muy negativo -> posible fuga
UMBRAL_RANGO_PRESION_ALTO = 2000    # oscilación anómala de presión


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

    for col in columnas_features:
        if col not in fila.columns:
            fila[col] = np.nan

    fila = fila[columnas_features]
    fila = fila.replace([np.inf, -np.inf], np.nan)

    fila_imputada = pd.DataFrame(imputer.transform(fila), columns=columnas_features)
    return fila_imputada


def calcular_eco_score(modelo, fila_imputada):
    """Devuelve el ECO SCORE (0-100): probabilidad de anomalía según el modelo."""
    probs = modelo.predict_proba(fila_imputada)[0]
    idx_anomalia = list(modelo.classes_).index("ANOMALIA")
    return probs[idx_anomalia] * 100


def clasificar_semaforo(eco_score, umbral_verde=UMBRAL_VERDE, umbral_amarillo=UMBRAL_AMARILLO):
    if eco_score < umbral_verde:
        return "🟢 ESTABLE", "normal", "El pozo opera dentro de parámetros normales."
    elif eco_score < umbral_amarillo:
        return "🟡 ATENCIÓN", "off", "Se detectan variaciones que conviene monitorear de cerca."
    else:
        return "🔴 ANOMALÍA", "inverse", "Comportamiento anómalo detectado. Se recomienda revisión."


def evaluar_riesgo_ambiental(feats):
    """Reglas simples y explicables para señalar comportamientos que podrían
    justificar una inspección ambiental (no reemplazan una inspección real)."""
    alertas = []

    slope_anular = feats.get("P-ANULAR_slope", 0)
    if slope_anular is not None and not pd.isna(slope_anular) and slope_anular < UMBRAL_CAIDA_PRESION_ANULAR:
        alertas.append("⚠️ Caída abrupta de presión anular — posible indicio de fuga en el sistema de contención.")

    rango_pdg = feats.get("P-PDG_range", 0)
    if rango_pdg is not None and not pd.isna(rango_pdg) and rango_pdg > UMBRAL_RANGO_PRESION_ALTO:
        alertas.append("⚠️ Oscilación de presión de fondo (PDG) por fuera de lo esperado — revisar integridad del pozo.")

    rango_tpt = feats.get("P-TPT_range", 0)
    if rango_tpt is not None and not pd.isna(rango_tpt) and rango_tpt > UMBRAL_RANGO_PRESION_ALTO:
        alertas.append("⚠️ Variación fuerte de presión en tubing (TPT) — posible inestabilidad de flujo con riesgo asociado.")

    return alertas


# ============================================================
# INTERFAZ
# ============================================================
st.title("🛢️🌱 ECOPOZO")
st.subheader("Monitoreo inteligente para detectar anomalías operativas y prevenir riesgos ambientales")

st.markdown(
    "Detectar manualmente cuándo un pozo comienza a comportarse de forma anormal puede ser difícil. "
    "ECOPOZO analiza automáticamente los datos del pozo y genera un diagnóstico simple y accionable, "
    "junto con una señal preventiva sobre posibles riesgos ambientales."
)

with st.sidebar:
    st.header("ℹ️ Sobre ECOPOZO")
    st.markdown("""
    **Objetivo del sistema:**
    Ayudar a detectar tempranamente cambios anormales en el
    comportamiento de un pozo — no perseguir un accuracy perfecto,
    sino dar una alerta temprana y explicable.

    **Semáforo operativo:**
    - 🟢 **Estable** — operación normal
    - 🟡 **Atención** — variaciones a monitorear
    - 🔴 **Anomalía** — revisión recomendada

    **Metodología:**
    - Modelo entrenado con datos **reales** del dataset 3W (Petrobras)
    - Umbrales del ECO SCORE calibrados con datos de test reales
    - Evaluado con pozos completamente distintos entre entrenamiento y prueba
    - Componente ambiental basado en reglas explicables sobre sensores clave
    """)

archivo = st.file_uploader(
    "Subí el archivo de la instancia a analizar (.parquet o .csv)",
    type=["parquet", "csv"]
)

if archivo is not None:
    try:
        if archivo.name.endswith(".parquet"):
            df_serie = pd.read_parquet(archivo)
        else:
            df_serie = pd.read_csv(archivo)

        st.success(f"Archivo cargado: {df_serie.shape[0]} filas, {df_serie.shape[1]} columnas")

        with st.spinner("Calculando ECO SCORE..."):
            modelo, imputer, columnas_features = cargar_modelo()
            feats = extraer_features(df_serie, SENSORES)
            fila_lista = preparar_para_prediccion(feats, columnas_features, imputer)

            eco_score = calcular_eco_score(modelo, fila_lista)
            etiqueta, delta_color, explicacion = clasificar_semaforo(eco_score)
            alertas_ambientales = evaluar_riesgo_ambiental(feats)

        st.divider()

        # ============================================================
        # RESULTADO PRINCIPAL
        # ============================================================
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"## {etiqueta}")
            st.caption(explicacion)
            st.metric("ECO SCORE", f"{eco_score:.1f} / 100")
            st.progress(min(int(eco_score), 100) / 100)

            st.markdown("#### 🌱 Componente ambiental")
            if alertas_ambientales:
                for alerta in alertas_ambientales:
                    st.warning(alerta)
            else:
                st.info("No se detectaron señales que ameriten inspección ambiental preventiva.")

        with col2:
            sensores_disponibles = [s for s in SENSORES if s in df_serie.columns]
            if sensores_disponibles:
                sensor_elegido = st.selectbox("Sensor a visualizar", sensores_disponibles)
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(df_serie[sensor_elegido].values, linewidth=0.8, color="#2E86AB")
                ax.set_title(f"Serie temporal — {sensor_elegido}")
                ax.set_xlabel("Muestra")
                ax.set_ylabel(sensor_elegido)
                st.pyplot(fig)

        with st.expander("Ver features calculadas (detalle técnico)"):
            st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "valor"}))

        with st.expander("¿Cómo se calcula el ECO SCORE?"):
            st.markdown(f"""
            El ECO SCORE es la probabilidad (0 a 100) de que el comportamiento del pozo
            corresponda a una anomalía, según un modelo de Random Forest entrenado con
            instancias **reales** del dataset 3W (Petrobras).

            - **🟢 Estable:** ECO SCORE < {UMBRAL_VERDE}
            - **🟡 Atención:** {UMBRAL_VERDE} ≤ ECO SCORE < {UMBRAL_AMARILLO}
            - **🔴 Anomalía:** ECO SCORE ≥ {UMBRAL_AMARILLO}

            Estos umbrales fueron calibrados analizando cómo se distribuye el ECO SCORE
            en instancias reales ya etiquetadas, priorizando que **ninguna anomalía real
            quede clasificada como estable** (aunque eso implique algunas falsas alarmas
            a revisar manualmente).
            """)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        st.info(
            "Verificá que el archivo tenga el formato esperado (columnas de sensores del dataset 3W, "
            "por ejemplo: P-TPT, T-TPT, P-PDG, etc.)"
        )
else:
    st.info("👆 Subí un archivo para comenzar el análisis.")