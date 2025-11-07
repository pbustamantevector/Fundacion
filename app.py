import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt

# --------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------------------
st.set_page_config(page_title="Análisis Aprendizaje Cooperativo", layout="wide")
st.title("📊 Análisis del Cuestionario de Aprendizaje Cooperativo")
st.markdown(
    "Visualización automática de resultados del formulario. "
    "Los datos provienen directamente de Google Sheets y se actualizan en tiempo real."
)

# --------------------------------------------------------
# CONFIGURACIÓN DE ACCESO A GOOGLE SHEETS
# --------------------------------------------------------
# 1️⃣  Crea una credencial de servicio en Google Cloud (JSON) y guárdala en el mismo repo.
# 2️⃣  En Streamlit Cloud: Settings → Secrets → agrega el contenido del JSON en 'gcp_service_account'
#      Ejemplo:  st.secrets["gcp_service_account"]

SHEET_NAME = "Cuestionario Aprendizaje Cooperativo (Ampliado) (respuestas)"

try:
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )

    gc = gspread.authorize(credentials)
    spreadsheet = gc.open(SHEET_NAME)
    worksheet = spreadsheet.sheet1
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

except Exception as e:
    st.error(f"❌ No se pudo conectar con Google Sheets.\n\n**Detalle técnico:** {e}")
    st.stop()

# --------------------------------------------------------
# LIMPIEZA Y ANÁLISIS
# --------------------------------------------------------
if df.empty:
    st.warning("No hay datos disponibles en el Google Sheet aún.")
    st.stop()

# Eliminar filas completamente vacías
df = df.dropna(how="all")

# Seleccionar solo columnas numéricas (las respuestas del cuestionario)
df_num = df.select_dtypes(include="number")

# Calcular promedios
promedios = df_num.mean().round(2)
promedio_total = round(promedios.mean(), 2)

# --------------------------------------------------------
# VISUALIZACIÓN
# --------------------------------------------------------
st.subheader("Promedio por Pregunta")

fig, ax = plt.subplots(figsize=(10, 4))
promedios.plot(kind="bar", ax=ax, color="#4B9CD3")
ax.set_ylabel("Promedio")
ax.set_xlabel("Pregunta")
ax.set_title("Promedio de Puntaje por Pregunta")
st.pyplot(fig)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tabla de promedios")
    st.dataframe(promedios.to_frame("Promedio"))

with col2:
    st.subheader("Promedio general del cuestionario")
    st.metric(label="Promedio total", value=promedio_total)

# --------------------------------------------------------
# DESCARGA OPCIONAL
# --------------------------------------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar datos (CSV)", csv, "respuestas_cuestionario.csv", "text/csv")
