import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

# --------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------------------
st.set_page_config(page_title="Análisis Aprendizaje Cooperativo", layout="wide")
st.title("📊 Dashboard Interactivo – Aprendizaje Cooperativo")

st.markdown("""
Este panel permite explorar los resultados del cuestionario con filtros dinámicos.
Selecciona curso, establecimiento o rango de fechas para ver cómo cambian los promedios.
""")

# --------------------------------------------------------
# FUENTE DE DATOS (reemplazar por tu Google Sheet si quieres)
# --------------------------------------------------------
# 🔹 Si quieres probar con datos simulados:
np.random.seed(42)
data = {
    "Establecimiento": np.random.choice(["Escuela A", "Escuela B", "Escuela C"], 100),
    "Curso": np.random.choice(["6°A", "7°B", "8°A", "8°B"], 100),
    "Fecha": pd.date_range("2025-10-01", periods=100, freq="D"),
    "P1": np.random.randint(1, 6, 100),
    "P2": np.random.randint(1, 6, 100),
    "P3": np.random.randint(1, 6, 100),
    "P4": np.random.randint(1, 6, 100),
    "P5": np.random.randint(1, 6, 100)
}
df = pd.DataFrame(data)

# 🔹 Si quieres conectar con Google Sheets, descomenta esto y configura tus credenciales:
"""
SHEET_NAME = "Cuestionario Aprendizaje Cooperativo (Ampliado) (respuestas)"
credentials = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
gc = gspread.authorize(credentials)
worksheet = gc.open(SHEET_NAME).sheet1
df = pd.DataFrame(worksheet.get_all_records())
"""

# --------------------------------------------------------
# FILTROS INTERACTIVOS
# --------------------------------------------------------
st.sidebar.header("🔍 Filtros")

establecimientos = ["Todos"] + sorted(df["Establecimiento"].unique().tolist())
cursos = ["Todos"] + sorted(df["Curso"].unique().tolist())
fecha_min, fecha_max = df["Fecha"].min(), df["Fecha"].max()

sel_establecimiento = st.sidebar.selectbox("Establecimiento", establecimientos)
sel_curso = st.sidebar.selectbox("Curso", cursos)
sel_fecha = st.sidebar.slider("Rango de fechas", fecha_min, fecha_max, (fecha_min, fecha_max))

# Aplicar filtros
filtro = (df["Fecha"].between(sel_fecha[0], sel_fecha[1]))
if sel_establecimiento != "Todos":
    filtro &= (df["Establecimiento"] == sel_establecimiento)
if sel_curso != "Todos":
    filtro &= (df["Curso"] == sel_curso)

df_filtrado = df[filtro]

st.markdown(f"**Registros mostrados:** {len(df_filtrado)}")

# --------------------------------------------------------
# ANÁLISIS
# --------------------------------------------------------
cols_preguntas = [c for c in df_filtrado.columns if c.startswith("P")]
promedios = df_filtrado[cols_preguntas].mean().round(2)

# --------------------------------------------------------
# VISUALIZACIÓN 1: Promedio por Pregunta (interactivo)
# --------------------------------------------------------
st.subheader("Promedio por Pregunta")

fig1 = px.bar(
    x=promedios.index,
    y=promedios.values,
    labels={"x": "Pregunta", "y": "Promedio"},
    title="Promedio de Puntaje por Pregunta",
    color=promedios.values,
    color_continuous_scale="Blues",
)
st.plotly_chart(fig1, use_container_width=True)

# --------------------------------------------------------
# VISUALIZACIÓN 2: Evolución temporal
# --------------------------------------------------------
st.subheader("Evolución temporal del promedio general")

df_filtrado["Promedio General"] = df_filtrado[cols_preguntas].mean(axis=1)
prom_tiempo = df_filtrado.groupby("Fecha")["Promedio General"].mean().reset_index()

fig2 = px.line(
    prom_tiempo,
    x="Fecha",
    y="Promedio General",
    title="Promedio general diario",
    markers=True,
)
st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------
# VISUALIZACIÓN 3: Comparación entre cursos
# --------------------------------------------------------
st.subheader("Comparación de Promedio por Curso")

prom_curso = (
    df_filtrado.groupby("Curso")[cols_preguntas]
    .mean()
    .mean(axis=1)
    .reset_index(name="Promedio General")
)

fig3 = px.bar(
    prom_curso,
    x="Curso",
    y="Promedio General",
    color="Promedio General",
    color_continuous_scale="Viridis",
    title="Promedio General por Curso"
)
st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------
# DESCARGA DE DATOS
# --------------------------------------------------------
csv = df_filtrado.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar datos filtrados (CSV)", csv, "datos_filtrados.csv", "text/csv")
