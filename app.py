import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------
st.set_page_config(page_title="Dashboard Aprendizaje Cooperativo", layout="wide")
st.title("📊 Dashboard Interactivo – Aprendizaje Cooperativo (Datos Simulados)")

st.markdown("""
Este panel muestra un ejemplo de análisis y visualización de resultados del cuestionario de aprendizaje cooperativo,
usando **datos simulados**.  
Luego, se puede conectar directamente al formulario de Google Sheets sin cambiar la lógica de análisis.
""")

# --------------------------------------------------------
# GENERAR DATOS DE EJEMPLO
# --------------------------------------------------------
np.random.seed(42)

n = 200  # cantidad de respuestas simuladas
data = {
    "Establecimiento": np.random.choice(["Escuela A", "Escuela B", "Escuela C"], n),
    "Curso": np.random.choice(["6°A", "7°A", "8°A", "8°B"], n),
    "Fecha": pd.date_range("2025-09-01", periods=n, freq="D"),
    "P1": np.random.randint(1, 6, n),
    "P2": np.random.randint(1, 6, n),
    "P3": np.random.randint(1, 6, n),
    "P4": np.random.randint(1, 6, n),
    "P5": np.random.randint(1, 6, n),
}
df = pd.DataFrame(data)

# --------------------------------------------------------
# FILTROS INTERACTIVOS
# --------------------------------------------------------
st.sidebar.header("🔍 Filtros")

establecimientos = ["Todos"] + sorted(df["Establecimiento"].unique().tolist())
cursos = ["Todos"] + sorted(df["Curso"].unique().tolist())
fecha_min, fecha_max = df["Fecha"].min(), df["Fecha"].max()

sel_establecimiento = st.sidebar.selectbox("Establecimiento", establecimientos)
sel_curso = st.sidebar.selectbox("Curso", cursos)
sel_fecha = st.sidebar.slider(
    "Rango de fechas", fecha_min, fecha_max, (fecha_min, fecha_max)
)

# aplicar filtros
filtro = df["Fecha"].between(sel_fecha[0], sel_fecha[1])
if sel_establecimiento != "Todos":
    filtro &= df["Establecimiento"] == sel_establecimiento
if sel_curso != "Todos":
    filtro &= df["Curso"] == sel_curso

df_filtrado = df[filtro]

st.markdown(f"**Respuestas mostradas:** {len(df_filtrado)}")

# --------------------------------------------------------
# CÁLCULO DE PROMEDIOS
# --------------------------------------------------------
cols_preguntas = [c for c in df.columns if c.startswith("P")]
promedios = df_filtrado[cols_preguntas].mean().round(2)
promedio_total = round(promedios.mean(), 2)

# --------------------------------------------------------
# GRÁFICO 1 – Promedio por pregunta (barras interactivas)
# --------------------------------------------------------
st.subheader("Promedio por pregunta")

fig1 = px.bar(
    x=promedios.index,
    y=promedios.values,
    title="Promedio por Pregunta",
    color=promedios.values,
    labels={"x": "Pregunta", "y": "Promedio"},
    color_continuous_scale="Blues",
)
fig1.update_layout(yaxis_range=[0, 5])
st.plotly_chart(fig1, use_container_width=True)

# --------------------------------------------------------
# GRÁFICO 2 – Evolución temporal
# --------------------------------------------------------
st.subheader("Evolución temporal del promedio general")

df_filtrado["Promedio General"] = df_filtrado[cols_preguntas].mean(axis=1)
prom_tiempo = (
    df_filtrado.groupby("Fecha")["Promedio General"].mean().reset_index()
)

fig2 = px.line(
    prom_tiempo,
    x="Fecha",
    y="Promedio General",
    title="Promedio General Diario",
    markers=True,
)
fig2.update_layout(yaxis_range=[0, 5])
st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------
# GRÁFICO 3 – Comparación entre cursos
# --------------------------------------------------------
st.subheader("Comparación de Promedio General por Curso")

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
    title="Promedio General por Curso",
    color_continuous_scale="Viridis",
)
fig3.update_layout(yaxis_range=[0, 5])
st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------
# TABLA Y DESCARGA
# --------------------------------------------------------
st.subheader("📋 Datos filtrados")
st.dataframe(df_filtrado.head(20), use_container_width=True)

csv = df_filtrado.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar datos filtrados (CSV)",
    csv,
    "datos_filtrados.csv",
    "text/csv",
)
