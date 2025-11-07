import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# CONFIGURACIÓN DE LA APP
# -----------------------------
st.set_page_config(page_title="Ejemplo básico de gráfico", layout="centered")
st.title("📊 Ejemplo simple – Promedio por pregunta (datos simulados)")

# -----------------------------
# DATOS SIMULADOS
# -----------------------------
# 10 preguntas y 30 respuestas aleatorias
np.random.seed(1)
n_respuestas = 30
data = {
    f"P{i+1}": np.random.randint(1, 6, n_respuestas) for i in range(10)
}
df = pd.DataFrame(data)

# Calcular promedio por pregunta
promedios = df.mean().round(2).reset_index()
promedios.columns = ["Pregunta", "Promedio"]

# -----------------------------
# GRÁFICO INTERACTIVO
# -----------------------------
st.subheader("Promedio por pregunta")

fig = px.bar(
    promedios,
    x="Pregunta",
    y="Promedio",
    color="Promedio",
    color_continuous_scale="Blues",
    title="Promedio de puntaje por pregunta",
)
fig.update_layout(yaxis_range=[0, 5])
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TABLA DE DATOS
# -----------------------------
st.subheader("Datos simulados (primeras filas)")
st.dataframe(df.head())

st.markdown("✅ Esta es una versión básica sin conexión ni filtros.")
