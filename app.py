import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# DATOS DE EJEMPLO
# -------------------------------
# Supongamos que tenemos 10 preguntas del cuestionario
preguntas = [f"P{i}" for i in range(1, 11)]
promedios = np.random.uniform(3.0, 5.0, size=10).round(2)  # Promedios entre 3 y 5
df = pd.DataFrame({"Pregunta": preguntas, "Promedio": promedios})

# -------------------------------
# CONFIG STREAMLIT
# -------------------------------
st.set_page_config(page_title="Ejemplo de Gráficos", layout="centered")
st.title("🎨 Ejemplo de Visualizaciones con Streamlit + Matplotlib")

st.markdown("Datos simulados del cuestionario de aprendizaje cooperativo:")

st.dataframe(df, use_container_width=True)

# -------------------------------
# GRÁFICO DE BARRAS
# -------------------------------
st.subheader("📊 Gráfico de barras")
fig1, ax1 = plt.subplots()
ax1.bar(df["Pregunta"], df["Promedio"], color="#4B9CD3")
ax1.set_ylim(0, 5)
ax1.set_ylabel("Promedio")
ax1.set_title("Promedio por Pregunta (Barras)")
st.pyplot(fig1)

# -------------------------------
# GRÁFICO DE LÍNEA
# -------------------------------
st.subheader("📈 Gráfico de línea")
fig2, ax2 = plt.subplots()
ax2.plot(df["Pregunta"], df["Promedio"], marker="o", color="#E76F51")
ax2.set_ylim(0, 5)
ax2.set_ylabel("Promedio")
ax2.set_title("Tendencia de Promedios (Línea)")
st.pyplot(fig2)

# -------------------------------
# GRÁFICO RADAR
# -------------------------------
st.subheader("🌐 Gráfico radar (espacial)")
# Convertimos los datos a coordenadas circulares
N = len(df)
valores = df["Promedio"].tolist()
valores += valores[:1]  # cerrar el círculo
angulos = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angulos += angulos[:1]

fig3, ax3 = plt.subplots(subplot_kw={'projection': 'polar'})
ax3.plot(angulos, valores, linewidth=2, linestyle='solid', color="#2A9D8F")
ax3.fill(angulos, valores, color="#2A9D8F", alpha=0.4)
ax3.set_thetagrids(np.degrees(angulos[:-1]), df["Pregunta"])
ax3.set_ylim(0, 5)
ax3.set_title("Promedio por Pregunta (Radar)")
st.pyplot(fig3)
