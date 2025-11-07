import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Metodología Cooperativa",
    page_icon="📊",
    layout="wide"
)

# Título
st.title("📊 Dashboard - Análisis de Metodología Cooperativa")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/respuestas_encuesta.csv")
    promedios = pd.read_csv("data/resultados_promedios.csv", index_col=0)
    significancia = pd.read_csv("data/resultados_significancia.csv", index_col=0)
    clima_motivacion = pd.read_csv("data/resultados_clima_motivacion.csv", index_col=0)
    return df, promedios, significancia, clima_motivacion

df, promedios, significancia, clima_motivacion = cargar_datos()

# Sidebar
st.sidebar.header("Filtros")
tipo_analisis = st.sidebar.selectbox(
    "Seleccione tipo de análisis",
    ["General", "Por Pregunta", "Clima y Motivación"]
)

# Análisis General
if tipo_analisis == "General":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Promedio General por Tipo de Test")
        columnas_preguntas = [col for col in df.columns if col.startswith('P')]
        promedio_general = df.groupby('tipo_test')[columnas_preguntas].mean().mean(axis=1)
        fig = px.bar(
            x=promedio_general.index,
            y=promedio_general.values,
            labels={'x': 'Tipo de Test', 'y': 'Promedio General'},
            color=promedio_general.index
        )
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Mejora por Pregunta")
        diferencias = promedios.loc['POST'] - promedios.loc['PRE']
        fig = px.bar(
            x=diferencias.index,
            y=diferencias.values,
            labels={'x': 'Pregunta', 'y': 'Diferencia (POST - PRE)'}
        )
        st.plotly_chart(fig)

# Análisis por Pregunta
elif tipo_analisis == "Por Pregunta":
    st.subheader("Análisis Detallado por Pregunta")
    
    pregunta = st.selectbox(
        "Seleccione una pregunta",
        [col for col in df.columns if col.startswith('P')]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Distribución de Respuestas - {pregunta}")
        fig = px.box(
            df,
            x='tipo_test',
            y=pregunta,
            points="all"
        )
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Estadísticas")
        stats_data = {
            'PRE': df[df['tipo_test'] == 'PRE'][pregunta].describe(),
            'POST': df[df['tipo_test'] == 'POST'][pregunta].describe()
        }
        st.dataframe(pd.DataFrame(stats_data))
        
        sig = significancia.loc[pregunta]
        st.write("### Prueba de Significancia")
        st.write(f"p-valor: {sig['p_value']:.4f}")
        st.write(f"¿Diferencia significativa?: {'Sí' if sig['significativo'] else 'No'}")

# Análisis de Clima y Motivación
else:
    st.subheader("Análisis de Clima y Motivación")
    
    # Crear gráfico combinado
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Clima', 'Motivación'))
    
    # Añadir datos de clima
    fig.add_trace(
        go.Bar(x=clima_motivacion.columns, y=clima_motivacion.loc['clima'],
               name='Clima'),
        row=1, col=1
    )
    
    # Añadir datos de motivación
    fig.add_trace(
        go.Bar(x=clima_motivacion.columns, y=clima_motivacion.loc['motivacion'],
               name='Motivación'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar estadísticas
    estudiantes_extra = df[df['clima'].notna()]
    n_estudiantes = len(estudiantes_extra['estudiante_id'].unique())
    
    st.write(f"### Estadísticas")
    st.write(f"Número de estudiantes con preguntas adicionales: {n_estudiantes}")
    st.write(f"Porcentaje del total: {(n_estudiantes/len(df['estudiante_id'].unique())*100):.1f}%")
    
    st.write("### Datos Detallados")
    st.dataframe(clima_motivacion)