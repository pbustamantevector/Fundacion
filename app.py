import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
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

# Constantes para generación de datos
N_ESTUDIANTES = 200
N_PREGUNTAS_BASE = 20
N_PREGUNTAS_EXTRA = 2
PORCENTAJE_ESTUDIANTES_EXTRA = 0.20

def generar_datos_simulados():
    """Genera datos simulados para el análisis"""
    # Generar IDs de estudiantes
    estudiantes = [f"EST_{str(i+1).zfill(3)}" for i in range(N_ESTUDIANTES)]
    
    # Determinar estudiantes con preguntas extra
    n_estudiantes_extra = int(N_ESTUDIANTES * PORCENTAJE_ESTUDIANTES_EXTRA)
    estudiantes_con_extra = np.random.choice(estudiantes, n_estudiantes_extra, replace=False)
    
    # Función para generar respuestas
    def generar_respuestas(es_post=False):
        base = np.random.normal(3.5, 0.8, N_PREGUNTAS_BASE)
        if es_post:
            mejora = np.random.normal(0.3, 0.2, N_PREGUNTAS_BASE)
            base = base + mejora
        return np.clip(np.round(base, 2), 1, 5)
    
    # Generar datos
    datos = []
    for tipo in ['PRE', 'POST']:
        for estudiante in estudiantes:
            respuestas = generar_respuestas(es_post=(tipo=='POST'))
            row = {
                'estudiante_id': estudiante,
                'tipo_test': tipo,
                'fecha': '2025-09-01' if tipo == 'PRE' else '2025-11-30'
            }
            # Añadir respuestas base
            for i, resp in enumerate(respuestas, 1):
                row[f'P{i}'] = resp
                
            # Añadir respuestas extra si corresponde
            if estudiante in estudiantes_con_extra:
                row['clima'] = round(np.random.normal(4, 0.5), 2)
                row['motivacion'] = round(np.random.normal(4, 0.5), 2)
                
            datos.append(row)
    
    return pd.DataFrame(datos)

# Generar o cargar datos
@st.cache_data
def obtener_datos():
    """Genera datos y calcula estadísticas básicas"""
    df = generar_datos_simulados()
    
    # Calcular promedios por pregunta
    columnas_preguntas = [col for col in df.columns if col.startswith('P')]
    promedios = df.groupby('tipo_test')[columnas_preguntas].mean()
    
    # Calcular significancia
    significancia = {}
    for pregunta in columnas_preguntas:
        pre = df[df['tipo_test'] == 'PRE'][pregunta]
        post = df[df['tipo_test'] == 'POST'][pregunta]
        t_stat, p_value = stats.ttest_rel(pre, post)
        significancia[pregunta] = {
            'p_value': p_value,
            'significativo': p_value < 0.05
        }
    
    # Análisis clima y motivación
    clima_mot = df[df['clima'].notna()].groupby('tipo_test')[['clima', 'motivacion']].mean()
    
    return df, promedios, pd.DataFrame(significancia).T, clima_mot

# Cargar datos
df, promedios, significancia, clima_motivacion = obtener_datos()

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