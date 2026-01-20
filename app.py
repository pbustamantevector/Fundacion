import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis Temporal - Metodología Aprendizaje Cooperativo",
    page_icon="📈",
    layout="wide"
)

# Estilo personalizado
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .main .block-container {
        background-color: black;
        padding: 2rem;
        color: white;
        max-width: 1400px;
    }
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo
try:
    st.image('LOGO_entrepares.png', width=200)
except:
    pass

# Estilo de gráficos
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Paleta de colores personalizada
COLOR_PRE = '#3498db'      # Azul suave
COLOR_POST = '#e74c3c'     # Rojo coral
COLOR_MEJORA = '#27ae60'   # Verde esmeralda
COLOR_DISMINUYE = '#e67e22' # Naranja
COLOR_NEUTRAL = '#95a5a6'  # Gris

# Título principal
st.title("📈 Análisis Temporal de Metodología de Aprendizaje Cooperativo")

# Botón para actualizar datos
col_titulo, col_boton = st.columns([4, 1])
with col_boton:
    if st.button("🔄 Actualizar Datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("""
Este análisis evalúa la evolución temporal de la metodología basada en 7 dimensiones:
- **Habilidades sociales** (Preguntas 1, 6, 11, 16)
- **Procesamiento grupal** (Preguntas 2, 7, 12, 17)
- **Interdependencia positiva** (Preguntas 3, 8, 13, 18)
- **Interacción promotora** (Preguntas 4, 9, 14, 19)
- **Responsabilidad Individual** (Preguntas 5, 10, 15, 20)
- **Clima** (Pregunta 21)
- **Motivación** (Pregunta 22)
""")

# Cargar datos
@st.cache_data
def cargar_datos():
    # Leer datos desde Google Sheets
    sheet_id = '12JoLMA_A_-MuLqxbTTEsmBPVBhNSNbllyAlaThO2HDc'
    gid = '788863411'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
    
    df = pd.read_csv(url)
    
    dimensiones = {
        'Habilidades sociales': [1, 6, 11, 16],
        'Procesamiento grupal': [2, 7, 12, 17],
        'Interdependencia positiva': [3, 8, 13, 18],
        'Interacción promotora': [4, 9, 14, 19],
        'Responsabilidad Individual': [5, 10, 15, 20],
        'Clima': [21],
        'Motivación': [22]
    }
    
    df.columns = df.columns.str.strip()
    
    # Extraer mes/año de la marca temporal
    df['Fecha'] = pd.to_datetime(df['Marca temporal'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['Mes_Año'] = df['Fecha'].dt.to_period('M')
    df['Mes'] = df['Fecha'].dt.month
    df['Año'] = df['Fecha'].dt.year
    
    # Crear claves compuestas
    df['Colegio_Nivel_Paralelo'] = df['1.  Selecciona tu colegio:'] + '_' + df['2. ¿En qué nivel estás?'] + '_' + df['3. Paralelo']
    df['Colegio_Nivel'] = df['1.  Selecciona tu colegio:'] + '_' + df['2. ¿En qué nivel estás?']
    
    # Identificar columnas de preguntas (1-22)
    columnas_metadatos = ['1.  Selecciona tu colegio:', '2. ¿En qué nivel estás?', '3. Paralelo']
    preguntas_cols = [col for col in df.columns if any(col.startswith(f'{i}.') for i in range(1, 23)) and col not in columnas_metadatos]
    
    # Convertir columnas de preguntas a numéricas
    for col in preguntas_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calcular puntaje promedio general
    df['Puntaje_Promedio'] = df[preguntas_cols].mean(axis=1)
    
    # Calcular puntajes por dimensión
    for dim, preguntas in dimensiones.items():
        cols_dim = [col for col in preguntas_cols if any(col.startswith(f'{p}.') for p in preguntas)]
        if cols_dim:
            df[f'Puntaje_{dim}'] = df[cols_dim].mean(axis=1)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    return df, dimensiones, preguntas_cols, timestamp

try:
    df, dimensiones, preguntas_cols, timestamp = cargar_datos()
    dims = list(dimensiones.keys())
    colegios = sorted(df['1.  Selecciona tu colegio:'].unique())
    niveles = sorted(df['2. ¿En qué nivel estás?'].unique())
    paralelos = sorted(df['3. Paralelo'].unique())
    
    st.success(f"✅ Datos cargados: {df.shape[0]} registros | Última actualización: {timestamp}")
    
except Exception as e:
    st.error(f"❌ Error al cargar datos: {e}")
    st.stop()

# ==================== FILTROS ====================
st.header("🔍 Filtros de Datos")

col1, col2, col3, col4 = st.columns(4)

with col1:
    colegios_sel = st.multiselect(
        "Seleccionar Colegios:",
        options=colegios,
        default=colegios[:3] if len(colegios) >= 3 else colegios
    )

with col2:
    niveles_sel = st.multiselect(
        "Seleccionar Niveles:",
        options=niveles,
        default=niveles[:3] if len(niveles) >= 3 else niveles
    )

with col3:
    paralelos_sel = st.multiselect(
        "Seleccionar Paralelos:",
        options=paralelos,
        default=paralelos[:2] if len(paralelos) >= 2 else paralelos
    )

with col4:
    dimensiones_sel = st.multiselect(
        "Seleccionar Dimensiones:",
        options=dims,
        default=[]
    )

# Aplicar filtros
df_filtrado = df.copy()

if len(colegios_sel) > 0:
    df_filtrado = df_filtrado[df_filtrado['1.  Selecciona tu colegio:'].isin(colegios_sel)]

if len(niveles_sel) > 0:
    df_filtrado = df_filtrado[df_filtrado['2. ¿En qué nivel estás?'].isin(niveles_sel)]

if len(paralelos_sel) > 0:
    df_filtrado = df_filtrado[df_filtrado['3. Paralelo'].isin(paralelos_sel)]

# Determinar qué columnas de puntaje usar según dimensiones seleccionadas
if len(dimensiones_sel) > 0:
    columnas_puntaje = [f'Puntaje_{dim}' for dim in dimensiones_sel]
    label_puntaje = f"Puntaje Promedio ({', '.join(dimensiones_sel)})"
else:
    columnas_puntaje = ['Puntaje_Promedio']
    label_puntaje = "Puntaje Promedio General"

# Calcular puntaje según filtro de dimensiones
if len(dimensiones_sel) > 0:
    df_filtrado['Puntaje_Filtrado'] = df_filtrado[columnas_puntaje].mean(axis=1)
else:
    df_filtrado['Puntaje_Filtrado'] = df_filtrado['Puntaje_Promedio']

st.markdown("---")

# ==cols_mostrar = ['1.  Selecciona tu colegio:', '2. ¿En qué nivel estás?', '3. Paralelo', 'Mes_Año', 'Puntaje_Filtrado']
    if len(dimensiones_sel) > 0:
        cols_mostrar.extend(columnas_puntaje)
    st.dataframe(df_filtrado[cols_mostrar].head(20))

# Mostrar dimensiones activas
if len(dimensiones_sel) > 0:
    st.info(f"📊 **Dimensiones activas:** {', '.join(dimensiones_sel)}")
else:
    st.info("📊 **Mostrando:** Promedio general de las 7 dimensiones"

col1, col2, col3, col4 = st.columns(4)Filtrad
with col1:
    st.metric("Total Registros", len(df_filtrado))
with col2:
    st.metric("Cursos Únicos", df_filtrado['Colegio_Nivel_Paralelo'].nunique())
with col3:
    st.metric("Niveles Únicos", df_filtrado['Colegio_Nivel'].nunique())
with col4:
    st.metric("Meses con Datos", df_filtrado['Mes_Año'].nunique())

with st.expander("📋 Ver vista previa de datos"):
    st.dataframe(df_filtrado[['1.  Selecciona tu colegio:', '2. ¿En qué nivel estás?', '3. Paralelo', 
                               'Mes_Año', 'Puntaje_Promedio']].head(20))

st.markdown("---")

# ==================== EVOLUCIÓN TEMPORAL POR CURSO ====================
st.header("📈 Evolución Temporal por Curso")

n_cursos = st.slider("Número de cursos a mostrar:", min_value=5, max_value=30, value=10, step=1)

# Agrupar datos por curso y mes
agrupado_curso = df_filtrado.groupby(['Colegio_Nivel_Paralelo', 'Mes_Año'])['Puntaje_Promedio'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
]).reset_index()label_puntaje
agrupado_curso.columns = ['Curso', 'Mes_Año', 'Promedio', 'Desv_Std', 'N_Estudiantes']
agrupado_curso['Fecha_Plot'] = agrupado_curso['Mes_Año'].dt.to_timestamp()

# Seleccionar top cursos
top_cursos = agrupado_curso.groupby('Curso')['N_Estudiantes'].sum().nlargest(n_cursos).index

# Crear gráficos
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor('white')

# Gráfico 1: Evolución del puntaje promedio
for curso in top_cursos:
    data = agrupado_curso[agrupado_curso['Curso'] == curso].sort_values('Fecha_Plot')
    if len(data) > 0:
        axes[0].plot(data['Fecha_Plot'], data['Promedio'], marker='o', linewidth=2, label=curso, alpha=0.8)
        axes[0].fill_between(data['Fecha_Plot'], 
                            data['Promedio'] - data['Desv_Std'], 
                            data['Promedio'] + data['Desv_Std'], 
                            alpha=0.15)

axes[0].set_xlabel('Mes', fontweight='bold')
axes[0].set_ylabel('Puntaje Promedio (1-5)', fontweight='bold')
axes[0].set_title(f'Evolución del Puntaje por Curso (Top {len(top_cursos)})', fontweight='bold')
axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(1, 5)

# Gráfico 2: Desviación estándar
for curso in top_cursos:
    data = agrupado_curso[agrupado_curso['Curso'] == curso].sort_values('Fecha_Plot')
    if len(data) > 0:
        axes[1].plot(data['Fecha_Plot'], data['Desv_Std'], marker='s', linewidth=2, label=curso, alpha=0.8)

axes[1].set_xlabel('Mes', fontweight='bold')
axes[1].set_ylabel('Desviación Estándar', fontweight='bold')
axes[1].set_title('Variabilidad por Curso', fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ==================== EVOLUCIÓN TEMPORAL POR NIVEL ====================
st.header("📈 Evolución Temporal por Nivel")

n_niveles = st.slider("Número de niveles a mostrar:", min_value=5, max_value=40, value=15, step=1)

# Agrupar datos por nivel y mes
agrupado_nivel = df_filtrado.groupby(['Colegio_Nivel', 'Mes_Año'])['Puntaje_Filtrado'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
]).reset_index()
agrupado_nivel.columns = ['Nivel', 'Mes_Año', 'Promedio', 'Desv_Std', 'N_Estudiantes']
agrupado_nivel['Fecha_Plot'] = agrupado_nivel['Mes_Año'].dt.to_timestamp()

# Seleccionar top niveles
top_niveles = agrupado_nivel.groupby('Nivel')['N_Estudiantes'].sum().nlargest(n_niveles).index

# Crear gráficos
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor('white')

# Gráfico 1: Evolución del puntaje promedio
for nivel in top_niveles:
    data = agrupado_nivel[agrupado_nivel['Nivel'] == nivel].sort_values('Fecha_Plot')
    if len(data) > 0:
        axes[0].plot(data['Fecha_Plot'], data['Promedio'], marker='o', linewidth=2.5, label=nivel, alpha=0.8)
        axes[0].fill_between(data['Fecha_Plot'], 
                            data['Promedio'] - data['Desv_Std'], 
                            data['Promedio'] + data['Desv_Std'], 
                            alpha=0.15)

axes[0].set_xlabel('Mes', fontweight='bold')
axes[0].set_ylabel(label_puntaje, fontweight='bold')
axes[0].set_title(f'Evolución del Puntaje por Nivel (Top {len(top_niveles)})', fontweight='bold')
axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(1, 5)

# Gráfico 2: Desviación estándar
for nivel in top_niveles:
    data = agrupado_nivel[agrupado_nivel['Nivel'] == nivel].sort_values('Fecha_Plot')
    if len(data) > 0:
        axes[1].plot(data['Fecha_Plot'], data['Desv_Std'], marker='s', linewidth=2.5, label=nivel, alpha=0.8)

axes[1].set_xlabel('Mes', fontweight='bold')
axes[1].set_ylabel('Desviación Estándar', fontweight='bold')
axes[1].set_title('Variabilidad por Nivel', fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ==================== TABLA RESUMEN POR CURSO ====================
st.header("📋 Resumen Estadístico por Curso")

# Resumen general por curso
resumen_curso = df_filtrado.groupby('Colegio_Nivel_Paralelo')['Puntaje_Filtrado'].agg([
    ('N_Estudiantes', 'count'),
    ('Promedio_General', 'mean'),
    ('Desv_Std', 'std'),
    ('Mínimo', 'min'),
    ('Máximo', 'max')
]).round(2).sort_values('N_Estudiantes', ascending=False)

st.dataframe(resumen_curso.head(20), use_container_width=True)

# Tabla pivote: Promedio por mes
st.subheader("📊 Promedios Mensuales por Curso")

resumen_curso_mes = df_filtrado.groupby(['Colegio_Nivel_Paralelo', 'Mes_Año'])['Puntaje_Filtrado'].agg([
    ('Promedio', 'mean'),
    ('N_Estudiantes', 'count')
]).round(2).reset_index()

tabla_promedio_curso = resumen_curso_mes.pivot(
    index='Colegio_Nivel_Paralelo', 
    columns='Mes_Año', 
    values='Promedio'
).round(2)

# Agregar columnas de totales
tabla_promedio_curso['Promedio_General'] = df_filtrado.groupby('Colegio_Nivel_Paralelo')['Puntaje_Filtrado'].mean().round(2)
tabla_promedio_curso = tabla_promedio_curso.sort_values('Promedio_General', ascending=False)

st.dataframe(tabla_promedio_curso.head(20), use_container_width=True)

st.markdown("---")

# ==================== TABLA RESUMEN POR NIVEL ====================
st.header("📋 Resumen Estadístico por Nivel")

# Resumen general por nivel
resumen_nivel = df_filtrado.groupby('Colegio_Nivel')['Puntaje_Filtrado'].agg([
    ('N_Estudiantes', 'count'),
    ('Promedio_General', 'mean'),
    ('Desv_Std', 'std'),
    ('Mínimo', 'min'),
    ('Máximo', 'max')
]).round(2).sort_values('N_Estudiantes', ascending=False)

st.dataframe(resumen_nivel.head(20), use_container_width=True)

# Tabla pivote: Promedio por mes
st.subheader("📊 Promedios Mensuales por Nivel")

resumen_nivel_mes = df_filtrado.groupby(['Colegio_Nivel', 'Mes_Año'])['Puntaje_Filtrado'].agg([
    ('Promedio', 'mean'),
    ('N_Estudiantes', 'count')
]).round(2).reset_index()

tabla_promedio_nivel = resumen_nivel_mes.pivot(
    index='Colegio_Nivel', 
    columns='Mes_Año', 
    values='Promedio'
).round(2)

# Agregar columnas de totales
tabla_promedio_nivel['Promedio_General'] = df_filtrado.groupby('Colegio_Nivel')['Puntaje_Filtrado'].mean().round(2)
tabla_promedio_nivel = tabla_promedio_nivel.sort_values('Promedio_General', ascending=False)

st.dataframe(tabla_promedio_nivel.head(20), use_container_width=True)

st.markdown("---")

# ==================== EVOLUCIÓN POR DIMENSIÓN ====================
st.header("📊 Evolución por Dimensión")

dim_seleccionada = st.selectbox("Seleccionar dimensión:", dims)

# Calcular evolución mensual de la dimensión seleccionada
col_dimension = f'Puntaje_{dim_seleccionada}'

if col_dimension in df_filtrado.columns:
    evolucion_dim = df_filtrado.groupby('Mes_Año')[col_dimension].agg([
        ('Promedio', 'mean'),
        ('Desv_Std', 'std'),
        ('N', 'count')
    ]).reset_index()
    evolucion_dim['Fecha_Plot'] = evolucion_dim['Mes_Año'].dt.to_timestamp()
    evolucion_dim = evolucion_dim.sort_values('Fecha_Plot')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    ax.plot(evolucion_dim['Fecha_Plot'], evolucion_dim['Promedio'], 
            marker='o', linewidth=3, color=COLOR_MEJORA, label='Promedio', alpha=0.9)
    ax.fill_between(evolucion_dim['Fecha_Plot'], 
                    evolucion_dim['Promedio'] - evolucion_dim['Desv_Std'], 
                    evolucion_dim['Promedio'] + evolucion_dim['Desv_Std'], 
                    alpha=0.2, color=COLOR_MEJORA)
    
    ax.set_xlabel('Mes', fontweight='bold')
    ax.set_ylabel('Puntaje Promedio (1-5)', fontweight='bold')
    ax.set_title(f'Evolución Temporal: {dim_seleccionada}', fontweight='bold', pad=15)
    ax.legend(frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1, 5)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Mostrar tabla de datos
    st.dataframe(evolucion_dim[['Mes_Año', 'Promedio', 'Desv_Std', 'N']].round(2), use_container_width=True)
else:
    st.warning(f"No se encontró la columna {col_dimension} en los datos")

st.markdown("---")

# ==================== TENDENCIAS Y REGRESIÓN ====================
st.header("📈 Análisis de Tendencias")

st.markdown("Análisis de regresión lineal para identificar tendencias de mejora o disminución a lo largo del tiempo.")

# Preparar datos para regresión
df_reg = df_filtrado.copy()
df_reg = df_reg[df_reg['Fecha'].notna() & df_reg['Puntaje_Filtrado'].notna()]
df_reg['Dias_Desde_Inicio'] = (df_reg['Fecha'] - df_reg['Fecha'].min()).dt.days

if len(df_reg) > 1:
    X = df_reg[['Dias_Desde_Inicio']].values
    y = df_reg['Puntaje_Filtrado'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    pendiente = model.coef_[0]
    intercepto = model.intercept_
    r_squared = model.score(X, y)
    
    # Calcular predicciones
    y_pred = model.predict(X)
    
    # Visualización
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    ax.scatter(df_reg['Fecha'], y, alpha=0.3, color=COLOR_PRE, label='Datos reales')
    ax.plot(df_reg['Fecha'].sort_values(), 
            model.predict(df_reg[df_reg['Fecha'].notna()].sort_values('Fecha')[['Dias_Desde_Inicio']]),
            color=COLOR_POST, linewidth=3, label='Tendencia (Regresión Lineal)')
    
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Puntaje Promedio', fontweight='bold')
    ax.set_title('Tendencia General de Puntajes', fontweight='bold', pad=15)
    ax.legend(frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Métricas de la regresión
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pendiente (cambio diario)", f"{pendiente:.6f}")
    with col2:
        st.metric("R² (ajuste del modelo)", f"{r_squared:.4f}")
    with col3:
        cambio_mensual = pendiente * 30
        emoji = "📈" if cambio_mensual > 0 else "📉"
        st.metric(f"{emoji} Cambio estimado mensual", f"{cambio_mensual:.4f}")
    
    if pendiente > 0:
        st.success(f"✅ Se observa una tendencia de mejora: +{cambio_mensual:.4f} puntos por mes en promedio")
    else:
        st.warning(f"⚠️ Se observa una tendencia de disminución: {cambio_mensual:.4f} puntos por mes en promedio")
else:
    st.warning("No hay suficientes datos para realizar el análisis de tendencias")

st.markdown("---")

# ==================== COMPARACIÓN ENTRE COLEGIOS ====================
st.header("🏫 Comparación entre Colegios")

# Promedio general por colegio
promedio_colegios = df_filtrado.groupby('1.  Selecciona tu colegio:')['Puntaje_Filtrado'].agg([
    ('Promedio', 'mean'),
    ('Desv_Std', 'std'),
    ('N_Estudiantes', 'count')
]).round(2).sort_values('Promedio', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')

colors = [COLOR_MEJORA if x >= promedio_colegios['Promedio'].mean() else COLOR_DISMINUYE 
          for x in promedio_colegios['Promedio']]
bars = ax.barh(promedio_colegios.index, promedio_colegios['Promedio'], 
               color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

# Línea del promedio general
ax.axvline(x=promedio_colegios['Promedio'].mean(), 
          color='#34495e', linestyle='--', linewidth=2, 
          label=f'Promedio General ({promedio_colegios["Promedio"].mean():.2f})')

ax.set_xlabel('Puntaje Promedio', fontweight='bold')
ax.set_title('Comparación de Puntajes entre Colegios', fontweight='bold', pad=15)
ax.legend(frameon=True, shadow=True, fancybox=True)
ax.grid(axis='x', alpha=0.3)

# Añadir valores en las barras
for i, (idx, row) in enumerate(promedio_colegios.iterrows()):
    ax.text(row['Promedio'], i, f" {row['Promedio']:.2f}", 
            va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

st.dataframe(promedio_colegios, use_container_width=True)

# ==================== F2: ANÁLISIS DE DOCENTES ====================
st.header("👨‍🏫 F2: Análisis de Aplicaciones por Docentes")

st.markdown("""
Este análisis muestra cuántas veces cada docente ha aplicado la metodología de aprendizaje cooperativo.
El "Número de semana de aplicación" indica el total de aplicaciones realizadas.
""")

# Cargar datos F2
@st.cache_data
def cargar_datos_f2():
    sheet_id = '12JoLMA_A_-MuLqxbTTEsmBPVBhNSNbllyAlaThO2HDc'
    gid = '1961154565'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
    
    df_f2 = pd.read_csv(url)
    df_f2.columns = df_f2.columns.str.strip()
    
    # Renombrar columnas
    col_mapping = {
        df_f2.columns[0]: 'Marca_Temporal',
        df_f2.columns[1]: 'Nombre_Docente',
        df_f2.columns[2]: 'Semana_Aplicacion',
        df_f2.columns[3]: 'Colegio',
        df_f2.columns[4]: 'Nivel',
        df_f2.columns[5]: 'Paralelo',
        df_f2.columns[6]: 'Metodologia'
    }
    df_f2 = df_f2.rename(columns=col_mapping)
    
    # Asegurar que Semana_Aplicacion sea numérico
    df_f2['Semana_Aplicacion'] = pd.to_numeric(df_f2['Semana_Aplicacion'], errors='coerce')
    
    # Crear claves compuestas
    df_f2['Curso'] = df_f2['Colegio'] + '_' + df_f2['Nivel'] + '_' + df_f2['Paralelo']
    df_f2['Colegio_Nivel'] = df_f2['Colegio'] + '_' + df_f2['Nivel']
    
    return df_f2

try:
    df_f2 = cargar_datos_f2()
    st.success(f"✅ Datos F2 cargados: {df_f2.shape[0]} registros de aplicaciones")
    
    # Análisis por Docente
    st.subheader("📊 Tabla 1: Aplicaciones por Docente")
    
    aplicaciones_por_docente = df_f2.groupby('Nombre_Docente').agg({
        'Semana_Aplicacion': 'max',
        'Colegio': lambda x: ', '.join(x.unique()),
        'Curso': 'nunique',
        'Metodologia': 'nunique'
    }).reset_index()
    
    aplicaciones_por_docente.columns = ['Docente', 'Total_Aplicaciones', 'Colegios', 'Cursos_Diferentes', 'Metodologias_Diferentes']
    aplicaciones_por_docente = aplicaciones_por_docente.sort_values('Total_Aplicaciones', ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Docentes", len(aplicaciones_por_docente))
    with col2:
        st.metric("Promedio Aplicaciones", f"{aplicaciones_por_docente['Total_Aplicaciones'].mean():.1f}")
    with col3:
        st.metric("Máximo", int(aplicaciones_por_docente['Total_Aplicaciones'].max()))
    with col4:
        st.metric("Mínimo", int(aplicaciones_por_docente['Total_Aplicaciones'].min()))
    
    st.dataframe(aplicaciones_por_docente, use_container_width=True, hide_index=True)
    
    # Análisis por Curso
    st.subheader("📊 Tabla 2: Aplicaciones por Curso")
    
    aplicaciones_por_curso = df_f2.groupby(['Colegio', 'Nivel', 'Paralelo']).agg({
        'Nombre_Docente': 'nunique',
        'Semana_Aplicacion': 'max',
        'Metodologia': 'nunique'
    }).reset_index()
    
    aplicaciones_por_curso.columns = ['Colegio', 'Nivel', 'Paralelo', 'Docentes_Diferentes', 'Total_Aplicaciones', 'Metodologias_Diferentes']
    aplicaciones_por_curso = aplicaciones_por_curso.sort_values('Total_Aplicaciones', ascending=False)
    
    st.dataframe(aplicaciones_por_curso, use_container_width=True, hide_index=True)
    
    # Análisis por Nivel
    st.subheader("📊 Tabla 3: Aplicaciones por Nivel")
    
    aplicaciones_por_nivel = df_f2.groupby(['Colegio', 'Nivel']).agg({
        'Nombre_Docente': 'nunique',
        'Semana_Aplicacion': 'max',
        'Paralelo': 'nunique',
        'Metodologia': 'nunique'
    }).reset_index()
    
    aplicaciones_por_nivel.columns = ['Colegio', 'Nivel', 'Docentes_Diferentes', 'Total_Aplicaciones', 'Paralelos_Diferentes', 'Metodologias_Diferentes']
    aplicaciones_por_nivel = aplicaciones_por_nivel.sort_values('Total_Aplicaciones', ascending=False)
    
    st.dataframe(aplicaciones_por_nivel, use_container_width=True, hide_index=True)
    
    # Análisis por Colegio
    st.subheader("📊 Tabla 4: Aplicaciones por Colegio")
    
    aplicaciones_por_colegio = df_f2.groupby('Colegio').agg({
        'Nombre_Docente': 'nunique',
        'Semana_Aplicacion': 'max',
        'Nivel': 'nunique',
        'Curso': 'nunique',
        'Metodologia': 'nunique'
    }).reset_index()
    
    aplicaciones_por_colegio.columns = ['Colegio', 'Docentes_Diferentes', 'Total_Aplicaciones', 'Niveles_Diferentes', 'Cursos_Diferentes', 'Metodologias_Diferentes']
    aplicaciones_por_colegio = aplicaciones_por_colegio.sort_values('Total_Aplicaciones', ascending=False)
    
    st.dataframe(aplicaciones_por_colegio, use_container_width=True, hide_index=True)
    
    # Gráfico de aplicaciones por docente
    st.subheader("📈 Visualización: Top 15 Docentes")
    
    top_15_docentes = aplicaciones_por_docente.head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    bars = ax.barh(range(len(top_15_docentes)), top_15_docentes['Total_Aplicaciones'], 
                   color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(top_15_docentes)))
    ax.set_yticklabels(top_15_docentes['Docente'])
    ax.set_xlabel('Total de Aplicaciones', fontweight='bold')
    ax.set_title('Top 15 Docentes por Aplicaciones de Metodología', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Añadir valores en las barras
    for i, val in enumerate(top_15_docentes['Total_Aplicaciones']):
        ax.text(val, i, f' {int(val)}', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"❌ Error al cargar datos F2: {e}")

st.markdown("---")

# ==================== F3: ANÁLISIS DE OBSERVACIÓN ====================
st.header("🔍 F3: Datos de Observación")

st.markdown("""
Esta sección muestra los datos de observación de la metodología de aprendizaje cooperativo.
""")

# Cargar datos F3
@st.cache_data
def cargar_datos_f3():
    sheet_id = '12JoLMA_A_-MuLqxbTTEsmBPVBhNSNbllyAlaThO2HDc'
    gid = '672136638'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
    
    df_f3 = pd.read_csv(url)
    df_f3.columns = df_f3.columns.str.strip()
    
    return df_f3

try:
    df_f3 = cargar_datos_f3()
    st.success(f"✅ Datos F3 cargados: {df_f3.shape[0]} registros, {df_f3.shape[1]} columnas")
    
    # Resumen de datos
    st.subheader("📊 Resumen de Datos F3")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Observaciones", len(df_f3))
    with col2:
        st.metric("Total de Columnas", len(df_f3.columns))
    
    # Mostrar información de columnas
    with st.expander("📋 Ver información de columnas"):
        info_columnas = pd.DataFrame({
            'Columna': df_f3.columns,
            'Tipo': df_f3.dtypes.values,
            'Valores Únicos': [df_f3[col].nunique() for col in df_f3.columns],
            'Nulos': [df_f3[col].isnull().sum() for col in df_f3.columns]
        })
        st.dataframe(info_columnas, use_container_width=True, hide_index=True)
    
    # Tabla completa de datos
    st.subheader("📋 Tabla Completa de Observaciones")
    
    # Configurar display de pandas
    pd.set_option('display.max_columns', None)
    
    st.dataframe(df_f3, use_container_width=True, hide_index=True)
    
    # Análisis de frecuencias por categorías
    st.subheader("📊 Análisis de Frecuencias")
    
    # Identificar columnas categóricas
    columnas_categoricas = []
    for col in df_f3.columns:
        if df_f3[col].dtype == 'object' or df_f3[col].nunique() < 20:
            columnas_categoricas.append(col)
    
    if len(columnas_categoricas) > 0:
        col_seleccionada = st.selectbox(
            "Seleccionar columna para análisis de frecuencias:",
            options=columnas_categoricas[:10]  # Primeras 10 columnas categóricas
        )
        
        if col_seleccionada:
            frecuencias = df_f3[col_seleccionada].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('white')
                
                if len(frecuencias) <= 15:
                    bars = ax.bar(range(len(frecuencias)), frecuencias.values, 
                                 color='coral', alpha=0.8, edgecolor='white')
                    ax.set_xticks(range(len(frecuencias)))
                    ax.set_xticklabels(frecuencias.index, rotation=45, ha='right')
                else:
                    # Solo mostrar top 15
                    top_15 = frecuencias.head(15)
                    bars = ax.bar(range(len(top_15)), top_15.values, 
                                 color='coral', alpha=0.8, edgecolor='white')
                    ax.set_xticks(range(len(top_15)))
                    ax.set_xticklabels(top_15.index, rotation=45, ha='right')
                
                ax.set_ylabel('Frecuencia', fontweight='bold')
                ax.set_title(f'Distribución de: {col_seleccionada}', fontweight='bold', pad=15)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.dataframe(frecuencias.to_frame('Frecuencia'), use_container_width=True)
    
    # Opción para exportar datos
    st.subheader("💾 Exportar Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_f3.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv,
            file_name="f3_observacion.csv",
            mime="text/csv",
        )
    
except Exception as e:
    st.error(f"❌ Error al cargar datos F3: {e}")

# Footer
st.markdown("---")
st.markdown("**Análisis Temporal de Metodología de Aprendizaje Cooperativo** | Desarrollado con Streamlit")
