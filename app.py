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
    page_title="Análisis Metodología Aprendizaje Cooperativo",
    page_icon="📊",
    layout="wide"
)

# Estilo personalizado con color de fondo y logo
st.markdown(
    """
    <style>
    .stApp {
        background-color: #edaf6b;
    }
    .main .block-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
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
st.title("📊 Análisis de Metodología de Aprendizaje Cooperativo")

st.markdown("""
Este análisis evalúa la efectividad de la metodología basada en 7 dimensiones:
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
    sheet_id = '1-K16AK3JmXJyJQhl_KVFJIkzG3PZ9fVL7i_lY5jCrrE'
    sheet_name = 'Respuestas%20de%20formulario%201'  # URL encoded (espacios = %20)
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    
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
    columnas_preguntas = df.columns[5:27]
    
    for dim, preguntas in dimensiones.items():
        cols = [columnas_preguntas[p-1] for p in preguntas]
        df[dim] = df[cols].mean(axis=1)
    
    df_pre = df[df.iloc[:, 1] == 'PRE'].copy()
    df_post = df[df.iloc[:, 1] == 'POST'].copy()
    
    return df, df_pre, df_post, dimensiones, columnas_preguntas

try:
    df, df_pre, df_post, dimensiones, columnas_preguntas = cargar_datos()
    dims = list(dimensiones.keys())
    colegio_col = df.columns[2]
    momento_col = df.columns[1]
    id_col = df.columns[0]
    nombre_col = '2. ¿Cuál es tu nombre?'  # Columna con el nombre del estudiante
    colegios = sorted(df[colegio_col].unique())
    
    st.success(f"✅ Datos cargados: {df.shape[0]} registros, {len(df_pre)} PRE, {len(df_post)} POST")
    
except Exception as e:
    st.error(f"❌ Error al cargar datos: {e}")
    st.stop()

# Sidebar para navegación
st.sidebar.title("📑 Navegación")
seccion = st.sidebar.radio(
    "Seleccionar análisis:",
    [
        "1. Resumen de Datos",
        "2. Estadísticas Descriptivas",
        "3. Cambios PRE vs POST",
        "4. Correlaciones",
        "5. Análisis por Estudiante",
        "6. Distribución por Colegio",
        "7. Ranking de Mejoras",
        "8. Outliers",
        "9. Frecuencias de Respuesta",
        "10. Análisis de Regresión",
        "11. Resumen Ejecutivo"
    ]
)

# Filtros en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtros")
mostrar_por_colegio = st.sidebar.checkbox("Mostrar análisis por colegio", value=False)
if mostrar_por_colegio:
    colegio_seleccionado = st.sidebar.selectbox("Seleccionar colegio:", ["Todos"] + list(colegios))
else:
    colegio_seleccionado = "Todos"

# ==================== SECCIÓN 1: RESUMEN DE DATOS ====================
if seccion == "1. Resumen de Datos":
    st.header("1️⃣ Resumen de Datos")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Registros", df.shape[0])
    with col2:
        st.metric("Registros PRE", len(df_pre))
    with col3:
        st.metric("Registros POST", len(df_post))
    with col4:
        st.metric("Colegios", df[colegio_col].nunique())
    
    st.subheader("Vista previa de datos")
    st.dataframe(df.head(10))
    
    st.subheader("Información de columnas")
    st.write(f"Columnas totales: {df.shape[1]}")
    st.write(f"Columnas de preguntas: {list(columnas_preguntas)}")

# ==================== SECCIÓN 2: ESTADÍSTICAS DESCRIPTIVAS ====================
elif seccion == "2. Estadísticas Descriptivas":
    st.header("2️⃣ Estadísticas Descriptivas por Dimensión")
    
    # Resumen general
    stats_summary = pd.DataFrame({
        'Dimensión': dims,
        'Media PRE': [df_pre[d].mean() for d in dims],
        'Mediana PRE': [df_pre[d].median() for d in dims],
        'Std PRE': [df_pre[d].std() for d in dims],
        'Media POST': [df_post[d].mean() for d in dims],
        'Mediana POST': [df_post[d].median() for d in dims],
        'Std POST': [df_post[d].std() for d in dims]
    })
    stats_summary['Cambio Media'] = stats_summary['Media POST'] - stats_summary['Media PRE']
    stats_summary['Cambio Mediana'] = stats_summary['Mediana POST'] - stats_summary['Mediana PRE']
    stats_summary = stats_summary.round(3)
    
    st.subheader("📊 Resumen General")
    st.dataframe(stats_summary, use_container_width=True)
    
    # Gráficos
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Medias", "📊 Medianas", "📉 Cambio Media", "📉 Cambio Mediana"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        x = np.arange(len(dims))
        width = 0.35
        ax.bar(x - width/2, stats_summary['Media PRE'], width, label='PRE', alpha=0.85, 
               color=COLOR_PRE, edgecolor='white', linewidth=1.5)
        ax.bar(x + width/2, stats_summary['Media POST'], width, label='POST', alpha=0.85, 
               color=COLOR_POST, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('Dimensiones', fontweight='bold')
        ax.set_ylabel('Media', fontweight='bold')
        ax.set_title('Comparación de Medias PRE vs POST (Total)', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(dims, rotation=45, ha='right')
        ax.legend(frameon=True, shadow=True, fancybox=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.bar(x - width/2, stats_summary['Mediana PRE'], width, label='PRE', alpha=0.85, 
               color=COLOR_PRE, edgecolor='white', linewidth=1.5)
        ax.bar(x + width/2, stats_summary['Mediana POST'], width, label='POST', alpha=0.85, 
               color=COLOR_POST, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('Dimensiones', fontweight='bold')
        ax.set_ylabel('Mediana', fontweight='bold')
        ax.set_title('Comparación de Medianas PRE vs POST (Total)', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(dims, rotation=45, ha='right')
        ax.legend(frameon=True, shadow=True, fancybox=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        colors = [COLOR_MEJORA if x > 0 else COLOR_DISMINUYE for x in stats_summary['Cambio Media']]
        bars = ax.barh(stats_summary['Dimensión'], stats_summary['Cambio Media'], 
                       color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=2, alpha=0.7)
        ax.set_xlabel('Cambio en Media (POST - PRE)', fontweight='bold')
        ax.set_title('Cambio en Media por Dimensión (Total)', fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        # Añadir valores en las barras
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                    f'{width_val:.3f}', ha='left' if width_val > 0 else 'right', 
                    va='center', fontweight='bold', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab4:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        colors = [COLOR_MEJORA if x > 0 else COLOR_DISMINUYE for x in stats_summary['Cambio Mediana']]
        bars = ax.barh(stats_summary['Dimensión'], stats_summary['Cambio Mediana'], 
                       color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=2, alpha=0.7)
        ax.set_xlabel('Cambio en Mediana (POST - PRE)', fontweight='bold')
        ax.set_title('Cambio en Mediana por Dimensión (Total)', fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        # Añadir valores en las barras
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                    f'{width_val:.3f}', ha='left' if width_val > 0 else 'right', 
                    va='center', fontweight='bold', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        plt.tight_layout()
        st.pyplot(fig)
    
    # Por colegio
    if mostrar_por_colegio and colegio_seleccionado != "Todos":
        st.subheader(f"📊 Análisis para: {colegio_seleccionado}")
        
        df_pre_col = df_pre[df_pre[colegio_col] == colegio_seleccionado]
        df_post_col = df_post[df_post[colegio_col] == colegio_seleccionado]
        
        stats_colegio = pd.DataFrame({
            'Dimensión': dims,
            'Media PRE': [df_pre_col[d].mean() for d in dims],
            'Mediana PRE': [df_pre_col[d].median() for d in dims],
            'Media POST': [df_post_col[d].mean() for d in dims],
            'Mediana POST': [df_post_col[d].median() for d in dims],
            'Cambio Media': [df_post_col[d].mean() - df_pre_col[d].mean() for d in dims],
            'N PRE': len(df_pre_col),
            'N POST': len(df_post_col)
        }).round(3)
        
        st.dataframe(stats_colegio, use_container_width=True)

# ==================== SECCIÓN 3: CAMBIOS PRE VS POST ====================
elif seccion == "3. Cambios PRE vs POST":
    st.header("3️⃣ Mejoría/Disminución por Dimensión")
    
    stats_summary = pd.DataFrame({
        'Dimensión': dims,
        'Media PRE': [df_pre[d].mean() for d in dims],
        'Media POST': [df_post[d].mean() for d in dims]
    })
    stats_summary['Cambio'] = stats_summary['Media POST'] - stats_summary['Media PRE']
    stats_summary = stats_summary.round(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        x = np.arange(len(dims))
        width = 0.35
        ax.bar(x - width/2, stats_summary['Media PRE'], width, label='PRE', alpha=0.85, 
               color=COLOR_PRE, edgecolor='white', linewidth=1.5)
        ax.bar(x + width/2, stats_summary['Media POST'], width, label='POST', alpha=0.85, 
               color=COLOR_POST, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('Dimensiones', fontweight='bold')
        ax.set_ylabel('Puntaje Promedio', fontweight='bold')
        ax.set_title('Comparación PRE vs POST', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(dims, rotation=45, ha='right')
        ax.legend(frameon=True, shadow=True, fancybox=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        colors = [COLOR_MEJORA if x > 0 else COLOR_DISMINUYE for x in stats_summary['Cambio']]
        bars = ax.barh(stats_summary['Dimensión'], stats_summary['Cambio'], 
                       color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=2, alpha=0.7)
        ax.set_xlabel('Cambio (POST - PRE)', fontweight='bold')
        ax.set_title('Mejoría/Disminución', fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        # Añadir valores en las barras
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                    f'{width_val:.3f}', ha='left' if width_val > 0 else 'right', 
                    va='center', fontweight='bold', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        plt.tight_layout()
        st.pyplot(fig)

# ==================== SECCIÓN 4: CORRELACIONES ====================
elif seccion == "4. Correlaciones":
    st.header("4️⃣ Correlación entre Dimensiones")
    
    st.markdown("""
    **Interpretación:**
    - **Correlación positiva (+1)**: Cuando una dimensión aumenta, la otra también aumenta
    - **Correlación negativa (-1)**: Cuando una aumenta, la otra disminuye
    - **Sin correlación (0)**: Las dimensiones varían independientemente
    """)
    
    corr_pre = df_pre[dims].corr()
    corr_post = df_post[dims].corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlaciones PRE")
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        sns.heatmap(corr_pre, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    square=True, ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Correlación'}, 
                    vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
                    annot_kws={'fontsize': 9, 'fontweight': 'bold'})
        ax.set_title('Correlaciones PRE', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Correlaciones POST")
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        sns.heatmap(corr_post, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    square=True, ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Correlación'}, 
                    vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
                    annot_kws={'fontsize': 9, 'fontweight': 'bold'})
        ax.set_title('Correlaciones POST', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Análisis de cambios
    st.subheader("📊 Cambios en Correlaciones")
    
    cambios_corr = []
    for i, dim1 in enumerate(dims):
        for j, dim2 in enumerate(dims):
            if i < j:
                corr_pre_val = corr_pre.loc[dim1, dim2]
                corr_post_val = corr_post.loc[dim1, dim2]
                cambio = corr_post_val - corr_pre_val
                
                cambios_corr.append({
                    'Dimensión 1': dim1,
                    'Dimensión 2': dim2,
                    'Corr PRE': round(corr_pre_val, 3),
                    'Corr POST': round(corr_post_val, 3),
                    'Cambio': round(cambio, 3),
                    'Interpretación': 'Más relacionadas' if cambio > 0.1 else ('Menos relacionadas' if cambio < -0.1 else 'Similar')
                })
    
    df_cambios_corr = pd.DataFrame(cambios_corr).sort_values('Cambio', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Mayor aumento en correlación:**")
        st.dataframe(df_cambios_corr.head(5), use_container_width=True)
    
    with col2:
        st.write("**Mayor disminución en correlación:**")
        st.dataframe(df_cambios_corr.tail(5), use_container_width=True)

# ==================== SECCIÓN 5: ANÁLISIS POR ESTUDIANTE ====================
elif seccion == "5. Análisis por Estudiante":
    st.header("5️⃣ Estudiantes que Aumentan/Disminuyen por Dimensión")
    
    df_paired = df_pre.merge(df_post, on=nombre_col, suffixes=('_pre', '_post'), how='inner')
    
    cambios = []
    for dim in dims:
        aumentan = (df_paired[f'{dim}_post'] > df_paired[f'{dim}_pre']).sum()
        disminuyen = (df_paired[f'{dim}_post'] < df_paired[f'{dim}_pre']).sum()
        igual = (df_paired[f'{dim}_post'] == df_paired[f'{dim}_pre']).sum()
        cambios.append({'Dimensión': dim, 'Aumentan': aumentan, 'Disminuyen': disminuyen, 'Igual': igual})
    
    df_cambios = pd.DataFrame(cambios)
    
    st.dataframe(df_cambios, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    x = np.arange(len(df_cambios))
    width = 0.25
    
    ax.bar(x - width, df_cambios['Aumentan'], width, label='Aumentan', 
           color=COLOR_MEJORA, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.bar(x, df_cambios['Disminuyen'], width, label='Disminuyen', 
           color=COLOR_DISMINUYE, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.bar(x + width, df_cambios['Igual'], width, label='Igual', 
           color=COLOR_NEUTRAL, alpha=0.85, edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Dimensiones', fontweight='bold')
    ax.set_ylabel('Número de Estudiantes', fontweight='bold')
    ax.set_title('Cambios en Puntaje por Dimensión', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_cambios['Dimensión'], rotation=45, ha='right')
    ax.legend(frameon=True, shadow=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

# ==================== SECCIÓN 6: DISTRIBUCIÓN POR COLEGIO ====================
elif seccion == "6. Distribución por Colegio":
    st.header("6️⃣ Distribución Boxplot por Colegio")
    
    dim_seleccionada = st.selectbox("Seleccionar dimensión:", dims)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    positions = []
    labels = []
    data_to_plot = []
    
    for i, colegio in enumerate(colegios):
        pre_data = df[(df[colegio_col] == colegio) & (df[momento_col] == 'PRE')][dim_seleccionada].dropna()
        post_data = df[(df[colegio_col] == colegio) & (df[momento_col] == 'POST')][dim_seleccionada].dropna()
        
        data_to_plot.append(pre_data)
        positions.append(i * 3)
        labels.append(f'{colegio}\nPRE')
        
        data_to_plot.append(post_data)
        positions.append(i * 3 + 1)
        labels.append(f'{colegio}\nPOST')
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2))
    
    for i, patch in enumerate(bp['boxes']):
        if i % 2 == 0:
            patch.set_facecolor('steelblue')
        else:
            patch.set_facecolor('coral')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Puntaje')
    ax.set_title(f'Distribución por Colegio: {dim_seleccionada}')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ==================== SECCIÓN 7: RANKING ====================
elif seccion == "7. Ranking de Mejoras":
    st.header("7️⃣ Ranking de Dimensiones con Mayor Mejora")
    
    stats_summary = pd.DataFrame({
        'Dimensión': dims,
        'Media PRE': [df_pre[d].mean() for d in dims],
        'Media POST': [df_post[d].mean() for d in dims]
    })
    stats_summary['Cambio'] = stats_summary['Media POST'] - stats_summary['Media PRE']
    stats_summary = stats_summary.round(3)
    
    ranking = stats_summary.sort_values('Cambio', ascending=False).reset_index(drop=True)
    ranking['Ranking'] = range(1, len(ranking) + 1)
    ranking = ranking[['Ranking', 'Dimensión', 'Media PRE', 'Media POST', 'Cambio']]
    
    st.dataframe(ranking, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    colors = [COLOR_MEJORA if x > 0 else COLOR_DISMINUYE for x in ranking['Cambio']]
    bars = ax.barh(ranking['Dimensión'], ranking['Cambio'], 
                   color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Cambio (POST - PRE)', fontweight='bold')
    ax.set_title('Ranking de Mejoras por Dimensión', fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    # Añadir valores en las barras
    for bar in bars:
        width_val = bar.get_width()
        ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                f'{width_val:.3f}', ha='left' if width_val > 0 else 'right', 
                va='center', fontweight='bold', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    st.pyplot(fig)

# ==================== SECCIÓN 8: OUTLIERS ====================
elif seccion == "8. Outliers":
    st.header("8️⃣ Detección de Outliers")
    
    def detectar_outliers(data, columna):
        Q1 = data[columna].quantile(0.25)
        Q3 = data[columna].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = data[(data[columna] < lower) | (data[columna] > upper)]
        return len(outliers), lower, upper
    
    outliers_summary = []
    for momento, df_temp in [('PRE', df_pre), ('POST', df_post)]:
        for dim in dims:
            n_outliers, lower, upper = detectar_outliers(df_temp, dim)
            outliers_summary.append({
                'Momento': momento,
                'Dimensión': dim,
                'N_Outliers': n_outliers,
                'Límite_Inferior': round(lower, 2),
                'Límite_Superior': round(upper, 2)
            })
    
    df_outliers = pd.DataFrame(outliers_summary)
    st.dataframe(df_outliers, use_container_width=True)

# ==================== SECCIÓN 9: FRECUENCIAS ====================
elif seccion == "9. Frecuencias de Respuesta":
    st.header("9️⃣ Frecuencias de Respuesta (Bajas 1-2 vs Altas 4-5)")
    
    freq_summary = []
    
    for momento, df_temp in [('PRE', df_pre), ('POST', df_post)]:
        for dim, preguntas in dimensiones.items():
            cols = [columnas_preguntas[p-1] for p in preguntas]
            data = df_temp[cols].values.flatten()
            data = data[~np.isnan(data)]
            
            total = len(data)
            bajas = np.sum((data >= 1) & (data <= 2)) / total * 100 if total > 0 else 0
            altas = np.sum((data >= 4) & (data <= 5)) / total * 100 if total > 0 else 0
            
            freq_summary.append({
                'Momento': momento,
                'Dimensión': dim,
                '% Bajas (1-2)': round(bajas, 2),
                '% Altas (4-5)': round(altas, 2)
            })
    
    df_freq = pd.DataFrame(freq_summary)
    st.dataframe(df_freq, use_container_width=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df_pre_freq = df_freq[df_freq['Momento'] == 'PRE']
    df_post_freq = df_freq[df_freq['Momento'] == 'POST']
    
    x = np.arange(len(dims))
    width = 0.35
    
    axes[0].bar(x - width/2, df_pre_freq['% Bajas (1-2)'], width, label='Bajas (1-2)', color='red', alpha=0.7)
    axes[0].bar(x + width/2, df_pre_freq['% Altas (4-5)'], width, label='Altas (4-5)', color='green', alpha=0.7)
    axes[0].set_title('PRE: Distribución de Respuestas')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dims, rotation=45, ha='right')
    axes[0].set_ylabel('Porcentaje (%)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(x - width/2, df_post_freq['% Bajas (1-2)'], width, label='Bajas (1-2)', color='red', alpha=0.7)
    axes[1].bar(x + width/2, df_post_freq['% Altas (4-5)'], width, label='Altas (4-5)', color='green', alpha=0.7)
    axes[1].set_title('POST: Distribución de Respuestas')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dims, rotation=45, ha='right')
    axes[1].set_ylabel('Porcentaje (%)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# ==================== SECCIÓN 10: REGRESIÓN ====================
elif seccion == "10. Análisis de Regresión":
    st.header("🔟 Análisis de Regresión: Predictores de Clima y Motivación")
    
    st.markdown("""
    **Interpretación del R²:**
    - **R² alto (>0.5)**: Predictor fuerte
    - **R² medio (0.3-0.5)**: Predictor moderado
    - **R² bajo (<0.3)**: Predictor débil
    """)
    
    dims_predictoras = [d for d in dims if d not in ['Clima', 'Motivación']]
    
    resultados_regresion = []
    
    for momento, df_temp in [('PRE', df_pre), ('POST', df_post)]:
        for target in ['Clima', 'Motivación']:
            for predictor in dims_predictoras:
                X = df_temp[[predictor]].dropna()
                y = df_temp.loc[X.index, target].dropna()
                X = X.loc[y.index]
                
                if len(X) > 0:
                    model = LinearRegression()
                    model.fit(X, y)
                    r2 = model.score(X, y)
                    coef = model.coef_[0]
                    
                    resultados_regresion.append({
                        'Momento': momento,
                        'Variable_Dependiente': target,
                        'Predictor': predictor,
                        'R²': round(r2, 4),
                        'Coeficiente': round(coef, 4)
                    })
    
    df_regresion = pd.DataFrame(resultados_regresion)
    
    st.subheader("📊 Resultados Completos")
    st.dataframe(df_regresion.sort_values(['Variable_Dependiente', 'Momento', 'R²'], ascending=[True, True, False]),
                 use_container_width=True)
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for idx, target in enumerate(['Clima', 'Motivación']):
        for jdx, momento in enumerate(['PRE', 'POST']):
            ax = axes[idx, jdx]
            data = df_regresion[(df_regresion['Variable_Dependiente'] == target) & 
                                (df_regresion['Momento'] == momento)].sort_values('R²', ascending=True)
            
            colors = ['green' if r > 0.5 else 'orange' if r > 0.3 else 'red' for r in data['R²']]
            ax.barh(data['Predictor'], data['R²'], color=colors, alpha=0.7)
            ax.set_xlabel('R² Score')
            ax.set_title(f'{target} - {momento}', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlim(0, 1)
            ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Mejores predictores
    st.subheader("🎯 Mejores Predictores")
    mejores = df_regresion.loc[df_regresion.groupby(['Momento', 'Variable_Dependiente'])['R²'].idxmax()]
    st.dataframe(mejores.sort_values(['Variable_Dependiente', 'Momento']), use_container_width=True)

# ==================== SECCIÓN 11: RESUMEN EJECUTIVO ====================
elif seccion == "11. Resumen Ejecutivo":
    st.header("📋 Resumen Ejecutivo del Análisis")
    
    # Calcular métricas
    stats_summary = pd.DataFrame({
        'Dimensión': dims,
        'Media PRE': [df_pre[d].mean() for d in dims],
        'Media POST': [df_post[d].mean() for d in dims]
    })
    stats_summary['Cambio'] = stats_summary['Media POST'] - stats_summary['Media PRE']
    
    ranking = stats_summary.sort_values('Cambio', ascending=False).reset_index(drop=True)
    
    df_paired = df_pre.merge(df_post, on=id_col, suffixes=('_pre', '_post'))
    
    # Mostrar resumen
    st.subheader("📊 Estadísticas Generales")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Estudiantes PRE", len(df_pre))
    with col2:
        st.metric("Total Estudiantes POST", len(df_post))
    with col3:
        st.metric("Colegios Evaluados", df[colegio_col].nunique())
    
    st.subheader("🏆 Dimensión con Mayor Mejora")
    top_mejora = ranking.iloc[0]
    st.success(f"**{top_mejora['Dimensión']}**: +{top_mejora['Cambio']:.3f} puntos")
    
    st.subheader("📉 Dimensión con Menor Cambio")
    menor_mejora = ranking.iloc[-1]
    if menor_mejora['Cambio'] < 0:
        st.error(f"**{menor_mejora['Dimensión']}**: {menor_mejora['Cambio']:.3f} puntos")
    else:
        st.info(f"**{menor_mejora['Dimensión']}**: +{menor_mejora['Cambio']:.3f} puntos")
    
    st.subheader("📈 Porcentaje de Estudiantes que Mejoran")
    mejoras_por_dim = {}
    for dim in dims:
        pct = (df_paired[f'{dim}_post'] > df_paired[f'{dim}_pre']).sum() / len(df_paired) * 100
        mejoras_por_dim[dim] = round(pct, 1)
    
    df_mejoras = pd.DataFrame(list(mejoras_por_dim.items()), columns=['Dimensión', '% Mejoran'])
    df_mejoras = df_mejoras.sort_values('% Mejoran', ascending=False)
    
    st.dataframe(df_mejoras, use_container_width=True)
    
    # Gráfico resumen
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    bars = ax.barh(df_mejoras['Dimensión'], df_mejoras['% Mejoran'], 
                   color=COLOR_MEJORA, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('% de Estudiantes que Mejoran', fontweight='bold')
    ax.set_title('Porcentaje de Mejora por Dimensión', fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    # Añadir valores en las barras
    for bar in bars:
        width_val = bar.get_width()
        ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                f'{width_val:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Análisis de Metodología de Aprendizaje Cooperativo** | Desarrollado con Streamlit")
