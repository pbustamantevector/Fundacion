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
        max-width: 1200px;
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
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
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
st.title("Análisis de Metodología de Aprendizaje Cooperativo")

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
    sheet_name = 'Respuestas%20de%20formulario%201'
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
    
    # IMPORTANTE: La respuesta "No" = Primera vez = PRE (baseline)
    #            La respuesta "Sí" = Segunda vez = POST (después de intervención)
    df_pre = df[df.iloc[:, 1] == 'No'].copy()
    df_post = df[df.iloc[:, 1] == 'Sí'].copy()
    
    return df, df_pre, df_post, dimensiones, columnas_preguntas

try:
    df, df_pre, df_post, dimensiones, columnas_preguntas = cargar_datos()
    dims = list(dimensiones.keys())
    colegio_col = df.columns[2]
    momento_col = df.columns[1]
    id_col = df.columns[0]
    nombre_col = '2. ¿Cuál es tu nombre?'
    colegios = sorted(df[colegio_col].unique())
    
    st.success(f"✅ Datos cargados: {df.shape[0]} registros, {len(df_pre)} PRE, {len(df_post)} POST")
    
except Exception as e:
    st.error(f"❌ Error al cargar datos: {e}")
    st.stop()

# ==================== RESUMEN DE DATOS ====================
st.header("Resumen de Datos")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Registros", df.shape[0])
with col2:
    st.metric("Registros PRE", len(df_pre))
with col3:
    st.metric("Registros POST", len(df_post))
with col4:
    st.metric("Colegios", df[colegio_col].nunique())

with st.expander("📋 Ver vista previa de datos"):
    st.dataframe(df.head(10))

st.markdown("---")

# ==================== ESTADÍSTICAS DESCRIPTIVAS ====================
st.header("Estadísticas Descriptivas por Dimensión")

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

st.subheader("Resumen General")
st.dataframe(stats_summary, use_container_width=True)

# Gráficos
st.subheader("Gráficos Comparativos")

tab1, tab2, tab3, tab4 = st.tabs(["Medias", "Medianas", "Cambio Media", "Cambio Mediana"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
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
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
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
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('white')
    colors = [COLOR_MEJORA if x > 0 else COLOR_DISMINUYE for x in stats_summary['Cambio Media']]
    bars = ax.barh(stats_summary['Dimensión'], stats_summary['Cambio Media'], 
                   color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Cambio en Media (POST - PRE)', fontweight='bold')
    ax.set_title('Cambio en Media por Dimensión (Total)', fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for bar in bars:
        width_val = bar.get_width()
        ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                f'{width_val:.3f}', ha='left' if width_val > 0 else 'right', 
                va='center', fontweight='bold', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('white')
    colors = [COLOR_MEJORA if x > 0 else COLOR_DISMINUYE for x in stats_summary['Cambio Mediana']]
    bars = ax.barh(stats_summary['Dimensión'], stats_summary['Cambio Mediana'], 
                   color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Cambio en Mediana (POST - PRE)', fontweight='bold')
    ax.set_title('Cambio en Mediana por Dimensión (Total)', fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for bar in bars:
        width_val = bar.get_width()
        ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                f'{width_val:.3f}', ha='left' if width_val > 0 else 'right', 
                va='center', fontweight='bold', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# ==================== CAMBIOS PRE VS POST ====================
st.header("Mejoría/Disminución por Dimensión")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
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
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('white')
    colors = [COLOR_MEJORA if x > 0 else COLOR_DISMINUYE for x in stats_summary['Cambio Media']]
    bars = ax.barh(stats_summary['Dimensión'], stats_summary['Cambio Media'], 
                   color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Cambio (POST - PRE)', fontweight='bold')
    ax.set_title('Mejoría/Disminución', fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for bar in bars:
        width_val = bar.get_width()
        ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                f'{width_val:.3f}', ha='left' if width_val > 0 else 'right', 
                va='center', fontweight='bold', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# ==================== CORRELACIONES ====================
st.header("Correlación entre Dimensiones")

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
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
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
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor('white')
    sns.heatmap(corr_post, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Correlación'}, 
                vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    ax.set_title('Correlaciones POST', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# ==================== ANÁLISIS POR ESTUDIANTE ====================
st.header("Estudiantes que Aumentan/Disminuyen por Dimensión")

df_paired = df_pre.merge(df_post, on=nombre_col, suffixes=('_pre', '_post'), how='inner')

cambios = []
for dim in dims:
    aumentan = (df_paired[f'{dim}_post'] > df_paired[f'{dim}_pre']).sum()
    disminuyen = (df_paired[f'{dim}_post'] < df_paired[f'{dim}_pre']).sum()
    igual = (df_paired[f'{dim}_post'] == df_paired[f'{dim}_pre']).sum()
    cambios.append({'Dimensión': dim, 'Aumentan': aumentan, 'Disminuyen': disminuyen, 'Igual': igual})

df_cambios = pd.DataFrame(cambios)

st.dataframe(df_cambios, use_container_width=True)

fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
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

st.markdown("---")

# ==================== DISTRIBUCIÓN POR COLEGIO ====================
st.header("Distribución Boxplot por Colegio")

dim_seleccionada = st.selectbox("Seleccionar dimensión:", dims)

fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

positions = []
labels = []
data_to_plot = []

for i, colegio in enumerate(colegios):
    pre_data = df[(df[colegio_col] == colegio) & (df[momento_col] == 'No')][dim_seleccionada].dropna()
    post_data = df[(df[colegio_col] == colegio) & (df[momento_col] == 'Sí')][dim_seleccionada].dropna()
    
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

st.markdown("---")

# ==================== RANKING DE MEJORAS ====================
st.header("Ranking de Dimensiones con Mayor Mejora")

ranking = stats_summary.sort_values('Cambio Media', ascending=False).reset_index(drop=True)
ranking['Ranking'] = range(1, len(ranking) + 1)
ranking = ranking[['Ranking', 'Dimensión', 'Media PRE', 'Media POST', 'Cambio Media']]

st.dataframe(ranking, use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
fig.patch.set_facecolor('white')
colors = [COLOR_MEJORA if x > 0 else COLOR_DISMINUYE for x in ranking['Cambio Media']]
bars = ax.barh(ranking['Dimensión'], ranking['Cambio Media'], 
               color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
ax.axvline(x=0, color='#34495e', linestyle='-', linewidth=2, alpha=0.7)
ax.set_xlabel('Cambio (POST - PRE)', fontweight='bold')
ax.set_title('Ranking de Mejoras por Dimensión', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
for bar in bars:
    width_val = bar.get_width()
    ax.text(width_val, bar.get_y() + bar.get_height()/2, 
            f'{width_val:.3f}', ha='left' if width_val > 0 else 'right', 
            va='center', fontweight='bold', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ==================== FRECUENCIAS DE RESPUESTA ====================
st.header("Frecuencias de Respuesta (Bajas 1-2 vs Altas 4-5)")

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

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

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

# Footer
st.markdown("---")
st.markdown("**Análisis de Metodología de Aprendizaje Cooperativo** | Desarrollado con Streamlit")
