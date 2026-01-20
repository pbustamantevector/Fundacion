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
    page_title="Análisis Completo - Metodología Aprendizaje Cooperativo",
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
COLOR_PRE = '#3498db'
COLOR_POST = '#e74c3c'
COLOR_MEJORA = '#27ae60'
COLOR_DISMINUYE = '#e67e22'
COLOR_NEUTRAL = '#95a5a6'

# Título principal
st.title("📈 Análisis Completo de Metodología de Aprendizaje Cooperativo")

# Navegación por pestañas
tab1, tab2, tab3 = st.tabs(["📊 F1: Análisis Estudiantes", "👨‍🏫 F2: Aplicaciones Docentes", "👁️ F3: Observación"])

# ==================== TAB 1: F1 - ANÁLISIS ESTUDIANTES ====================
with tab1:
    st.header("Análisis Temporal - Encuesta Estudiantes")
    
    # Botón para actualizar datos
    col_titulo, col_boton = st.columns([4, 1])
    with col_boton:
        if st.button("🔄 Actualizar Datos F1", use_container_width=True):
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

    # Cargar datos F1
    @st.cache_data
    def cargar_datos_f1():
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
        df['Fecha'] = pd.to_datetime(df['Marca temporal'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df['Mes_Año'] = df['Fecha'].dt.to_period('M')
        df['Mes'] = df['Fecha'].dt.month
        df['Año'] = df['Fecha'].dt.year
        df['Colegio_Nivel_Paralelo'] = df['1.  Selecciona tu colegio:'] + '_' + df['2. ¿En qué nivel estás?'] + '_' + df['3. Paralelo']
        df['Colegio_Nivel'] = df['1.  Selecciona tu colegio:'] + '_' + df['2. ¿En qué nivel estás?']
        
        columnas_metadatos = ['1.  Selecciona tu colegio:', '2. ¿En qué nivel estás?', '3. Paralelo']
        preguntas_cols = [col for col in df.columns if any(col.startswith(f'{i}.') for i in range(1, 23)) and col not in columnas_metadatos]
        
        for col in preguntas_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Puntaje_Promedio'] = df[preguntas_cols].mean(axis=1)
        
        for dim, preguntas in dimensiones.items():
            cols_dim = [col for col in preguntas_cols if any(col.startswith(f'{p}.') for p in preguntas)]
            if cols_dim:
                df[f'Puntaje_{dim}'] = df[cols_dim].mean(axis=1)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        return df, dimensiones, preguntas_cols, timestamp

    try:
        df, dimensiones, preguntas_cols, timestamp = cargar_datos_f1()
        dims = list(dimensiones.keys())
        colegios = sorted(df['1.  Selecciona tu colegio:'].unique())
        niveles = sorted(df['2. ¿En qué nivel estás?'].unique())
        paralelos = sorted(df['3. Paralelo'].unique())
        
        st.success(f"✅ Datos cargados: {df.shape[0]} registros | Última actualización: {timestamp}")
    except Exception as e:
        st.error(f"❌ Error al cargar datos F1: {e}")
        st.stop()

    # ==================== FILTROS INTERACTIVOS ====================
    st.header("🔍 Filtros Interactivos")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        colegios_sel = st.multiselect(
            "Colegios:",
            options=colegios,
            default=[],
            placeholder="Elegir opciones",
            key="colegios_f1"
        )

    with col2:
        niveles_sel = st.multiselect(
            "Niveles:",
            options=niveles,
            default=[],
            placeholder="Elegir opciones",
            key="niveles_f1"
        )

    with col3:
        paralelos_sel = st.multiselect(
            "Paralelos:",
            options=paralelos,
            default=[],
            placeholder="Elegir opciones",
            key="paralelos_f1"
        )
    
    with col4:
        dimensiones_sel = st.multiselect(
            "Dimensiones:",
            options=dims,
            default=[],
            placeholder="Elegir opciones",
            key="dimensiones_f1",
            help="Deja vacío para mostrar promedio general de las 7 dimensiones"
        )

    # Aplicar filtros
    df_filtrado = df.copy()

    if len(colegios_sel) > 0:
        df_filtrado = df_filtrado[df_filtrado['1.  Selecciona tu colegio:'].isin(colegios_sel)]
    if len(niveles_sel) > 0:
        df_filtrado = df_filtrado[df_filtrado['2. ¿En qué nivel estás?'].isin(niveles_sel)]
    if len(paralelos_sel) > 0:
        df_filtrado = df_filtrado[df_filtrado['3. Paralelo'].isin(paralelos_sel)]
    
    # Determinar columnas de puntaje según dimensiones seleccionadas
    if len(dimensiones_sel) > 0:
        columnas_puntaje = [f'Puntaje_{dim}' for dim in dimensiones_sel]
        label_puntaje = f"Puntaje ({', '.join(dimensiones_sel)})"
        df_filtrado['Puntaje_Filtrado'] = df_filtrado[columnas_puntaje].mean(axis=1)
        st.info(f"📊 Dimensiones activas: {', '.join(dimensiones_sel)}")
    else:
        df_filtrado['Puntaje_Filtrado'] = df_filtrado['Puntaje_Promedio']
        label_puntaje = "Puntaje Promedio General"
        st.info(f"📊 Mostrando: Promedio general de las 7 dimensiones")

    if len(df_filtrado) == 0:
        st.warning("⚠️ No hay datos con los filtros seleccionados")
    else:
        st.markdown("---")
        
        # ==================== GRÁFICOS INTERACTIVOS ====================
        st.header("📈 Gráficos Interactivos")
        
        # Agrupar datos por curso y nivel
        agrupado_curso = df_filtrado.groupby(['Colegio_Nivel_Paralelo', 'Mes_Año'])['Puntaje_Filtrado'].agg(['mean', 'std', 'count']).reset_index()
        agrupado_curso.columns = ['Curso', 'Mes_Año', 'Promedio', 'Desv_Std', 'N_Estudiantes']
        agrupado_curso['Fecha_Plot'] = agrupado_curso['Mes_Año'].dt.to_timestamp()
        
        agrupado_nivel = df_filtrado.groupby(['Colegio_Nivel', 'Mes_Año'])['Puntaje_Filtrado'].agg(['mean', 'std', 'count']).reset_index()
        agrupado_nivel.columns = ['Nivel', 'Mes_Año', 'Promedio', 'Desv_Std', 'N_Estudiantes']
        agrupado_nivel['Fecha_Plot'] = agrupado_nivel['Mes_Año'].dt.to_timestamp()
        
        # Usar todos los cursos y niveles disponibles
        top_cursos = agrupado_curso['Curso'].unique()
        top_niveles = agrupado_nivel['Nivel'].unique()
        
        # Crear figura con 4 subgráficos
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Análisis Temporal de Puntajes por Curso y Nivel', fontsize=16, fontweight='bold', y=0.995)
        
        # GRÁFICO 1: Evolución del puntaje promedio por curso (Barras Agrupadas)
        pivot_curso = agrupado_curso[agrupado_curso['Curso'].isin(top_cursos)].pivot(
            index='Fecha_Plot', columns='Curso', values='Promedio'
        )
        
        if not pivot_curso.empty:
            pivot_curso.plot(kind='bar', ax=axes[0, 0], alpha=0.85, width=0.8, edgecolor='white', linewidth=0.5)
            axes[0, 0].set_xlabel('Mes', fontweight='bold')
            axes[0, 0].set_ylabel(label_puntaje, fontweight='bold')
            axes[0, 0].set_title(f'Evolución del Puntaje por Curso (Top {len(top_cursos)})', fontweight='bold')
            axes[0, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, title='Cursos')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylim(1, 5)
            # Formatear fechas sin hora
            labels = [label.get_text()[:10] if len(label.get_text()) > 10 else label.get_text() for label in axes[0, 0].get_xticklabels()]
            axes[0, 0].set_xticklabels(labels)
        
        # GRÁFICO 2: Desviación estándar por curso
        pivot_std_curso = agrupado_curso[agrupado_curso['Curso'].isin(top_cursos)].pivot(
            index='Fecha_Plot', columns='Curso', values='Desv_Std'
        )
        
        if not pivot_std_curso.empty:
            pivot_std_curso.plot(kind='bar', ax=axes[0, 1], alpha=0.85, width=0.8, edgecolor='white', linewidth=0.5)
            axes[0, 1].set_xlabel('Mes', fontweight='bold')
            axes[0, 1].set_ylabel('Desviación Estándar', fontweight='bold')
            axes[0, 1].set_title('Variabilidad por Curso', fontweight='bold')
            axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, title='Cursos')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            axes[0, 1].tick_params(axis='x', rotation=45)
            # Formatear fechas sin hora
            labels = [label.get_text()[:10] if len(label.get_text()) > 10 else label.get_text() for label in axes[0, 1].get_xticklabels()]
            axes[0, 1].set_xticklabels(labels)
        
        # GRÁFICO 3: Evolución del puntaje promedio por nivel
        pivot_nivel = agrupado_nivel[agrupado_nivel['Nivel'].isin(top_niveles)].pivot(
            index='Fecha_Plot', columns='Nivel', values='Promedio'
        )
        
        if not pivot_nivel.empty:
            pivot_nivel.plot(kind='bar', ax=axes[1, 0], alpha=0.85, width=0.8, edgecolor='white', linewidth=0.5)
            axes[1, 0].set_xlabel('Mes', fontweight='bold')
            axes[1, 0].set_ylabel(label_puntaje, fontweight='bold')
            axes[1, 0].set_title(f'Evolución del Puntaje por Nivel (Top {len(top_niveles)})', fontweight='bold')
            axes[1, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, title='Niveles')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylim(1, 5)
            # Formatear fechas sin hora
            labels = [label.get_text()[:10] if len(label.get_text()) > 10 else label.get_text() for label in axes[1, 0].get_xticklabels()]
            axes[1, 0].set_xticklabels(labels)
        
        # GRÁFICO 4: Desviación estándar por nivel
        pivot_std_nivel = agrupado_nivel[agrupado_nivel['Nivel'].isin(top_niveles)].pivot(
            index='Fecha_Plot', columns='Nivel', values='Desv_Std'
        )
        
        if not pivot_std_nivel.empty:
            pivot_std_nivel.plot(kind='bar', ax=axes[1, 1], alpha=0.85, width=0.8, edgecolor='white', linewidth=0.5)
            axes[1, 1].set_xlabel('Mes', fontweight='bold')
            axes[1, 1].set_ylabel('Desviación Estándar', fontweight='bold')
            axes[1, 1].set_title('Variabilidad por Nivel', fontweight='bold')
            axes[1, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, title='Niveles')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].tick_params(axis='x', rotation=45)
            # Formatear fechas sin hora
            labels = [label.get_text()[:10] if len(label.get_text()) > 10 else label.get_text() for label in axes[1, 1].get_xticklabels()]
            axes[1, 1].set_xticklabels(labels)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Mostrar estadísticas
        st.success(f"✓ Registros mostrados: {len(df_filtrado):,} | Cursos graficados: {len(top_cursos)} | Niveles graficados: {len(top_niveles)}")


# ==================== TAB 2: F2 - APLICACIONES DOCENTES ====================
with tab2:
    st.header("Aplicaciones de Metodología por Docentes")
    
    # Botón para actualizar datos
    col_titulo, col_boton = st.columns([4, 1])
    with col_boton:
        if st.button("🔄 Actualizar Datos F2", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("""
    Análisis de aplicaciones de metodologías de aprendizaje cooperativo por docente, curso, nivel y colegio.
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
        
        df_f2['Semana_Aplicacion'] = pd.to_numeric(df_f2['Semana_Aplicacion'], errors='coerce')
        df_f2['Curso'] = df_f2['Colegio'] + '_' + df_f2['Nivel'] + '_' + df_f2['Paralelo']
        df_f2['Colegio_Nivel'] = df_f2['Colegio'] + '_' + df_f2['Nivel']
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        return df_f2, timestamp
    
    try:
        df_f2, timestamp = cargar_datos_f2()
        st.success(f"✅ Datos F2 cargados: {df_f2.shape[0]} registros | Última actualización: {timestamp}")
        
        # Mostrar información general
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Docentes Únicos", df_f2['Nombre_Docente'].nunique())
        with col2:
            st.metric("Colegios", df_f2['Colegio'].nunique())
        with col3:
            st.metric("Cursos Únicos", df_f2['Curso'].nunique())
        with col4:
            st.metric("Rango Semanas", f"{df_f2['Semana_Aplicacion'].min():.0f} - {df_f2['Semana_Aplicacion'].max():.0f}")
        
        st.markdown("---")
        
        # ==================== TABLA 1: APLICACIONES POR DOCENTE ====================
        st.subheader("📊 Tabla 1: Aplicaciones por Docente")
        
        aplicaciones_por_docente = df_f2.groupby('Nombre_Docente').agg({
            'Semana_Aplicacion': 'max',
            'Colegio': lambda x: ', '.join(x.unique()),
            'Curso': 'nunique',
            'Metodologia': 'nunique'
        }).reset_index()
        
        aplicaciones_por_docente.columns = ['Docente', 'Total_Aplicaciones', 'Colegios', 'Cursos_Diferentes', 'Metodologias_Diferentes']
        aplicaciones_por_docente = aplicaciones_por_docente.sort_values('Total_Aplicaciones', ascending=False)
        
        st.write(f"**Total de docentes:** {len(aplicaciones_por_docente)}")
        st.write(f"**Promedio de aplicaciones por docente:** {aplicaciones_por_docente['Total_Aplicaciones'].mean():.1f}")
        st.write(f"**Máximo de aplicaciones:** {aplicaciones_por_docente['Total_Aplicaciones'].max():.0f}")
        st.write(f"**Mínimo de aplicaciones:** {aplicaciones_por_docente['Total_Aplicaciones'].min():.0f}")
        
        st.dataframe(aplicaciones_por_docente, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # ==================== TABLA 2: APLICACIONES POR CURSO ====================
        st.subheader("📊 Tabla 2: Aplicaciones por Curso")
        
        aplicaciones_por_curso = df_f2.groupby(['Colegio', 'Nivel', 'Paralelo']).agg({
            'Nombre_Docente': 'nunique',
            'Semana_Aplicacion': 'max',
            'Metodologia': 'nunique'
        }).reset_index()
        
        aplicaciones_por_curso.columns = ['Colegio', 'Nivel', 'Paralelo', 'Docentes_Diferentes', 'Total_Aplicaciones', 'Metodologias_Diferentes']
        aplicaciones_por_curso = aplicaciones_por_curso.sort_values('Total_Aplicaciones', ascending=False)
        
        st.write(f"**Total de cursos:** {len(aplicaciones_por_curso)}")
        st.write(f"**Promedio de aplicaciones por curso:** {aplicaciones_por_curso['Total_Aplicaciones'].mean():.1f}")
        st.write(f"**Curso con más aplicaciones:** {aplicaciones_por_curso['Total_Aplicaciones'].max():.0f}")
        
        st.dataframe(aplicaciones_por_curso, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # ==================== TABLA 3: APLICACIONES POR NIVEL ====================
        st.subheader("📊 Tabla 3: Aplicaciones por Nivel")
        
        aplicaciones_por_nivel = df_f2.groupby(['Colegio', 'Nivel']).agg({
            'Nombre_Docente': 'nunique',
            'Semana_Aplicacion': 'max',
            'Paralelo': 'nunique',
            'Metodologia': 'nunique'
        }).reset_index()
        
        aplicaciones_por_nivel.columns = ['Colegio', 'Nivel', 'Docentes_Diferentes', 'Total_Aplicaciones', 'Paralelos_Diferentes', 'Metodologias_Diferentes']
        aplicaciones_por_nivel = aplicaciones_por_nivel.sort_values('Total_Aplicaciones', ascending=False)
        
        st.write(f"**Total de niveles:** {len(aplicaciones_por_nivel)}")
        st.write(f"**Promedio de aplicaciones por nivel:** {aplicaciones_por_nivel['Total_Aplicaciones'].mean():.1f}")
        st.write(f"**Nivel con más aplicaciones:** {aplicaciones_por_nivel['Total_Aplicaciones'].max():.0f}")
        
        st.dataframe(aplicaciones_por_nivel, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # ==================== TABLA 4: APLICACIONES POR COLEGIO ====================
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
        
        st.write(f"**Total de colegios:** {len(aplicaciones_por_colegio)}")
        st.write(f"**Promedio de aplicaciones por colegio:** {aplicaciones_por_colegio['Total_Aplicaciones'].mean():.1f}")
        st.write(f"**Colegio con más aplicaciones:** {aplicaciones_por_colegio['Total_Aplicaciones'].max():.0f}")
        
        st.dataframe(aplicaciones_por_colegio, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # ==================== TABLA 5: DETALLE DOCENTE X CURSO ====================
        st.subheader("📊 Tabla 5: Detalle Docente x Curso")
        
        detalle_docente_curso = df_f2.groupby(['Nombre_Docente', 'Colegio', 'Nivel', 'Paralelo']).agg({
            'Semana_Aplicacion': 'max',
            'Metodologia': lambda x: ', '.join(x.unique())
        }).reset_index()
        
        detalle_docente_curso.columns = ['Docente', 'Colegio', 'Nivel', 'Paralelo', 'Total_Aplicaciones', 'Metodologias_Aplicadas']
        detalle_docente_curso = detalle_docente_curso.sort_values(['Docente', 'Total_Aplicaciones'], ascending=[True, False])
        
        st.write(f"**Total de combinaciones Docente-Curso:** {len(detalle_docente_curso)}")
        
        st.dataframe(detalle_docente_curso, use_container_width=True, height=400)
        
    except Exception as e:
        st.error(f"❌ Error al cargar datos F2: {e}")


# ==================== TAB 3: F3 - OBSERVACIÓN ====================
with tab3:
    st.header("Observación - Análisis en Tabla")
    
    # Botón para actualizar datos
    col_titulo, col_boton = st.columns([4, 1])
    with col_boton:
        if st.button("🔄 Actualizar Datos F3", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("""
    Datos de observación del proceso de aplicación de metodologías.
    """)
    
    # Cargar datos F3
    @st.cache_data
    def cargar_datos_f3():
        sheet_id = '12JoLMA_A_-MuLqxbTTEsmBPVBhNSNbllyAlaThO2HDc'
        gid = '672136638'
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
        
        df_f3 = pd.read_csv(url)
        df_f3.columns = df_f3.columns.str.strip()
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        return df_f3, timestamp
    
    try:
        df_f3, timestamp = cargar_datos_f3()
        st.success(f"✅ Datos F3 cargados: {df_f3.shape[0]} registros, {df_f3.shape[1]} columnas | Última actualización: {timestamp}")
        
        # Mostrar información de columnas
        with st.expander("📋 Ver columnas disponibles"):
            st.write("**Columnas en el dataset:**")
            for i, col in enumerate(df_f3.columns, 1):
                st.write(f"{i}. {col}")
        
        st.markdown("---")
        
        # Mostrar tabla completa
        st.subheader("📊 Datos de Observación F3")
        st.dataframe(df_f3, use_container_width=True, height=600)
        
        # Mostrar estadísticas básicas
        if len(df_f3) > 0:
            st.markdown("---")
            st.subheader("📈 Estadísticas Básicas")
            
            # Intentar identificar columnas numéricas
            numeric_cols = df_f3.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                st.write("**Columnas numéricas:**")
                st.dataframe(df_f3[numeric_cols].describe(), use_container_width=True)
            
            # Mostrar información de columnas categóricas
            categorical_cols = df_f3.select_dtypes(include=['object']).columns.tolist()
            
            if len(categorical_cols) > 0:
                st.write("**Valores únicos por columna categórica:**")
                unique_counts = {col: df_f3[col].nunique() for col in categorical_cols}
                st.write(unique_counts)
        
    except Exception as e:
        st.error(f"❌ Error al cargar datos F3: {e}")


# Footer
st.markdown("---")
st.markdown("**Análisis Completo de Metodología de Aprendizaje Cooperativo** | Desarrollado con Streamlit")
