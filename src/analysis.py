import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos(filepath="../data/respuestas_encuesta.csv"):
    """
    Carga los datos desde el archivo CSV
    """
    return pd.read_csv(filepath)

def calcular_promedios_por_pregunta(df):
    """
    Calcula los promedios por pregunta para PRE y POST tests
    """
    columnas_preguntas = [col for col in df.columns if col.startswith('P')]
    
    promedios = df.groupby('tipo_test')[columnas_preguntas].mean()
    diferencias = promedios.loc['POST'] - promedios.loc['PRE']
    
    return promedios, diferencias

def analizar_significancia(df):
    """
    Realiza pruebas t para cada pregunta comparando PRE y POST
    """
    resultados = {}
    columnas_preguntas = [col for col in df.columns if col.startswith('P')]
    
    for pregunta in columnas_preguntas:
        pre = df[df['tipo_test'] == 'PRE'][pregunta]
        post = df[df['tipo_test'] == 'POST'][pregunta]
        
        t_stat, p_value = stats.ttest_rel(pre, post)
        resultados[pregunta] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significativo': p_value < 0.05
        }
    
    return pd.DataFrame(resultados).T

def analizar_clima_motivacion(df):
    """
    Analiza las preguntas adicionales de clima y motivación
    """
    estudiantes_extra = df[df['clima'].notna()]
    
    resultados = {
        'PRE': {
            'clima_promedio': estudiantes_extra[estudiantes_extra['tipo_test'] == 'PRE']['clima'].mean(),
            'motivacion_promedio': estudiantes_extra[estudiantes_extra['tipo_test'] == 'PRE']['motivacion'].mean()
        },
        'POST': {
            'clima_promedio': estudiantes_extra[estudiantes_extra['tipo_test'] == 'POST']['clima'].mean(),
            'motivacion_promedio': estudiantes_extra[estudiantes_extra['tipo_test'] == 'POST']['motivacion'].mean()
        }
    }
    
    return pd.DataFrame(resultados)

def generar_graficos(df, output_dir="../data/"):
    """
    Genera gráficos de análisis
    """
    # Promedios por pregunta
    plt.figure(figsize=(15, 6))
    promedios, _ = calcular_promedios_por_pregunta(df)
    promedios.plot(kind='bar')
    plt.title('Promedios por Pregunta: PRE vs POST')
    plt.xlabel('Tipo de Test')
    plt.ylabel('Promedio')
    plt.tight_layout()
    plt.savefig(f"{output_dir}promedios_preguntas.png")
    plt.close()
    
    # Distribución de respuestas
    columnas_preguntas = [col for col in df.columns if col.startswith('P')]
    for pregunta in columnas_preguntas:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='tipo_test', y=pregunta, data=df)
        plt.title(f'Distribución de Respuestas - {pregunta}')
        plt.savefig(f"{output_dir}distribucion_{pregunta}.png")
        plt.close()

if __name__ == "__main__":
    # Cargar datos
    df = cargar_datos()
    
    # Realizar análisis
    promedios, diferencias = calcular_promedios_por_pregunta(df)
    significancia = analizar_significancia(df)
    clima_motivacion = analizar_clima_motivacion(df)
    
    # Generar gráficos
    generar_graficos(df)
    
    # Guardar resultados
    promedios.to_csv("../data/resultados_promedios.csv")
    significancia.to_csv("../data/resultados_significancia.csv")
    clima_motivacion.to_csv("../data/resultados_clima_motivacion.csv")