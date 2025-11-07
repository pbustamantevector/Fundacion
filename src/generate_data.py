import pandas as pd
import numpy as np
from datetime import datetime
import json

# Constantes
N_ESTUDIANTES = 200
N_PREGUNTAS_BASE = 20  # Número de preguntas en el cuestionario base
N_PREGUNTAS_EXTRA = 2  # Preguntas adicionales para el 20% de estudiantes
PORCENTAJE_ESTUDIANTES_EXTRA = 0.20

def generar_respuestas_base(n_estudiantes, es_post=False):
    """
    Genera respuestas base para el cuestionario (1-5)
    Si es post test, se añade una pequeña mejora en las respuestas
    """
    base = np.random.normal(3.5, 0.8, (n_estudiantes, N_PREGUNTAS_BASE))
    if es_post:
        # Añadir una mejora promedio de 0.3 puntos en el post test
        mejora = np.random.normal(0.3, 0.2, (n_estudiantes, N_PREGUNTAS_BASE))
        base = base + mejora
    
    # Asegurar que los valores estén entre 1 y 5
    base = np.clip(base, 1, 5)
    return np.round(base, 2)

def generar_respuestas_extra():
    """
    Genera respuestas para las preguntas adicionales de clima y motivación (1-5)
    """
    return np.round(np.random.normal(4, 0.5, (2)), 2)

def crear_dataset():
    # Generar IDs de estudiantes
    estudiantes = [f"EST_{str(i+1).zfill(3)}" for i in range(N_ESTUDIANTES)]
    
    # Determinar qué estudiantes tendrán preguntas extra
    n_estudiantes_extra = int(N_ESTUDIANTES * PORCENTAJE_ESTUDIANTES_EXTRA)
    estudiantes_con_extra = np.random.choice(estudiantes, n_estudiantes_extra, replace=False)
    
    # Generar datos PRE test
    data_pre = []
    for estudiante in estudiantes:
        respuestas = generar_respuestas_base(1)[0]
        row = {
            'estudiante_id': estudiante,
            'tipo_test': 'PRE',
            'fecha': '2025-09-01'
        }
        # Añadir respuestas base
        for i, resp in enumerate(respuestas, 1):
            row[f'P{i}'] = resp
            
        # Añadir respuestas extra si corresponde
        if estudiante in estudiantes_con_extra:
            respuestas_extra = generar_respuestas_extra()
            row['clima'] = respuestas_extra[0]
            row['motivacion'] = respuestas_extra[1]
            
        data_pre.append(row)
    
    # Generar datos POST test
    data_post = []
    for estudiante in estudiantes:
        respuestas = generar_respuestas_base(1, es_post=True)[0]
        row = {
            'estudiante_id': estudiante,
            'tipo_test': 'POST',
            'fecha': '2025-11-30'
        }
        # Añadir respuestas base
        for i, resp in enumerate(respuestas, 1):
            row[f'P{i}'] = resp
            
        # Añadir respuestas extra si corresponde
        if estudiante in estudiantes_con_extra:
            respuestas_extra = generar_respuestas_extra()
            row['clima'] = respuestas_extra[0]
            row['motivacion'] = respuestas_extra[1]
            
        data_post.append(row)
    
    # Combinar datos PRE y POST
    all_data = pd.DataFrame(data_pre + data_post)
    return all_data

if __name__ == "__main__":
    # Generar datos
    df = crear_dataset()
    
    # Guardar datos en CSV
    output_path = "../data/respuestas_encuesta.csv"
    df.to_csv(output_path, index=False)
    print(f"Datos generados y guardados en: {output_path}")