# Análisis de Metodología Cooperativa

Este proyecto analiza los beneficios del uso de metodología cooperativa en intervenciones educativas, basado en el trabajo de Fernandez-Río (2017).

## Estructura del Proyecto

```
├── data/                  # Directorio para almacenar datos y resultados
├── src/                   # Código fuente
│   ├── generate_data.py   # Genera datos simulados de encuestas
│   ├── analysis.py        # Scripts de análisis de datos
│   └── app.py            # Dashboard Streamlit
├── requirements.txt       # Dependencias del proyecto
└── README.md             # Este archivo
```

## Características

- Simulación de 200 respuestas de estudiantes
- Análisis PRE y POST intervención
- Análisis especial para subgrupo (20%) con preguntas adicionales de clima y motivación
- Dashboard interactivo con Streamlit
- Análisis estadístico y visualización de resultados

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL-DEL-REPOSITORIO]
cd [NOMBRE-DEL-REPOSITORIO]
```

2. Crear y activar un entorno virtual:
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Unix o MacOS:
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Generar datos simulados:
```bash
python src/generate_data.py
```

2. Ejecutar análisis:
```bash
python src/analysis.py
```

3. Iniciar el dashboard:
```bash
streamlit run src/app.py
```

## Dashboard

El dashboard incluye:
- Visualización de promedios PRE vs POST
- Análisis detallado por pregunta
- Pruebas de significancia estadística
- Análisis especial de clima y motivación
- Gráficos interactivos

## Dependencias Principales

- pandas
- numpy
- scipy
- matplotlib
- seaborn
- streamlit
- plotly

## Licencia

[Especificar la licencia]