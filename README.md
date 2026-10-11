# TC5061 – MLOps

Tareas del curso. Un solo entorno de Python (`uv`) compartido; cada tarea vive en su propia carpeta y es autocontenida.

## Estructura

```
mlops/
├── pyproject.toml / uv.lock     # dependencias compartidas
├── tareas/
│   ├── 01-baseline-telco-churn/
│   │   ├── TC5061_Notebook_Baseline_GutierrezOscar.ipynb   # entregable
│   │   ├── data/                 # datos de entrada (versionados)
│   │   ├── img/                  # capturas para la entrega
│   │   ├── outputs/              # figuras generadas        (ignorado por git)
│   │   ├── mlflow.db             # tracking de MLflow       (ignorado por git)
│   │   └── mlruns/               # artefactos de MLflow     (ignorado por git)
│   └── 02-experimentacion-mlflow/
│       ├── train.py              # script de entrenamiento con MLflow
│       ├── requirements.txt      # versiones de las librerías
│       ├── README.md             # cómo ejecutar
│       ├── data/                 # datos de entrada (versionados)
│       └── img/                  # capturas para la entrega
└── ejemplos/
    └── hello_mlflow.py           # prueba mínima de MLflow
```

Cada tarea tiene su propio `mlflow.db`: los experimentos de una no se mezclan con los de otra, y la carpeta se puede entregar o mover completa.

## Uso

```bash
uv sync                                         # instalar dependencias
cd tareas/01-baseline-telco-churn
uv run jupyter lab                              # abrir el notebook
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # ver experimentos
```

## Nueva tarea

1. Crear `tareas/NN-nombre-corto/` con `data/` e `img/`.
2. En el notebook, usar rutas relativas a esa carpeta (`data/...`, `sqlite:///mlflow.db`).
3. Si hace falta una librería nueva: `uv add <paquete>` desde la raíz.
