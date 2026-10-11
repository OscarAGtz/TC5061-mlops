# Experimentación con MLflow – Telco Customer Churn

Script de entrenamiento (`train.py`) de una regresión logística para predecir `Churn`. Cada ejecución queda registrada en MLflow (experimento `telco-churn-logreg-hiperparametros`). La exploración de datos sigue en el notebook de la tarea 01.

## Instalación

Python 3.13.

```bash
pip install -r requirements.txt
```

## Ejecución

Desde esta carpeta:

```bash
python train.py                                   # valores por defecto
python train.py --C 0.1 --l1-ratio 0.5 --class-weight balanced
python train.py --help                            # ver todos los argumentos
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la regularización (menor = más fuerte) |
| `--l1-ratio` | 0.0 | Mezcla de regularización: 0 = L2, 1 = L1 |
| `--class-weight` | none | `balanced` da más peso a la clase que abandona |
| `--seed` | 42 | Seed del split y del modelo |
| `--data` | data/Telco-Customer-Churn.csv | Ruta del CSV |

## Ver resultados

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Reproducibilidad

La seed se pasa al `train_test_split` y al modelo, así que dos ejecuciones con los mismos argumentos dan las mismas métricas. Las versiones de las librerías están fijas en `requirements.txt`.
