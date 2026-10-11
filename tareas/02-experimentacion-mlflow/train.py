"""Entrena una regresión logística para predecir Churn y registra cada run en MLflow."""
import argparse

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--C", type=float, default=1.0, help="Inverso de la regularización (menor = más fuerte)")
    p.add_argument("--l1-ratio", type=float, default=0.0, help="Mezcla de regularización: 0 = L2, 1 = L1")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="'balanced' da más peso a la clase minoritaria (abandona)")
    p.add_argument("--seed", type=int, default=42, help="Seed para el split y el modelo")
    p.add_argument("--data", default="data/Telco-Customer-Churn.csv", help="Ruta del CSV")
    return p.parse_args()


def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)  # vacíos = clientes nuevos
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df.drop(columns=["customerID", "Churn"]), df["Churn"]


def construir_modelo(X, args):
    """Pipeline: escala numéricas, one-hot a categóricas y regresión logística."""
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols),
    ])
    clf = LogisticRegression(C=args.C, l1_ratio=args.l1_ratio, solver="saga", max_iter=5000,
                             class_weight=None if args.class_weight == "none" else "balanced",
                             random_state=args.seed)
    return Pipeline([("prep", prep), ("clf", clf)])


def evaluar(modelo, X, y):
    pred = modelo.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, modelo.predict_proba(X)[:, 1]),
    }


def guardar_matriz(modelo, X, y, ruta="confusion_matrix.png"):
    disp = ConfusionMatrixDisplay.from_estimator(modelo, X, y, display_labels=["Se queda", "Abandona"])
    disp.figure_.savefig(ruta, bbox_inches="tight")
    plt.close(disp.figure_)
    return ruta


def main():
    args = parse_args()
    X, y = cargar_datos(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("telco-churn-logreg-hiperparametros")
    with mlflow.start_run(run_name=f"C={args.C}_l1={args.l1_ratio}_{args.class_weight}"):
        mlflow.log_params({"C": args.C, "l1_ratio": args.l1_ratio,
                           "class_weight": args.class_weight, "seed": args.seed})

        modelo = construir_modelo(X_train, args).fit(X_train, y_train)
        metricas = evaluar(modelo, X_test, y_test)
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(guardar_matriz(modelo, X_test, y_test))
        mlflow.sklearn.log_model(modelo, name="modelo",
                                 signature=infer_signature(X_train, modelo.predict(X_train)))
    print({k: round(v, 4) for k, v in metricas.items()})


if __name__ == "__main__":
    main()
