import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow

ARTIFACT_DIR = Path("outputs")
ARTIFACT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("hello-mlflow")


def fake_training(learning_rate: float, epochs: int, seed: int = 42):
    rng = random.Random(seed)
    history = []
    for epoch in range(1, epochs + 1):
        loss = math.exp(-learning_rate * epoch) + 0.05 + rng.uniform(0.0, 0.02)
        accuracy = 1.0 - loss / 2
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
    return history


def main() -> None:
    params = {"learning_rate": 0.35, "epochs": 20, "optimizer": "adam", "seed": 42}

    with mlflow.start_run(run_name="hello-run") as run:
        for key, value in params.items():
            mlflow.log_param(key, value)

        history = fake_training(params["learning_rate"], params["epochs"], params["seed"])
        for record in history:
            mlflow.log_metric("loss", record["loss"], step=record["epoch"])
            mlflow.log_metric("accuracy", record["accuracy"], step=record["epoch"])

        mlflow.log_metric("final_loss", history[-1]["loss"])
        mlflow.log_metric("final_accuracy", history[-1]["accuracy"])

        csv_path = ARTIFACT_DIR / "training_history.csv"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("epoch,loss,accuracy\n")
            for record in history:
                fh.write(f"{record['epoch']},{record['loss']:.6f},{record['accuracy']:.6f}\n")
        mlflow.log_artifact(str(csv_path))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["epoch"] for r in history], [r["loss"] for r in history], marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Curva de pérdida")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        plot_path = ARTIFACT_DIR / "loss_curve.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        config_path = ARTIFACT_DIR / "config.json"
        config_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(config_path), artifact_path="config")

        print(f"run_id      : {run.info.run_id}")
        print(f"experiment  : {run.info.experiment_id}")
        print(f"final_loss  : {history[-1]['loss']:.4f}")
        print("Revisa la UI con:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
