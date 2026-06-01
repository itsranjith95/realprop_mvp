from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

TRAIN_PATH = Path("data/labeled/classification/train.csv")
VAL_PATH = Path("data/labeled/classification/val.csv")
MODEL_DIR = Path("models/classification")
MODEL_PATH = MODEL_DIR / "tfidf_logreg.joblib"

REQUIRED_COLUMNS = {"text", "label"}


def load_split(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    return df["text"].tolist(), df["label"].tolist()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(TRAIN_PATH)
    X_val, y_val = load_split(VAL_PATH)

    mlflow.set_experiment("doctype_classifier_mvp")

    with mlflow.start_run(run_name="tfidf_logreg_motherdeed_khata_other"):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
        )

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        model.fit(X_train_vec, y_train)

        X_val_vec = vectorizer.transform(X_val)
        val_preds = model.predict(X_val_vec)

        val_acc = accuracy_score(y_val, val_preds)
        report = classification_report(y_val, val_preds, output_dict=True)

        bundle = {
            "vectorizer": vectorizer,
            "model": model,
            "labels": list(model.classes_),
        }

        joblib.dump(bundle, MODEL_PATH)

        mlflow.log_param("model_type", "tfidf_logreg")
        mlflow.log_param("ngram_range", "1,2")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("val_accuracy", val_acc)

        for label in model.classes_:
            if label in report:
                mlflow.log_metric(f"{label}_f1", report[label]["f1-score"])

        mlflow.log_artifact(str(MODEL_PATH))

        print(f"Saved model to {MODEL_PATH}")
        print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()