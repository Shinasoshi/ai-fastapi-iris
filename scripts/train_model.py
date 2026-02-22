from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_PATH = Path("models/iris_model.joblib")

def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    artifact = {
        "model": pipe,
        "target_names": iris.target_names.tolist(),
        "feature_names": iris.feature_names,
        "metrics": {"accuracy": float(acc)},
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
