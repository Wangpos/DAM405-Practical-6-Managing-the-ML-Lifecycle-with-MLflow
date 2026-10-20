"""Tasks 2 & 3: track RandomForest training runs on the Wine dataset with MLflow.

Each call to run() trains one model and logs its hyperparameters, accuracy
and the fitted model artifact to the 'wine-classifier' experiment.
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "wine-classifier"

# Fixed split (random_state=42) so every run is evaluated on the same test set.
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)


def run(n_estimators, max_depth):
    """Train one model inside an MLflow run and log params, metric and model."""
    with mlflow.start_run():
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        ).fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))

        # Parameters = inputs we chose; metrics = results we measured.
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        # Save the model itself as an artifact so it can be registered later.
        # MLflow 3.x serialises with skops, which refuses tree internals unless
        # trusted; safe here because we trained the model ourselves.
        mlflow.sklearn.log_model(
            clf,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"n={n_estimators} depth={max_depth} acc={acc:.4f}")


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Task 3: grid of 3 x 3 = 9 runs to compare in the UI.
    for n in (50, 100, 200):
        for d in (3, 5, None):
            run(n, d)
