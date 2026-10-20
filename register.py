"""Task 5: find the best run by accuracy and register its model.

Instead of copying a run ID by hand, query the experiment for the run with
the highest accuracy and register that one as 'wine-classifier'.
"""
import mlflow

EXPERIMENT_NAME = "wine-classifier"
MODEL_NAME = "wine-classifier"

if __name__ == "__main__":
    # Sort descending by accuracy so the first row is the best run.
    best = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    if best.empty:
        raise SystemExit("No runs found - run train.py first.")

    run_id = best.loc[0, "run_id"]
    print(f"Best run: {run_id}")
    print(f"  n_estimators={best.loc[0, 'params.n_estimators']} "
          f"max_depth={best.loc[0, 'params.max_depth']} "
          f"accuracy={best.loc[0, 'metrics.accuracy']:.4f}")

    # Each call creates a new version under the same registered model name.
    result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    print("Registered version:", result.version)
