# DAM405 Practical 6 – Managing the ML Lifecycle with MLflow

Trains a RandomForest classifier on the scikit-learn Wine dataset and takes it
through the MLflow lifecycle: **track → compare → register → load**.

| File | Task |
|------|------|
| `train.py` | Tasks 2–3: logs 9 runs (n_estimators ∈ {50,100,200} × max_depth ∈ {3,5,None}) |
| `register.py` | Task 5: finds the best run by accuracy and registers it as `wine-classifier` |
| `predict.py` | Task 6: loads `models:/wine-classifier/latest` and predicts |

## Setup (Task 1)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mlflow --version
```

## Run

```bash
python train.py      # 9 tracked runs
python register.py   # registers the best run's model
python predict.py    # loads registered model and predicts
```

## Compare runs in the UI (Task 4)

```bash
mlflow ui --port 5000
```

Open http://localhost:5000, select the **wine-classifier** experiment and sort
by the `accuracy` column. The registered model is under the **Models** tab.

> Run `mlflow ui` from this folder so it reads the same local tracking store
> (`mlruns/` or `mlflow.db`, depending on MLflow version) the scripts wrote to.

## Optional: serve over REST

```bash
mlflow models serve -m "models:/wine-classifier/latest" -p 5001 --env-manager local
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050]]}'
```
