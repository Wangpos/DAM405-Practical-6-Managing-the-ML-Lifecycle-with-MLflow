"""Task 6: load the latest registered model from the Model Registry and predict."""
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

MODEL_URI = "models:/wine-classifier/latest"

if __name__ == "__main__":
    # Recreate the same test split used during training.
    X, y = load_wine(return_X_y=True)
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load by registry name, not run ID - callers don't need to know which run won.
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("Loaded model prediction:", model.predict(Xte[:5]))
    print("Actual labels:          ", yte[:5])
