




import joblib

from src.config import MODEL_PATH, THRESHOLD


def load_model(path=MODEL_PATH):
    """Load the saved pipeline."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model not found: {path}\nTrain and save the model first."
        )


def predict_email(model, text, threshold=THRESHOLD):
    """Return ('spam' or 'ham', probability of spam) for one email."""
    probability = float(model.predict_proba([text])[0, 1])
    label = "spam" if probability >= threshold else "ham"
    return label, probability




