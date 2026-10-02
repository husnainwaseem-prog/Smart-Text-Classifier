"""Load the saved model and classify new text."""
from functools import lru_cache

import joblib

from src.preprocessing import clean_text
from src.utils import MODEL_PATH, InvalidInputError, ModelNotFoundError


@lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    """Load the trained pipeline once and reuse it (cached)."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError as error:
        raise ModelNotFoundError(
            f"Model file not found: {model_path}. Train it first with:  python -m src.train"
        ) from error
    except Exception as error:  # corrupted file, version mismatch, etc.
        raise ModelNotFoundError(f"Could not load the model: {error}") from error


def predict_text(text, model=None) -> dict:
    """Classify one piece of text.

    Returns a dict with: category, confidence (0-1), probabilities (per class),
    and warning (None, or a short note for very short input).
    Raises InvalidInputError for input that cannot be classified.
    """
    if not isinstance(text, str):
        raise InvalidInputError("Input must be text (a string).")
    cleaned = clean_text(text)
    if not cleaned:
        raise InvalidInputError("Please enter some text containing letters or numbers.")

    model = model if model is not None else load_model()
    probabilities = model.predict_proba([text])[0]
    best = int(probabilities.argmax())

    warning = None
    if len(cleaned.split()) < 2:
        warning = "Very short input: the prediction may be unreliable."
    return {
        "category": str(model.classes_[best]),
        "confidence": float(probabilities[best]),
        "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probabilities)},
        "warning": warning,
    }


if __name__ == "__main__":
    print(predict_text("My order arrived broken."))
