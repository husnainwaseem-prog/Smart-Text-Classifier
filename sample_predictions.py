"""Run 12 new example sentences (not in the dataset) through the trained model.

Run from the project root:  python sample_predictions.py
The 'Expected' column is what a human would label the sentence; 'Predicted'
is the real output of the saved model. Wrong predictions are shown, not hidden.
"""
from src.predict import predict_text
from src.utils import ModelNotFoundError

SAMPLES = [
    ("My order arrived broken.", "Complaint"),
    ("The refund I was promised never showed up", "Complaint"),
    ("My internet has stopped working again", "Complaint"),
    ("How do I reset my password?", "Inquiry"),
    ("Do you deliver on Sundays?", "Inquiry"),
    ("What documents are needed to renew my membership?", "Inquiry"),
    ("The service was very helpful.", "Feedback"),
    ("I really like the new mobile app", "Feedback"),
    ("It would be better if the website had a night mode", "Feedback"),
    ("I went shopping yesterday.", "Other"),
    ("My friend is coming over for dinner tonight", "Other"),
    ("The mountains look lovely under the snow", "Other"),
]


def main() -> None:
    try:
        results = [(text, expected, predict_text(text)) for text, expected in SAMPLES]
    except ModelNotFoundError as error:
        print(error)
        return
    correct = 0
    print(f"{'Input':<52} {'Expected':<10} {'Predicted':<10} Conf.  OK?")
    for text, expected, result in results:
        ok = result["category"] == expected
        correct += ok
        print(f"{text:<52} {expected:<10} {result['category']:<10} "
              f"{result['confidence']:.0%}   {'yes' if ok else 'NO'}")
    print(f"\n{correct}/{len(SAMPLES)} matched the expected label.")


if __name__ == "__main__":
    main()
