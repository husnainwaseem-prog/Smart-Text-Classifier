"""Basic tests. Written with unittest classes so they run with pytest OR plain Python."""
import tempfile
import unittest
from pathlib import Path

from src.predict import load_model, predict_text
from src.preprocessing import clean_text
from src.train import train_and_save
from src.utils import (CATEGORIES, DATASET_PATH, MODEL_PATH, DatasetError,
                       InvalidInputError, load_dataset)


def write_csv(folder, content, name="data.csv"):
    path = Path(folder) / name
    path.write_text(content, encoding="utf-8")
    return path


class TestDataset(unittest.TestCase):
    def test_real_dataset_loads_and_is_valid(self):
        df, report = load_dataset(DATASET_PATH)
        self.assertGreaterEqual(len(df), 200)
        self.assertEqual(set(df["category"]), set(CATEGORIES))
        self.assertEqual(report["rows_removed_duplicates"], 0)
        self.assertFalse(df.duplicated().any())

    def test_missing_file(self):
        with self.assertRaises(DatasetError):
            load_dataset("does_not_exist.csv")

    def test_missing_column(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_csv(folder, "text,label\nhello,Other\n")
            with self.assertRaises(DatasetError):
                load_dataset(path)

    def test_missing_labels_are_dropped(self):
        rows = "text,category\n" + "".join(
            f"sentence number {i} for {c},{c}\n" for c in CATEGORIES for i in range(3))
        rows += "a text without label,\n"
        with tempfile.TemporaryDirectory() as folder:
            df, report = load_dataset(write_csv(folder, rows))
        self.assertEqual(len(df), 12)
        self.assertEqual(report["rows_removed_missing_or_empty"], 1)

    def test_unknown_label_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_csv(folder, "text,category\nhello there,Spam\n")
            with self.assertRaises(DatasetError):
                load_dataset(path)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(DatasetError):
                load_dataset(write_csv(folder, ""))


class TestPreprocessing(unittest.TestCase):
    def test_lowercase_punctuation_whitespace(self):
        self.assertEqual(clean_text("  Hello,   WORLD!!  "), "hello world")

    def test_apostrophe_removed(self):
        self.assertEqual(clean_text("It won't work"), "it wont work")

    def test_keeps_useful_words(self):
        self.assertEqual(clean_text("How do I reset it?"), "how do i reset it")

    def test_empty_and_invalid_values(self):
        for value in ["", "   ", None, float("nan"), 123, "!!!"]:
            self.assertEqual(clean_text(value), "")


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.is_file():
            train_and_save()
        cls.model = load_model()

    def test_output_structure(self):
        result = predict_text("My order arrived broken.", self.model)
        self.assertIn(result["category"], CATEGORIES)
        self.assertTrue(0 <= result["confidence"] <= 1)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)

    def test_empty_input(self):
        for bad in ["", "   ", "?!?!"]:
            with self.assertRaises(InvalidInputError):
                predict_text(bad, self.model)

    def test_invalid_type(self):
        for bad in [None, 42, ["text"]]:
            with self.assertRaises(InvalidInputError):
                predict_text(bad, self.model)

    def test_very_short_input_gives_warning(self):
        self.assertIsNotNone(predict_text("hello", self.model)["warning"])

    def test_known_samples(self):
        # Sanity checks on clear examples; not a guarantee for every sentence.
        cases = {
            "How do I reset my password?": "Inquiry",
            "The refund I was promised never showed up": "Complaint",
            "The service was very helpful.": "Feedback",
            "I went shopping yesterday.": "Other",
        }
        for text, expected in cases.items():
            self.assertEqual(predict_text(text, self.model)["category"], expected, text)


if __name__ == "__main__":
    unittest.main()
