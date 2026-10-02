# What I Need To Understand Before Submission

1. **What the project does.** You type a short sentence; the program says if it is a Complaint, Inquiry, Feedback or Other, with a confidence percentage.
2. **How the dataset works.** `dataset.csv` has 240 sentences, each with a label. The model learns from these examples. I wrote them myself (say this openly). 60 per class so no class dominates.
3. **What preprocessing does.** Makes text uniform: lowercase, no punctuation, one space between words. "Won't!" and "wont" become the same token. We keep words like "not"/"how" because they carry meaning for our classes.
4. **What TF-IDF does.** Converts a sentence into numbers. TF = how often a word is in this sentence. IDF = how rare the word is across all sentences (rare words are more informative; "the" is not). TF × IDF = importance of a word in this sentence. We also use word pairs ("not working").
5. **What Logistic Regression does.** Learns a weight for every word feature for each class. For a new sentence it adds up the weights per class, converts them into probabilities (softmax), and picks the highest. Despite the name, it is a classifier.
6. **Train/test split.** 80% (192) of sentences are used to learn, 20% (48) are hidden and used only to check performance on unseen data. `random_state=42` makes the split repeatable; `stratify` keeps classes equal in both parts.
7. **Accuracy.** Correct predictions ÷ all predictions. Ours: 72.9% (35 of 48).
8. **F1-score.** Combines precision (when I say X, am I right?) and recall (did I find all X?). Weighted F1 averages per-class F1 by class size. Ours: 0.7293.
9. **How prediction works.** `predict_text` checks the input, then the saved pipeline cleans → TF-IDF → probabilities for 4 classes → highest wins. The model is loaded from `model.joblib`, not retrained.
10. **How Streamlit works.** The script `app.py` reruns from top to bottom on each interaction. `st.text_area` gives the text, `st.button` is True when clicked, then we call `predict_text` and display results with `st.success`, `st.write`, etc.
11. **Why the model makes mistakes.** Small dataset; it only sees words, not meaning; categories overlap; it overfits (100% on training, ~73% on test, ~62% in cross-validation).
12. **Why not deep learning.** Only 240 sentences: a neural network would overfit and be harder to explain. The assignment asked for a moderate project. A linear model is fast, explainable and sufficient as a baseline.
13. **What each file does.**
    - `src/preprocessing.py` – `clean_text`.
    - `src/utils.py` – paths, category list, custom errors, `load_dataset` (validation: missing file/columns/labels, duplicates, unknown labels).
    - `src/train.py` – builds the pipeline, splits, trains, evaluates, saves model + metrics.
    - `src/predict.py` – loads the model, `predict_text` with input checks.
    - `app.py` – web interface. `sample_predictions.py` – 12 demo predictions.
    - `tests/test_classifier.py` – automated checks.

# Likely Evaluator Questions
- **Why TF-IDF and not just word counts?** It down-weights very common words, so informative words matter more.
- **What is data leakage and how did you avoid it?** Using test information during training (e.g. fitting TF-IDF on all data). The Pipeline fits TF-IDF on training data only.
- **Why not evaluate on training data?** The model has seen it; 100% training accuracy only shows memorisation.
- **Your accuracy is ~73%. Is that good?** For 240 hand-written sentences it is a reasonable baseline; the cross-validation score (~62%) shows it is unstable. More data is the main fix.
- **Why weighted F1?** It accounts for class size; with balanced classes macro and weighted are identical.
- **What does 87% confidence mean?** The model's estimated probability, which can be wrong; not a guarantee.
- **Where is the weakest class?** Complaint (F1 0.61), confused with Feedback.
- **Did you tune on the test set?** No. I compared settings with cross-validation on training data only and saw no real difference.
- **How would you improve it?** More real data, compare Linear SVM/Naive Bayes, confidence threshold, multilingual support.
- **How is bad input handled?** Empty/non-text input raises `InvalidInputError`, shown as a friendly message; very short input gives a warning.
- **Why `python -m src.train`?** So Python treats `src` as a package and the saved model can find `src.preprocessing.clean_text` when loaded.
- **Did you use AI?** Yes, as allowed; I can explain every part (use this guide).
