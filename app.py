"""Streamlit interface for the Smart Text Classifier.

Run from the project root:  streamlit run app.py
"""
import streamlit as st

from src.predict import load_model, predict_text
from src.utils import InvalidInputError, ModelNotFoundError

CATEGORY_HELP = {
    "Complaint": "Reports a problem or expresses dissatisfaction (\"My internet is not working\").",
    "Inquiry": "Asks a question or requests information (\"How can I reset my password?\").",
    "Feedback": "Opinions, praise or suggestions (\"The service was excellent\").",
    "Other": "Everyday statements unrelated to the above (\"I went to the market today\").",
}

st.set_page_config(page_title="Smart Text Classifier", page_icon="💬")
st.title("Smart Text Classifier")
st.write("Type a short message and the model will classify it as "
         "**Complaint**, **Inquiry**, **Feedback** or **Other**.")

user_text = st.text_area("Enter your message:", placeholder="My internet has stopped working")

if st.button("Predict"):
    try:
        load_model()  # fails early with a friendly message if no model exists
        result = predict_text(user_text)
    except ModelNotFoundError as error:
        st.error(str(error))
    except InvalidInputError as error:
        st.warning(str(error))
    else:
        st.subheader("Predicted Category:")
        st.success(result["category"])
        st.write(f"Confidence: **{result['confidence']:.1%}**")
        st.progress(result["confidence"])
        st.caption("This is the model's estimated probability, not a guarantee "
                   "that the prediction is correct.")
        if result["warning"]:
            st.info(result["warning"])
        with st.expander("Probability for each category"):
            for category, probability in sorted(
                    result["probabilities"].items(), key=lambda item: -item[1]):
                st.write(f"{category}: {probability:.1%}")

with st.expander("What do the categories mean?"):
    for category, description in CATEGORY_HELP.items():
        st.markdown(f"- **{category}**: {description}")
