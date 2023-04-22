# import streamlit as st  #Web App
# from transformers import pipeline

# # title
# st.title("Extract Sentiment from Text")

# # subtitle
# st.markdown("## Sentiment Analysis - Using `streamlit` -  hosted on 🤗 Spaces")

# st.markdown("")

# # sentiment analyzer
# classifier = pipeline(task="sentiment-analysis")

# # text input
# default = "I am happy today."
# text = st.text_area("Enter text here", "")

# # sentiment analysis of input text
# if st.button("Submit"):
#     # analyze the text
#     prediction = classifier(text)
#     preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in prediction]
#     # print the sentiment
#     st.write(preds)

import pandas as pd
import streamlit as st
from transformers import pipeline, AutoTokenizer

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-classification", model=model_name, tokenizer=tokenizer)
    return model


def predict_toxicity(model, text):
    predictions = model(text)
    max_label, max_prob = max(predictions, key=lambda x: x['score'])
    return max_label, max_prob

# title
st.title("Text Toxicity Classifier")

# subtitle
st.markdown("## Toxicity Classifier - Using `streamlit` -  hosted on :hugging_face: Spaces")

st.markdown("")

models = {
    "bert-base-uncased": "bert-base-uncased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "roberta-base": "roberta-base",
    "electra-base": "google/electra-base-discriminator",
}

model_name = st.sidebar.selectbox("Select a finetuned model", list(models.keys()))

model = load_model(models[model_name])

df = pd.read_csv("test.csv")

results = []
for text in df["comment_text"]:
    max_label, max_prob = predict_toxicity(model, text)
    results.append((text, max_label, max_prob))

st.write(pd.DataFrame(results, columns=["Text", "Toxicity Class", "Probability"]))