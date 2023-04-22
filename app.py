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

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

st.title("Toxic Comment Classification")
st.markdown("## Using HuggingFace Transformers - Hosted on :hugging_face: Spaces")

model_names = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "google/electra-base-discriminator"]
model_name = st.selectbox("Select Pre-trained Model", model_names)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_toxicity(tweet):
    inputs = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    result = {}
    for i, label in enumerate(classes):
        result[label] = probabilities[i]
    return result

default_text = "I hate you"
text = st.text_input("Enter text to classify toxicity", default_text)
if st.button("Classify"):
    results = predict_toxicity(text)

if 'results' in locals():
    df = pd.read_csv('test.csv')
    df['predicted_toxicity'] = df['comment_text'].apply(lambda x: max(predict_toxicity(x), key=predict_toxicity(x).get))
    df['probability'] = df['comment_text'].apply(lambda x: predict_toxicity(x)[max(predict_toxicity(x), key=predict_toxicity(x).get)])
    st.table(df[['comment_text', 'predicted_toxicity', 'probability']])