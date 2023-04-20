import streamlit as st  #Web App
from transformers import pipeline

# title
st.title("Extract Sentiment from Text")

# subtitle
st.markdown("## Sentiment Analysis - Using `streamlit` -  hosted on 🤗 Spaces")

st.markdown("")

# sentiment analyzer
classifier = pipeline(task="sentiment-analysis")

# text input
default = "I am happy today."
text = st.text_area("Enter text here", "")

# sentiment analysis of input text
if st.button("Submit"):
    # analyze the text
    prediction = classifier(text)
    preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in prediction]
    # print the sentiment
    st.write(preds)