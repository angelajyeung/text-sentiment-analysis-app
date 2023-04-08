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
default = "I am happy."
text = st.text_area("Enter text here", default)

# sentiment analysis of input text
if st.button("Submit"):
    # analyze the text
    prediction = classifier(default)
    preds = [{"Score: ": round(pred["score"], 4), "Label: ": pred["label"]} for pred in prediction]
    # print the sentiment
    st.write(preds)