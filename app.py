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
import pandas as pd
from transformers import pipeline

# Set up the model and the tokenizer
model_name = st.sidebar.selectbox("Select Model", ["bert-base-uncased", "roberta-base"])
model = pipeline("text-classification", model=model_name, tokenizer=model_name)
toxicity_classes = ["threat", "obscene", "insult", "identity_hate"]

# Define the function to extract the highest toxicity class and its probability
def extract_toxicity(text):
    results = model(text)
    max_class_idx = results[0].argmax()
    toxicity_class = class_names[max_class_idx]
    probability = results[0][max_class_idx].item()
    return toxicity_class, probability

# Load the dataset
dataset = pd.read_csv('test.csv')

# Define the columns for the table
columns = ["Tweet", "Toxicity Class", "Probability"]

# Set up the app
st.title("Toxicity Detector")
st.sidebar.markdown("### Model Configuration")
st.sidebar.markdown("Select a model and click on 'Run' to load it.")
if st.sidebar.button("Run"):
    st.markdown("## Results")
    # Display the table of results
    for i, row in dataset.iterrows():
        tweet = row["comment_text"]
        toxicity_class, probability = extract_toxicity(tweet)
        data = [tweet, toxicity_class, probability]
        st.write(pd.DataFrame([data], columns=columns))