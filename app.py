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
from transformers import pipeline
import pandas as pd

# Set up the model and the tokenizer
model_name = st.sidebar.selectbox("Select Model", ["bert-base-uncased", "roberta-base"])
model = pipeline("text-classification", model=model_name, tokenizer=model_name)
toxicity_classes = ["threat", "obscene", "insult", "identity_hate"]

# Define the function to extract the highest toxicity class and its probability
def extract_toxicity(text):
    results = model(text, multi_label=True)
    max_class_idx = results[0]["scores"].argmax()
    return toxicity_classes[max_class_idx], results[0]["scores"][max_class_idx]

# Load the dataset
dataset = pd.read_csv('data.csv')

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
        st.write(f"**Tweet {i+1}:** {row['comment_text']}")
        # Extract the highest toxicity class and its probability
        class_label, class_prob = extract_toxicity(row["comment_text"])
        # Display the results in a table
        st.table([[row["comment_text"], class_label, class_prob]])