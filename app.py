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
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained sentiment analysis model
model_name = st.sidebar.selectbox("Select pre-trained model", ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"])
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Load the Jigsaw Toxic Comment Classification Challenge dataset
data = pd.read_csv('test.csv')

# Define the toxicity labels
toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Create the app layout
st.title("Toxicity Classification")
st.sidebar.title("Select Options")

# Create the sidebar for selecting the model
st.sidebar.subheader("Select pre-trained model")

# Show the dropdown menu for selecting the model
st.sidebar.selectbox("Select pre-trained model", ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"])

# Create the table to display the results
st.subheader("Results")
st.write("")

# Display the table with the tweet, the highest toxicity class, and its probability
result_table = pd.DataFrame(columns=['Tweet', 'Toxicity Class', 'Probability'])
for i in range(len(data)):
    tweet = data.iloc[i]['comment_text']
    scores = classifier(tweet)
    highest_index = scores[0]['scores'].index(max(scores[0]['scores']))
    highest_class = toxicity_labels[highest_index]
    highest_prob = scores[0]['scores'][highest_index]
    result_table = result_table.append({'Tweet': tweet, 'Toxicity Class': highest_class, 'Probability': highest_prob}, ignore_index=True)
st.table(result_table)