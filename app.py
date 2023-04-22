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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the pre-trained tokenizer and model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
sa_pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Load the test dataset
test_df = pd.read_csv('tst.csv')

# Set up the app
st.title("Toxicity Classifier App")
st.markdown("## Built with `streamlit` and `HuggingFace`")

# Set up the model selection dropdown
model_selection = st.selectbox("Select a fine-tuned model:", ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"])

# Set up the table to display results
st.write("## Results")
st.write(test_df.head())

# Define a function to predict the toxicity class and its probability for each tweet
def predict_toxicity(tweet):
    result = sa_pipeline(tweet, max_length=512, truncation=True)
    return result[0]['label'], result[0]['score']

# Apply the predict_toxicity function to each tweet in the test dataset
test_df["toxicity_class"], test_df["toxicity_prob"] = zip(*test_df["tweet"].apply(predict_toxicity))

# Display the results in a table with tweet, toxicity class, and toxicity probability columns
st.write(test_df[["tweet", "toxicity_class", "toxicity_prob"]])