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

# import streamlit as st
# import pandas as pd
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# # Load the fine-tuned model from Hugging Face Model Hub
# @st.cache(allow_output_mutation=True)
# def load_model(model_name):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
#     return classifier

# # Define the labels and their corresponding colors
# LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# COLORS = ["#ff5733", "#ffcc33", "#cc33ff", "#33ff57", "#57b5ff", "#c733ff"]

# # Define a function to extract the highest-scoring label and its probability
# def extract_labels(predictions):
#     scores = [p["score"] for p in predictions]
#     max_score = max(scores)
#     max_index = scores.index(max_score)
#     return LABELS[max_index], max_score

# # Define the Streamlit app
# def main():
#     # Title
#     st.title("Toxic Comment Classification")

#     # Subtitle
#     st.markdown("## Multi-Class Classification - Using `HuggingFace` - Hosted on :hugging_face: Spaces")

#     # Select a model
#     model_name = st.selectbox("Select a fine-tuned model:", ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"])

#     # Load the model
#     classifier = load_model(model_name)

#     # Load the dataset
#     data = pd.read_csv("test.csv")

#     # Create a table to display the tweet and its predicted class
#     st.write("## Test Data")
#     st.write(data[["comment_text"]].head(10))

#     # Predict the labels for each tweet
#     predictions = classifier(data["comment_text"].tolist())

#     # Create a table to display the predicted classes and their probabilities
#     st.write("## Predicted Classes and Probabilities")
#     for i in range(10):
#         tweet = data["comment_text"][i]
#         predicted_label, predicted_score = extract_labels(predictions[i])
#         st.write(
#             {
#                 "Tweet": tweet,
#                 "Predicted Class": predicted_label,
#                 "Probability": f"{predicted_score:.2f}",
#             },
#             unsafe_allow_html=True,
#         )

#         # Add some color to the predicted label
#         predicted_color = COLORS[LABELS.index(predicted_label)]
#         st.markdown(
#             f'<span style="color:{predicted_color}">{predicted_label}</span>',
#             unsafe_allow_html=True,
#         )

# if __name__ == "__main__":
#     main()

import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# title
st.title("Sentiment Analysis App")

# subtitle
st.markdown("## Using Streamlit and Hugging Face to Analyze Sentiments")

# specify the GitHub URL and model file path
github_url = "https://github.com/angelajyeung/text-sentiment-analysis-app"
model_file_path = "blob/main/model_final.ipynb"

# load the model and tokenizer from the GitHub repository
model = AutoModelForSequenceClassification.from_pretrained(f"{github_url}/{model_file_path}")
tokenizer = AutoTokenizer.from_pretrained(f"{github_url}/{model_file_path}")

# Sidebar menu to select model
model_file = st.selectbox("Select Model", ["Fine-tuned model"])

# # Load the model and tokenizer
# model = AutoModelForSequenceClassification.from_pretrained(model_file)
# tokenizer = AutoTokenizer.from_pretrained(model_file)

# sentiment analyzer pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# text input
text = st.text_area("Enter text here", "")

# sentiment analysis of input text
if st.button("Submit"):
    # analyze the text
    prediction = classifier(text, return_all_scores=True)

    # store the results in a dataframe
    df = pd.DataFrame(prediction, columns=['Label', 'Score'])

    # sort the dataframe by scores in descending order
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # extract the top 2 labels and scores
    top1_label, top1_score = df.iloc[0]['Label'], df.iloc[0]['Score']
    top2_label, top2_score = df.iloc[1]['Label'], df.iloc[1]['Score']

    # display the results in a table
    st.write(pd.DataFrame({'Tweet': [text], 
                           'Highest Label': [top1_label], 'Highest Score': [top1_score], 
                           'Second Highest Label': [top2_label], 'Second Highest Score': [top2_score]}))