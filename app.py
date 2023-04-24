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
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoConfig, FlaxAutoModelForVision2Seq

# title
st.title("Toxicity Classifier")

# subtitle
st.markdown("## Text Classification - Using `streamlit` -  hosted on 🤗 Spaces")

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("bert-base-uncased", path = "model_final", local_files_only = True)
model = FlaxAutoModelForVision2Seq.from_config(config)

# Define the tokenizer and the pipeline for inference
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Set up the dropdown menu with the model names
model_names = ["Fine-tuned Model"]
model_name = st.sidebar.selectbox('Select Model', model_names, index=0)

# Set up the text area for user input
text = st.text_area("Enter text here", "")

# Create a function to predict the toxicity class of the input text
def predict(text):
    prediction = classifier(text)

    # Sort the predictions by probability in descending order
    sorted_predictions = sorted(prediction, key=lambda x: x["score"], reverse=True)

    # Create a dataframe with the predictions
    df = pd.DataFrame(sorted_predictions[:2])
    df = df.rename(columns={'label': 'Toxicity Class', 'score': 'Probability'})
    df["Rank"] = df["Probability"].rank(method="dense", ascending=False)
    df = df[["Rank", "Toxicity Class", "Probability"]]

    return df

# Show the predictions in a table when the user clicks the "Submit" button
if st.button("Submit"):
    df = predict(text)
    st.write(df)