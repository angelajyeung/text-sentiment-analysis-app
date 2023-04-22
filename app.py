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

# Load the necessary libraries
import streamlit as st
import pandas as pd
from transformers import pipeline

# Define a function to classify the text based on the selected pre-trained model
def classify_text(text, model_name):
    # Load the pre-trained model
    classifier = pipeline('text-classification', model=model_name, tokenizer=model_name)
    # Classify the text
    results = classifier(text, max_length=512)
    # Extract the highest toxicity class and its probability
    labels = results[0]['labels']
    scores = results[0]['scores']
    highest_toxicity = labels[0]
    highest_prob = scores[0]
    return highest_toxicity, highest_prob

# Define the list of pre-trained models to use in the app
model_list = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']

# Create the Streamlit app
st.title("Toxicity Classification App")
st.markdown("Select a pre-trained model to classify the toxicity of the text:")

# Create a dropdown menu to select the pre-trained model
model_name = st.selectbox("Model", model_list)

# Create a text area to input the text to classify
text = st.text_area("Enter the text to classify:")

# Classify the text when the user submits it
if st.button("Classify"):
    # Classify the text based on the selected model
    highest_toxicity, highest_prob = classify_text(text, model_name)
    # Create a table to display the results
    results_df = pd.DataFrame({'Text': [text], 'Toxicity Class': [highest_toxicity], 'Probability': [highest_prob]})
    st.table(results_df)