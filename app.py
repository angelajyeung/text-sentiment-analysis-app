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

# import streamlit as st
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoConfig, FlaxAutoModelForVision2Seq

# st.title("Toxicity Classification App")
# st.markdown("Select a model and enter a text to classify its toxicity.")

# model_name = "angelajyeung/model"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name)
# # classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# # # Create a function to predict the toxicity class of the input text
# # def predict(text):
# #     prediction = classifier(text)

# #     # Sort the predictions by probability in descending order
# #     sorted_predictions = sorted(prediction, key=lambda x: x["score"], reverse=True)

# #     # Create a dataframe with the predictions
# #     df = pd.DataFrame(sorted_predictions[:2])
# #     df = df.rename(columns={'label': 'Toxicity Class', 'score': 'Probability'})
# #     df["Rank"] = df["Probability"].rank(method="dense", ascending=False)
# #     df = df[["Rank", "Toxicity Class", "Probability"]]

# #     return df

# def predict(text):
#     inputs = tokenizer.encode_plus(
#         text,
#         max_length=128,
#         truncation=True,
#         padding="max_length",
#         return_tensors="pt"
#     )

#     outputs = model(**inputs)
#     probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]

#     classes = tokenizer.get_vocab().keys()
#     results = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)

#     return pd.DataFrame(results[:2], columns=["Toxicity Class", "Probability"])

# # Model selection
# model_options = ["Fine-Tuned Model"]
# model_selection = st.selectbox("Select a model", model_options)

# # Text input
# text_input = st.text_area("Enter some text")

# # Classification
# if st.button("Classify"):
#     results = predict(text_input)
#     st.dataframe(results)

# # Show the predictions in a table when the user clicks the "Submit" button
# if st.button("Submit"):
#     df = predict(text)
#     st.write(df)

# import streamlit as st
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, FlaxAutoModelForVision2Seq

# st.title("Toxicity Classification App")
# st.markdown("Select a model and enter a text to classify its toxicity.")

# model_name = "angelajyeung/model"
# model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# def predict(text):
#     inputs = tokenizer.encode_plus(
#         text,
#         max_length=128,
#         truncation=True,
#         padding="max_length",
#         return_tensors="pt"
#     )

#     outputs = model(**inputs)
#     probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]

#     classes = list(tokenizer.get_vocab().keys())  
#     results = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)

#     return pd.DataFrame(results[:2], columns=[classes[0], classes[1]])

# # Model selection
# model_options = ["Fine-Tuned Model"]
# model_selection = st.selectbox("Select a model", model_options)

# # Text input
# text_input = st.text_area("Enter some text")

# # Classification
# if st.button("Classify"):
#     results = predict(text_input)
#     st.dataframe(results)

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoConfig

st.title("Toxicity Classification App")
st.markdown("Select a model and enter a text to classify its toxicity.")

model_name = "angelajyeung/model"
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def predict(text, tokenizer):
    inputs = tokenizer.encode_plus(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]

    classes = tokenizer.get_vocab().keys()
    results = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)

    return pd.DataFrame(results[:2], columns=["Toxicity Class", "Probability"])

# Model selection
model_options = ["Fine-Tuned Model"]
model_selection = st.selectbox("Select a model", model_options)

# Text input
text_input = st.text_area("Enter some text")

# Classification
if st.button("Classify"):
    results = predict(text_input, tokenizer)
    st.dataframe(results)
