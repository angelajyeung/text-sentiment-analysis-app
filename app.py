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
text = st.text_area("Enter text here", default)

# sentiment analysis of input text
if st.button("Submit"):
    # analyze the text
    prediction = classifier(text)
    for pred in prediction:
        result_dict = {
            "Score": round(pred["score"], 4),
            "label": pred["label"]
        }
        classification_df = pd.DataFrame([result_dict])
        st.dataframe(classification_df)

# # Import libraries
# import streamlit as st
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # HF Space Formatting
# st.title("Toxicity Classification App")
# st.markdown("Select a model and enter a text to classify its toxicity.")

# # Import the model
# model_name = "angelajyeung/results"

# # Load fine-tuned model and tokenizer
# model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# # Run the input text through the model to predict the toxicity class
# def predict(text, tokenizer):
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

# # Read tweets from the test dataset
# test_df = pd.read_csv("test.csv")
# tweets = test_df["comment_text"].tolist()[:10]

# # Prepopulate table with tweets and their classifications
# results = []
# for tweet in tweets:
#     result = predict(tweet, tokenizer)
#     result_dict = {
#         "Text": tweet,
#         "Class 1": result.iloc[0]["Toxicity Class"],
#         "Probability 1": result.iloc[0]["Probability"],
#         "Class 2": result.iloc[1]["Toxicity Class"],
#         "Probability 2": result.iloc[1]["Probability"]
#     }
#     results.append(result_dict)
# prepopulated_df = pd.DataFrame(results)
# st.dataframe(prepopulated_df)

# # Model selection
# model_options = ["Fine-Tuned Model"]
# model_selection = st.selectbox("Select a model", model_options)

# # Text input
# text_input = st.text_area("Enter some text")

# # Classification
# if st.button("Classify"):
#     results = predict(text_input, tokenizer)
#     result_dict = {
#         "Text": text_input,
#         "Class 1": results.iloc[0]["Toxicity Class"],
#         "Probability 1": results.iloc[0]["Probability"],
#         "Class 2": results.iloc[1]["Toxicity Class"],
#         "Probability 2": results.iloc[1]["Probability"]
#     }
#     classification_df = pd.DataFrame([result_dict])
#     st.dataframe(classification_df)

# import streamlit as st
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoConfig

# st.title("Toxicity Classification App")
# st.markdown("Select a model and enter a text to classify its toxicity.")

# model_name = "angelajyeung/model"
# model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# def predict(text, tokenizer):
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
#     results = predict(text_input, tokenizer)
#     st.dataframe(results)
