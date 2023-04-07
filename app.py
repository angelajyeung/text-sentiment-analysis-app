import streamlit as st  #Web App
from pysentimiento import create_analyzer 

# title
st.title("Extract Sentiment from Text")

# subtitle
st.markdown("## Sentiment Analysis - Using `streamlit` -  hosted on 🤗 Spaces")

st.markdown("")

# sentiment analyzer
analyzer = create_analyzer(task="sentiment", lang="en")

# text input
default = "I am happy."
text = st.text_input("Enter text here", default)

# sentiment analysis of input text
if st.button("Submit"):
    # analyze the text
    prediction = analyzer.predict(text)
    # print the sentiment
    st.write(prediction)