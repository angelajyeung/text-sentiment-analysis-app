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
text = st.text_input("Enter text here", "")

# submit button
button = st.button("Submit")

# sentiment analysis of input text
if button is not None:
    # analyze the text
    prediction = analyzer.predict(text)
    # print the sentiment
    st.write(prediction)