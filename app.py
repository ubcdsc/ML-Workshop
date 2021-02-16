import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

st.sidebar.image("images/DSC University of British Columbia  Light Vertical-Logo.png", width=300, height=500)

st.sidebar.header("""
© ML Workshop By DSC UBC
""")
st.sidebar.markdown(
    "This app allows users to input text of their choice using the tools provided, and ask questions with the answer "
    "being extracted from the text.")
st.sidebar.markdown(
    "_When running the app the first time, it may take some time to initialise due to the requirements needing to be "
    "downloaded._")
tool = st.sidebar.selectbox("Tool", ["Website Q&A", "Sentiment Analysis"])


@st.cache(suppress_st_warning=True)
def generateAnswer(question, context):
    nlp = pipeline("question-answering")
    answer = nlp(question=question, context=context)
    return answer


@st.cache(suppress_st_warning=True)
def generatesentiment(text):
    nlp = pipeline('sentiment-analysis')
    answer = nlp(text)
    return answer


def website_qna():
    st.write("# Website QnA")
    user_input = st.text_input("Website Link:", value="https://en.wikipedia.org/wiki/Machine_learning")
    question = st.text_input("Question:", value="What is Machine Learning?")

    if st.button("Get Answer"):
        scraped_data = requests.get(user_input)
        article = scraped_data.text

        parsed_article = BeautifulSoup(article, 'lxml')

        paragraphs = parsed_article.find_all('p')

        article_text = ""
        for p in paragraphs:
            article_text += p.text

        answer = generateAnswer(question, article_text)
        st.header("Answer")
        st.write(answer)


def sentiment():
    st.write("# Sentiment Analysis")
    user_input = st.text_input("Enter Text")

    if st.button('Get Sentiment'):
        answer = generatesentiment(user_input)
        st.header("Answer")
        st.write(answer)


if tool == "Website Q&A":
    website_qna()

if tool == "Sentiment Analysis":
    sentiment()
