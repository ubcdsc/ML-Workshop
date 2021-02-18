import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="ML Workshop By DSC UBC",
    layout='wide'
)

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
tool = st.sidebar.selectbox("Tool", ["Website Q&A", "Sentiment Analysis", "Text Generation", "Summary Generation"])


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


@st.cache(suppress_st_warning=True)
def generatetext(starting_text):
    gpt2 = pipeline('text-generation')
    answer = gpt2(starting_text, max_length=50, num_return_sequences=2)
    return answer


@st.cache(suppress_st_warning=True)
def generatesummary(text):
    summarizer = pipeline("summarization")
    answer = (summarizer(text, max_length=100, min_length=30, do_sample=False))
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
    user_input = st.text_input("Enter Text",value="I love Machine Learning")

    if st.button('Get Sentiment'):
        answer = generatesentiment(user_input)
        st.header("Answer")
        st.write(answer)


def text():
    st.write("# GPT-2 Text Generation")
    user_input = st.text_input("Enter Text",value="I love Machine Learning but")

    if st.button('Get Text Answer'):
        answer = generatetext(user_input)
        st.header("Answer")
        st.write(answer)


def summary():
    st.write("# Summary Generation")
    user_input = st.text_area("Enter Text",value="Machine learning (ML) is the study of computer algorithms that "
                                                 "improve automatically through experience.[1] It is seen as a part "
                                                 "of artificial intelligence. Machine learning algorithms build a "
                                                 "model based on sample data, known as training data, in order to "
                                                 "make predictions or decisions without being explicitly programmed "
                                                 "to do so.[2] Machine learning algorithms are used in a wide variety "
                                                 "of applications, such as email filtering and computer vision, "
                                                 "where it is difficult or unfeasible to develop conventional "
                                                 "algorithms to perform the needed tasks. A subset of machine "
                                                 "learning is closely related to computational statistics, "
                                                 "which focuses on making predictions using computers; but not all "
                                                 "machine learning is statistical learning. The study of mathematical "
                                                 "optimization delivers methods, theory and application domains to "
                                                 "the field of machine learning. Data mining is a related field of "
                                                 "study, focusing on exploratory data analysis through unsupervised "
                                                 "learning.[4][5] In its application across business problems, "
                                                 "machine learning is also referred to as predictive analytics.")

    if st.button('Get Summary'):
        answer = generatesummary(user_input)
        st.header("Answer")
        st.write(answer)


if tool == "Website Q&A":
    website_qna()

if tool == "Sentiment Analysis":
    sentiment()

if tool == "Text Generation":
    text()

if tool == "Summary Generation":
    summary()
