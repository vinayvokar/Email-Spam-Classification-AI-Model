import streamlit as st
import pickle
import nltk

nltk.download('punkt_tab')


# function to do data preprocessing

from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

def trans_text(text):
    #convert into lower case
    text=text.lower()
    #convert into words
    text=nltk.word_tokenize(text)

    #Remove special characters
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()

    #removing stop words and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    #Stemming
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))


    return y

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.markdown("<h1 style='text-align: center; color: #32CD32;'>Vokar AI solution</h1>", unsafe_allow_html=True)
st.title("Email and SMS Spam Classifier")
st.markdown("<p >An AI email spam classifier allows users to enter a message, which is then analyzed using machine learning algorithms to determine whether it's spam or legitimate. </p>", unsafe_allow_html=True)

input_msg=st.text_area("Enter the message to check",height=300)

if st.button("Check"):
    # 1 Preprocess

    transformed_msg = trans_text(input_msg)

    # 2 Vectorize
    if isinstance(transformed_msg, list):
        transformed_msg = ' '.join(transformed_msg)

    vector_input = tfidf.transform([transformed_msg])
    # 3 Predict

    result = model.predict(vector_input)[0]

    # 4 Display

    if result == 1:
        st.header("Your message has been classified as spam.")
        st.warning(
            "⚠️ **Note:** This AI is still learning and may occasionally provide false predictions. Please verify results before taking action."
        )
    else:
        st.header("Your message has been classified as not spam.")
        st.warning(
            "⚠️ **Note:** This AI is still learning and may occasionally provide false predictions. Please verify results before taking action."
        )


