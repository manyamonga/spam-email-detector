import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):

    # Lowercase
    text = text.lower()

    # Tokenization
    words = nltk.word_tokenize(text)

    clean_words = []

    # Remove special characters
    for word in words:
        if word.isalnum():
            clean_words.append(word)

    words = clean_words[:]
    clean_words.clear()

    # Remove stopwords
    for word in words:
        if word not in stopwords.words('english') and word not in string.punctuation:
            clean_words.append(word)

    words = clean_words[:]
    clean_words.clear()

    # Stemming
    for word in words:
        clean_words.append(ps.stem(word))

    return " ".join(clean_words)

st.title("📧 Spam Email Detector")

st.write("Enter an email or message below to check whether it is Spam or Not Spam.")

# Text Area
input_sms = st.text_area("Enter Your Message")

# Button
if st.button('Check Email'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = vectorizer.transform([transformed_sms])

    # 3. Predict
    prediction = model.predict(vector_input)[0]

    # 4. Confidence Score
    probability = model.predict_proba(vector_input)[0]

    spam_confidence = round(probability[0][1] * 100, 2)
    ham_confidence = round(probability[0][0] * 100, 2)

    # 5. Display Result
    st.subheader("Result")

    if prediction == 1:
        st.error("🚨 Spam Message")
        st.write(f"Confidence: {spam_confidence}%")
    else:
        st.success("✅ Not Spam")
        st.write(f"Confidence: {ham_confidence}%")