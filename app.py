import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load FAQ data
with open("faq.json") as file:
    data = json.load(file)

questions = data["questions"]
answers = data["answers"]


# Convert questions into vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


# Chatbot function
def chatbot_response(user_input):

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)

    index = similarity.argmax()

    score = similarity[0][index]

    if score < 0.2:
        return "Sorry, I don't know the answer."

    return answers[index]


# Streamlit UI
st.title("🤖 FAQ Chatbot")

user_question = st.text_input("Ask your question:")

if user_question:
    response = chatbot_response(user_question)
    st.success(response)