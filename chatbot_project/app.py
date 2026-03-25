import streamlit as st
import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize
lemmatizer = WordNetLemmatizer()

# Load intents
with open('data/intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
label_encoder = pickle.load(open('model/label_encoder.pkl', 'rb'))
model = pickle.load(open('model/chatbot_model.pkl', 'rb'))

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    probabilities = model.predict_proba([bow])[0]
    max_prob_index = np.argmax(probabilities)
    max_prob = probabilities[max_prob_index]
    
    if max_prob > 0.25:
        predicted_label = label_encoder.inverse_transform([max_prob_index])[0]
        return [{'intent': predicted_label, 'probability': str(max_prob)}]
    return []

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

def chat(message):
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    return response

# Page config
st.set_page_config(
    page_title="ML ChatBot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 ML-Based ChatBot")
st.caption("Powered by Scikit-learn Neural Network")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today? 😊"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    response = chat(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("ML-based chatbot using Neural Network")
    
    st.markdown("---")
    
    st.write("**💡 Try saying:**")
    st.code("Hello")
    st.code("Show products")
    st.code("Any discount?")
    st.code("How to order?")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today? 😊"}
        ]
        st.rerun()