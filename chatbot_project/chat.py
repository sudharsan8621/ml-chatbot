import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize
lemmatizer = WordNetLemmatizer()

# Load intents - WITH UTF-8 ENCODING
with open('data/intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
label_encoder = pickle.load(open('model/label_encoder.pkl', 'rb'))
model = pickle.load(open('model/chatbot_model.pkl', 'rb'))

def clean_sentence(sentence):
    """Tokenize and lemmatize sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert sentence to bag of words"""
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict intent class"""
    bow = bag_of_words(sentence)
    
    # Get prediction probabilities
    probabilities = model.predict_proba([bow])[0]
    
    # Get best prediction
    max_prob_index = np.argmax(probabilities)
    max_prob = probabilities[max_prob_index]
    
    ERROR_THRESHOLD = 0.25
    
    if max_prob > ERROR_THRESHOLD:
        predicted_label = label_encoder.inverse_transform([max_prob_index])[0]
        return [{'intent': predicted_label, 'probability': str(max_prob)}]
    
    return []

def get_response(intents_list, intents_json):
    """Get response for predicted intent"""
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"
    
    tag = intents_list[0]['intent']
    
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I didn't understand that."

def chat(message):
    """Main chat function"""
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    return response

# Test chatbot
if __name__ == "__main__":
    print("=" * 50)
    print("🤖 ML ChatBot Ready!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("🤖 ChatBot: Goodbye! Have a nice day! 👋")
            break
        
        if not user_input.strip():
            continue
            
        response = chat(user_input)
        print(f"🤖 ChatBot: {response}")