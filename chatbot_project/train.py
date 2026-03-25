import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents - WITH UTF-8 ENCODING
with open('data/intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"📚 Words: {len(words)}")
print(f"🏷️ Classes: {len(classes)}")
print(f"📄 Documents: {len(documents)}")

# Save words and classes
pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

# Create training data
training_data = []
labels = []

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    training_data.append(bag)
    labels.append(document[1])

# Convert to numpy arrays
X_train = np.array(training_data)
y_train = np.array(labels)

print(f"✅ Training data shape: {X_train.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Save label encoder
pickle.dump(label_encoder, open('model/label_encoder.pkl', 'wb'))

# Build and train model using MLPClassifier (Neural Network)
print("🚀 Training started...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    verbose=True
)

model.fit(X_train, y_train_encoded)

# Save model
pickle.dump(model, open('model/chatbot_model.pkl', 'wb'))

print("✅ Model saved successfully!")
print(f"🎯 Training accuracy: {model.score(X_train, y_train_encoded) * 100:.2f}%")