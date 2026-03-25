# 🤖 ML-Based Chatbot

An intelligent chatbot built using Machine Learning and Natural Language Processing.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK-green.svg)

## 🌐 Live Demo

👉 **[Try the Chatbot Here](YOUR_STREAMLIT_APP_LINK)**

---

## 📸 Screenshot

![Chatbot Screenshot](https://via.placeholder.com/800x400?text=ML+Chatbot+Screenshot)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 Intent Classification | Understands user intent using ML |
| 💬 Multiple Responses | Random responses for natural feel |
| 🎨 Clean UI | Built with Streamlit |
| ⚡ Fast Response | Real-time predictions |
| 📱 Mobile Friendly | Works on all devices |

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **ML Framework:** Scikit-learn (MLPClassifier)
- **NLP Library:** NLTK
- **Web Framework:** Streamlit
- **Model:** Neural Network (Multi-layer Perceptron)

ml-chatbot/
│
├── data/
│ └── intents.json # Training data
│
├── model/
│ ├── words.pkl # Vocabulary
│ ├── classes.pkl # Intent classes
│ ├── label_encoder.pkl # Label encoder
│ └── chatbot_model.pkl # Trained model
│
├── app.py # Streamlit web app
├── train.py # Model training script
├── chat.py # Chat logic
├── requirements.txt # Dependencies
├── nltk.txt # NLTK packages
└── README.md # Documentation


---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-chatbot.git
cd ml-chatbot

2. Install Dependencies
pip install -r requirements.txt
---

3. Train Model
python train.py

4. Run Chatbot
# Terminal version
python chat.py

# Web version
streamlit run app.py

Sample Conversations
You Say	Bot Responds
"Hello"	"Hi there! What can I do for you?"
"Show products"	"We have Phones, Laptops, Cameras..."
"Any discount?"	"Use code SAVE10 for 10% off!"
"How to order?"	"Easy! Select product, Add to cart..."
"Thank you"	"You're welcome!"

Supported Intents
Intent	Examples
Greeting	Hi, Hello, Hey
Goodbye	Bye, See you
Products	Show products, What do you sell?
Price	How much?, Cost?
Order	How to order?, Buy now
Payment	Payment methods?, UPI accepted?
Delivery	Delivery time?, Free delivery?
Return	Return policy?, Refund?
Contact	Phone number?, Email?

Model Architecture
Input Layer (Bag of Words)
        ↓
Dense Layer (128 neurons, ReLU)
        ↓
Dense Layer (64 neurons, ReLU)
        ↓
Output Layer (Softmax - 20 classes)

Training Accuracy: 100%

Customization
Add New Intents
Edit data/intents.json:
{
  "tag": "new_intent",
  "patterns": ["pattern1", "pattern2", "pattern3"],
  "responses": ["response1", "response2"]
}

Then retrain:
python train.py

Contributing
Contributions are welcome! Feel free to:

Fork the repository
Create a new branch
Make changes
Submit a pull request

License
This project is licensed under the MIT License.


