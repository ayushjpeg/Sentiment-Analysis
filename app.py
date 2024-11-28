from flask import Flask, request, render_template, jsonify
import numpy as np
from keras.models import load_model
import pickle
from flask import jsonify


app = Flask(__name__)

# Load pre-trained model and vectorizer
model = load_model('sentiment_model.h5')
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def predict_sentiment(user_input, model, vectorizer):
    user_input_vectorized = vectorizer.transform([user_input]).toarray()
    additional_features = np.array([[0, 0]])  # Adjust with actual features if needed
    user_input_new = np.concatenate((user_input_vectorized, additional_features), axis=1)
    prediction = model.predict(user_input_new)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[predicted_class]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer
    user_input = request.form['user_input']
    sentiment = predict_sentiment(user_input, model, vectorizer)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
