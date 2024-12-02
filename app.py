from flask import Flask, request, render_template, jsonify
import numpy as np
import torch
from model_loader import get_model, build_graph, preprocess_text
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model_name', 'keras')
    user_input = request.form['user_input']

    try:
        if model_name == 'keras':
            model, vectorizer = get_model('keras')
            user_input_vectorized = vectorizer.transform([user_input]).toarray()
            additional_features = np.array([[0, 0]])  # Adjust as needed
            user_input_new = np.concatenate((user_input_vectorized, additional_features), axis=1)
            prediction = model.predict(user_input_new)
            predicted_class = np.argmax(prediction, axis=1)[0]
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = label_map[predicted_class]

        elif model_name == 'cnn':
            print("entered")
            cnn_model, cnn_tokenizer = get_model('cnn')

            # Preprocess the user input
            clean_sentence = preprocess_text(user_input)

            # Tokenize and pad the input
            max_sequence_length = 50  # Define the maximum sequence length
            sequence = cnn_tokenizer.texts_to_sequences([clean_sentence])
            padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

            # Predict sentiment
            prediction = cnn_model.predict(padded_sequence)
            sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        elif model_name == 'nonAI':
            import pandas as pd
            from nltk.corpus import opinion_lexicon

            # Load the lexicons
            pos_list = set(opinion_lexicon.positive())
            neg_list = set(opinion_lexicon.negative())
            if not isinstance(user_input, str):
                return 'neutral'
            c = 0
            for word in user_input.split():
                if word in pos_list:
                    c += 1
                elif word in neg_list:
                    c -= 1
            if c > 0:
                sentiment =  'Positive'
            elif c < 0:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
        elif model_name == 'gnn':

            model, tokenizer, bert_model = get_model('gnn')

            # Tokenize and encode user input

            encodings = tokenizer([user_input], padding=True, truncation=True, max_length=50, return_tensors='pt')

            with torch.no_grad():

                embeddings = bert_model(**encodings).last_hidden_state.mean(dim=1)

            # Build a graph for the embeddings

            edge_index = build_graph(embeddings)

            # Make predictions

            with torch.no_grad():

                output = model(embeddings, edge_index)

                predicted_class = output.argmax(dim=1).item()

                label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

                sentiment = label_map[predicted_class]

        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
