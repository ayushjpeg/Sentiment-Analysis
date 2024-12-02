
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from keras.models import load_model
import pickle

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def load_keras_model():
    model = load_model('sentiment_model.h5')
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


# Define the GNN model class
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Function to build a fully connected graph
def build_graph(embeddings):
    num_nodes = embeddings.shape[0]
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Bidirectional edges
    return edge_index


# Load the GNN model, tokenizer, and BERT model
def load_gnn_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')

    input_dim = 768  # BERT embedding size
    hidden_dim = 128
    output_dim = 3  # Sentiment classes

    model = GNN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('gnn_model.pth'))
    model.eval()

    return model, tokenizer, bert_model

import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import string


# Load CNN model and tokenizer
def load_cnn_model():
    # Load the tokenizer
    with open('tokenizer_CNN.pkl', 'rb') as f:
        cnn_tokenizer = pickle.load(f)

    # Load the CNN model
    cnn_model = load_model('sentiment_model_CNN.h5')
    print('success')

    return cnn_model, cnn_tokenizer


# Text preprocessing function for CNN
def preprocess_text(text):
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text


def get_model(model_name):
    if model_name == 'keras':
        return load_keras_model()
    elif model_name == 'gnn':
        return load_gnn_model()
    elif model_name == 'cnn':
        return load_cnn_model()
    else:
        raise ValueError("Unsupported model name.")
