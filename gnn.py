import torch
import torch.nn as nn  # This is the necessary import for nn.Module
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GCNConv


# Define the same GNN model architecture that was used for training
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


# Load the tokenizer and BERT model for text embeddings
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

# Define the function to encode texts
def encode_texts(texts, max_len=50):
    encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    with torch.no_grad():
        embeddings = bert_model(**encodings).last_hidden_state.mean(dim=1)
    return embeddings


# Load the saved model
model_path = "gnn_model.pth"
input_dim = 768  # BERT embedding size
hidden_dim = 128
output_dim = 3  # Number of sentiment classes (Adjust this based on your case)

model = GNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully!")


import torch

# Function to build graph (fully connected graph)
def build_graph(embeddings):
    num_nodes = embeddings.shape[0]
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make edges bidirectional
    return edge_index

# New input text for prediction
new_texts = ["I love this product!", "This is terrible.", "Neutral feeling about this."]

# Encode the new texts into embeddings
new_embeddings = encode_texts(new_texts)

# Build the graph for the new embeddings
new_edge_index = build_graph(new_embeddings)

# Create a PyTorch Geometric Data object for the new input
new_data = Data(x=new_embeddings, edge_index=new_edge_index)

# Make predictions with the trained model
with torch.no_grad():
    output = model(new_data.x, new_data.edge_index)
    predictions = output.argmax(dim=1)

# Decode predictions back to sentiment labels (if you used LabelEncoder for labels)
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust based on your label encoding
predicted_labels = [label_mapping[pred.item()] for pred in predictions]

# Print the results
for text, label in zip(new_texts, predicted_labels):
    print(f"Text: {text}\nPredicted Sentiment: {label}\n")
