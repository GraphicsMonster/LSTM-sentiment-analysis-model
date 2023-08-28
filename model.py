import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from keras.preprocessing.text import Tokenizer
import preprocessor as p
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.vocab import GloVe

# Load dataset
path = 'Dataset/dataset.csv'
df = pd.read_csv(path)
df = df[:15000]
text = df[df['Language'] == 'en']['Text'].apply(lambda x: p.clean(x))
labels = df[df['Language'] == 'en']['Label']

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = torch.tensor(labels, dtype=torch.long)


# train test split
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_text, test_text = text[:train_size], text[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]
vocab_size = 10000
batch_size = 32

# Tokenize the text
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_text)
sequences = tokenizer.texts_to_sequences(train_text)
sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)

# Create a DataLoader for training
train_sequences = torch.tensor(sequences, dtype=torch.long)
train_dataset = torch.utils.data.TensorDataset(train_sequences, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained GloVe embeddings
glove = GloVe(name='6B', dim=50)  # You can choose different dimensions (e.g., 50, 100, 200, 300)
embedding_dim = glove.dim
vocab_size = len(tokenizer.word_index) + 1

# Initialize the embedding layer with pre-trained embeddings
embedding_matrix = torch.zeros(vocab_size, embedding_dim)
for word, idx in tokenizer.word_index.items():
    if word in glove.stoi:
        embedding_matrix[idx] = torch.tensor(glove[word])

embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super().__init__()
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.relu(lstm_out)
        output = self.fc(lstm_out)
        # output = self.relu(output)
        return output

# Define hyperparameters
vocab_size = 10000
embedding_dim = 50
hidden_dim = 100
num_classes = 4
output_dim = num_classes
num_layers = 4
num_epochs = 100
batch_size = 32
dropout_prob = 0.2

# Initialize the model
model = SentimentAnalysisModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout_prob=dropout_prob)
print(model)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Train the model
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for batch_sequences, batch_labels in train_loader:
        optimizer.zero_grad()
        output = model.forward(batch_sequences)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

# Tokenize the test text
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(test_text)
sequences = tokenizer.texts_to_sequences(test_text)
sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)

# Create a DataLoader for training
test_sequences = torch.tensor(sequences, dtype=torch.long)
test_dataset = torch.utils.data.TensorDataset(test_sequences, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    for batch_sequences, batch_labels in test_loader:
        output = model(batch_sequences)
        _, predicted = torch.max(output.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test data: {accuracy:.2f}%')