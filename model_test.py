import torch
import torch.nn as nn
from model import SentimentAnalysisModel
from torch.nn.utils.rnn import pad_sequence
import pickle

# Define hyperparameters
embedding_dim = 50
hidden_dim = 100
num_classes = 4
output_dim = num_classes
num_layers = 4
num_epochs = 150
batch_size = 32
dropout_prob = 0.2

# Initialize the model and move to GPU
model = SentimentAnalysisModel(vocab_size=10000, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout_prob=dropout_prob)
print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def applytokenizer(text):
    text = tokenizer.texts_to_sequences([text])[0]  # Extract the inner list
    return text

# define function to predict the sentiment of a given text
def predict_sentiment(text):
    model.eval()  # Set the model to evaluation mode
    tokenized_text = applytokenizer(text)
    indexed_text = torch.tensor(tokenized_text, dtype=torch.long)  # Convert to tensor and add batch dimension
    padded_text = pad_sequence([indexed_text], batch_first=True)  # Wrap with another list

    with torch.no_grad():
        output = model(padded_text)
        predicted_class = output.argmax(dim=1)

    return predicted_class.item()  # Get the index of the predicted class

test_text = "This is a very shit movie."
predicted_sentiment = predict_sentiment(test_text)

# let's convert the class index to the correct sentiment label
sentiment_labels = {0: 'Negative', 1: 'Uncertain', 2: 'Positive', 3: 'Litigious'}
predicted_sentiment_label = sentiment_labels[predicted_sentiment]

print(f'The sentiment of "{test_text}" is "{predicted_sentiment_label}".')