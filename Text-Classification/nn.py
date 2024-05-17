import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import classification_report

# Please note: glove.6B.50d.txt is not included in the repository. You can download it from https://nlp.stanford.edu/projects/glove/
# Then, add the glove.6B.50d.txt from glove.6B.zip to the data folder

torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_size, embeddings):
        super(Net, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_size, embeddings):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output[:, -1, :])  
        return output

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_size, embeddings):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.fc(output[:, -1, :]) 
        return output

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_size, embeddings):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size) 

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output[:, -1, :])  
        return output

def load_embeddings(embedding_file, word_index, embed_dim):
    embeddings_index = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    return torch.FloatTensor(embeddings_matrix)

def load_data(file_path):
    data = pd.read_csv(file_path)
    tweets = data['tweet'].tolist()
    labels = torch.tensor(data['class'], dtype=torch.long)
    return tweets, labels

def preprocess_data(tweets, vocab):
    indexed_tweets = [[vocab[token] for token in tweet.split()] for tweet in tweets]
    padded_tweets = pad_sequence([torch.tensor(tweet) for tweet in indexed_tweets], batch_first=True)
    return padded_tweets

def load_vocab(tweets):
    vocab = {}
    idx = 0
    for tweet in tweets:
        tokens = tweet.split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_tweets, batch_labels in tqdm(train_loader, desc='Training', unit='batch'):
        batch_tweets, batch_labels = batch_tweets.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_tweets)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_tweets, batch_labels in tqdm(data_loader, desc='Evaluation', unit='batch'):
            batch_tweets, batch_labels = batch_tweets.to(device), batch_labels.to(device)

            outputs = model(batch_tweets)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_labels.tolist())
            y_pred.extend(predicted.tolist())

    return classification_report(y_true, y_pred, output_dict=True)

def train_ff_model(args, train_loader, test_loader, vocab_size, device, embeddings):
    model = Net(vocab_size, args.embed_dim, args.hidden_dim, 3, embeddings) 
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}')

    test_report = evaluate_model(model, test_loader, device)
    print("Test Report:")
    print(test_report)

def train_lstm_model(args, train_loader, test_loader, vocab_size, device, embeddings):
    model = LSTMModel(vocab_size, args.embed_dim, args.hidden_dim, 3, embeddings)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}')

    test_report = evaluate_model(model, test_loader, device)
    print("Test Report:")
    print(test_report)

def train_gru_model(args, train_loader, test_loader, vocab_size, device, embeddings):
    model = GRUModel(vocab_size, args.embed_dim, args.hidden_dim, 3, embeddings)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}')

    test_report = evaluate_model(model, test_loader, device)
    print("Test Report:")
    print(test_report)

def train_bilstm_model(args, train_loader, test_loader, vocab_size, device, embeddings):
    model = BiLSTMModel(vocab_size, args.embed_dim, args.hidden_dim, 3, embeddings)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}')

    test_report = evaluate_model(model, test_loader, device)
    print("Test Report:")
    print(test_report)

def main(args):
    tweets, labels = load_data(args.data_file)
    vocab = load_vocab(tweets)
    vocab_size = len(vocab)

    embeddings = load_embeddings('data/glove.6B.50d.txt', vocab, args.embed_dim)

    padded_tweets = preprocess_data(tweets, vocab)
    dataset = torch.utils.data.TensorDataset(padded_tweets, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'ff':
        train_ff_model(args, train_loader, test_loader, vocab_size, device, embeddings)
    elif args.model == 'lstm':
        train_lstm_model(args, train_loader, test_loader, vocab_size, device, embeddings)
    elif args.model == 'gru':
        train_gru_model(args, train_loader, test_loader, vocab_size, device, embeddings)
    elif args.model == 'bilstm':
        train_bilstm_model(args, train_loader, test_loader, vocab_size, device, embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', default='data/labeled_data.csv')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for gradient descent.')
    parser.add_argument('--lowercase', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--pretrained', action='store_true', help='Whether to load pre-trained word embeddings.')
    parser.add_argument('--embed_dim', type=int, default=50, help='Default embedding dimension for GloVe.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Default hidden layer dimension.')
    parser.add_argument('--batch_size', type=int, default=16, help='Default number of examples per minibatch.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--model', default='ff', choices=['ff', 'lstm', 'gru', 'bilstm'], help='Type of model to use (choose from: ff, lstm, gru, bilstm)')

    args = parser.parse_args()
    main(args)
