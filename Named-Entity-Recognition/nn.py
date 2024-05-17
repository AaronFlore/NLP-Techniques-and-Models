"""
Description: Neural Network for Named Entity Recognition using PyTorch
Author: Aaron Floreani
Date: 3/26/2024
"""

import argparse
import numpy as np
import loader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Tagger(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_y):
        super(Tagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_y)

    def forward(self, sentences):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.fc(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores

def train_model(model, train_data, optimizer, epochs=10):
    model.train()
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for data in tqdm(train_data):
            model.zero_grad()

            sentence_in = torch.tensor(data['words'], dtype=torch.long).to(device)
            targets = torch.tensor(data['tags'], dtype=torch.long).to(device)

            tag_scores = model(sentence_in.unsqueeze(0))

            loss = loss_function(tag_scores.view(-1, model.fc.out_features), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data)}")


def evaluate_model(model, test_data, id_to_tag):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data in test_data:
            sentence_in = torch.tensor(data['words'], dtype=torch.long).to(device)
            targets = torch.tensor(data['tags'], dtype=torch.long).to(device)

            tag_scores = model(sentence_in.unsqueeze(0))  
            predicted_tags = torch.argmax(tag_scores, dim=2)
            predictions.extend(predicted_tags.view(-1).cpu().numpy())
            actuals.extend(targets.view(-1).cpu().numpy())

    converted_predictions = [id_to_tag[id] for id in predictions if id in id_to_tag]
    converted_actuals = [id_to_tag[id] for id in actuals if id in id_to_tag]

    print(classification_report(converted_actuals, converted_predictions))

def main(args):

    # Load the training data
    train_sentences = loader.load_sentences(args.train_file, args.lower)
    train_corpus, dics = loader.prepare_dataset(train_sentences, mode='train', lower=args.lower)
    vocab_size = len(dics['word_to_id'])
    num_y = len(dics['tag_to_id'])

    # Build the model
    model = Tagger(args.embed_dim, args.hidden_dim, vocab_size, num_y).to(device)

    # Train the NN model for the specified number of epochs.
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_model(model, train_corpus, optimizer, epochs=args.epochs)

    # Load the validation data for testing.
    test_sentences = loader.load_sentences(args.test_file, args.lower)
    test_corpus = loader.prepare_dataset(test_sentences, mode='test', lower=args.lower, 
                                         word_to_id=dics['word_to_id'], tag_to_id=dics['tag_to_id'])
    
    # Evaluate the NN model and compare to the HMM baseline.
    evaluate_model(model, test_corpus, dics['id_to_tag'])

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='data/eng.train')
    parser.add_argument('--test_file', default='data/eng.val')
    parser.add_argument('--lower', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for gradient descent.')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden layer dimension.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--model', default='lstm', choices=['ff', 'lstm'])

    args = parser.parse_args()
    main(args)
