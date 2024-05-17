"""
Description: NLP Text Generator
Date: January 30th, 2024
Author: Aaron Floreani
"""

import math
import argparse
import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize


class LanguageModel:

    def __init__(self, args):
        """
        A bigram language model with add-alpha smoothing.

        Attributes
        --------------------
            alpha -- alpha for add-alpha smoothing
            train_tokens -- list of training data tokens
            val_tokens -- list of validation data tokens
            vocab -- vocabulary frequency dict (key = word, val = frequency)
            token_to_idx -- vocabulary index dict (key = word, val = index)
            bigrams -- 2D np array of bigram probabilities, where each (i, j)
                       value is the smoothed prob of the bigram starting with
                       vocab token index i followed by vocab token index j
        """
        self.alpha = args.alpha
        self.train_tokens = self.tokenize(args.train_file)
        self.val_tokens = self.tokenize(args.val_file)
        self.show_plot = args.show_plot

        # Use only the specified fraction of training data.
        num_samples = int(args.train_fraction * len(self.train_tokens))
        self.train_tokens = self.train_tokens[: num_samples]
        self.vocab = self.make_vocab(self.train_tokens)
        self.token_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.bigrams = self.compute_bigrams(self.train_tokens)

    def get_indices(self, tokens):
        """
        Converts each of the string tokens to indices in the vocab.

        Parameters
        --------------------
            tokens    -- list of tokens

        Returns
        --------------------
            list of token indices in the vocabulary
        """
        return [self.token_to_idx[token] for token in tokens if token in self.token_to_idx]

    def compute_bigrams(self, tokens):
        """
        Populates probability values for a 2D np array of all bigrams.

        Parameters
        --------------------
            tokens    -- list of tokens
            alpha     -- alpha for add-alpha smoothing

        Returns
        --------------------
            bigrams   -- 2D np array of bigram probabilities, where each (i, j)
                       value is the smoothed prob of the bigram starting with
                       vocab token index i followed by vocab token index j
        """
        counts = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        probs = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        tokens = self.get_indices(tokens)

        # Count up all the bigrams.
        # Estimate bigram probabilities using the counts (and alpha).
        # First, add alpha to each bigram count.
        # Then, divide by the sum of the bigram counts + alpha times |V|.

        for i in range(len(tokens) - 1):
            counts[tokens[i]][tokens[i+1]] += 1

        for i in range(len(self.vocab)):
            total_count = np.sum(counts[i]) + self.alpha * len(self.vocab)
            probs[i] = (counts[i] + self.alpha) / total_count

        return probs

    def compute_perplexity(self, tokens):
        """
        Evaluates the LM by calculating perplexity on the given tokens.

        Parameters
        --------------------
            bigrams    -- 2D np array of bigram probabilities, where each (i, j)
                       value is the smoothed prob of the bigram starting with
                       vocab token index i followed by vocab token index j
            tokens     -- list of tokens

        Returns
        --------------------
            perplexity
        """
        tokens = self.get_indices(tokens)

        # Sum up all the bigram log probabilities in the test corpus.
        # Be sure to divide by the number of tokens, not the vocab size!

        if not tokens:
            print("Warning: Empty tokens list. Unable to calculate perplexity.")
            return None 

        bigrams = self.compute_bigrams(self.train_tokens)

        log_prob_sum = 0
        for i in range(len(tokens) - 1):
            log_prob_sum += np.log(bigrams[tokens[i]][tokens[i+1]])

        perplexity = np.exp(-log_prob_sum / len(tokens))
        return perplexity

    def tokenize(self, corpus):
        """
        Splits the given corpus file into tokens using nltk's tokenizer.

        Parameters
        --------------------
            corpus    -- filename as a string

        Returns
        --------------------
            tokens    -- list of tokens
        """
        #part a
        with open(corpus, 'r') as file:
            text = file.read()
        return word_tokenize(text)

    def make_vocab(self, train_tokens):
        """
        Creates a vocabulary dictionary that maps tokens to frequencies.

        Parameters
        --------------------
            train_tokens    -- list of training tokens

        Returns
        --------------------
            vocab           -- vocab frequency dict (key = word, val = freq)
        """
        vocab = defaultdict(int)
        for token in train_tokens:
            vocab[token] += 1
        return vocab

    def plot_vocab(self, vocab):
        """
        Plots words from most to least common, with frequency on the y-axis.

        Parameters
        --------------------
            vocab           -- vocab frequency dict (key = word, val = freq)
        """
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        words, frequencies = zip(*sorted_vocab)
        token_indices = list(range(len(words)))

        plt.figure(figsize=(12, 6))
        plt.plot(token_indices, frequencies, linestyle='-', color='b')
        plt.xlabel('Tokens')
        plt.ylabel('Frequency')
        plt.title('Word Frequencies')

        plt.show()
    
    def compute_ngrams(self, tokens, n):
        """
        Creates probability values for a 2D np array of all n-grams.

        Parameters
        --------------------
            tokens    -- list of tokens
            n         -- order of the n-grams

        Returns
        --------------------
            ngrams    -- 2D np array of n-gram probabilities
        """
        counts = np.zeros(tuple([len(self.vocab)] * n), dtype=float)
        probs = np.zeros(tuple([len(self.vocab)] * n), dtype=float)
        tokens = self.get_indices(tokens)

        for i in range(len(tokens) - n + 1):
            indices = tuple(tokens[i:i+n])
            counts[indices] += 1

        for i in range(len(self.vocab)):
            total_count = np.sum(counts[(i,) * n]) + self.alpha * len(self.vocab)**n
            probs[(i,) * n] = (counts[(i,) * n] + self.alpha) / total_count

        return probs

    def generate_text(self, n, length=100):
        """
        Generate text in the style of the book using n-grams.

        Parameters
        --------------------
            n         -- order of the n-grams
            length    -- length of the generated text (number of tokens)

        Returns
        --------------------
            generated_text -- list of generated tokens
        """
        if n < 1:
            raise ValueError("n must be greater than or equal to 1.")

        generated_text = []

        current_word = random.choice(list(self.token_to_idx.keys()))

        for _ in range(length):
            next_word_probs = self.bigrams[self.token_to_idx[current_word]]
            next_word_idx = np.random.choice(len(next_word_probs), p=next_word_probs)

            next_word = list(self.token_to_idx.keys())[next_word_idx]
            generated_text.append(next_word)
            current_word = next_word

        return generated_text


def main(args):
    lm = LanguageModel(args)

    # Plot word frequencies by setting command-line arg show_plot.
    if args.show_plot:
        lm.plot_vocab(lm.vocab)

    # Plot training and validation perplexities as a function of alpha.
    alpha_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    train_perplexities = []
    val_perplexities = []

    for alpha in alpha_values:
        lm.alpha = alpha
        train_perplexity = lm.compute_perplexity(lm.train_tokens)
        val_perplexity = lm.compute_perplexity(lm.val_tokens)

        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)

    plt.figure(figsize=(10, 5))
    plt.semilogx(alpha_values, train_perplexities, marker='o', label='Training Perplexity')
    plt.semilogx(alpha_values, val_perplexities, marker='o', label='Validation Perplexity')
    plt.xlabel('Alpha')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexities vs. Alpha')
    plt.legend()

    for (alpha, train_ppl, val_ppl) in zip(alpha_values, train_perplexities, val_perplexities):
        plt.text(alpha, train_ppl, f'({alpha:.4f}, {train_ppl:.2f})', ha='right', va='bottom')
        plt.text(alpha, val_ppl, f'({alpha:.4f}, {val_ppl:.2f})', ha='right', va='bottom')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot train/val perplexities for varying amounta of training data.
    fractions = np.arange(0.1, 1.1, 0.1)
    train_perplexities_fraction = []
    val_perplexities_fraction = []

    for fraction in fractions:
        num_samples = int(fraction * len(lm.train_tokens))
        lm.train_tokens = lm.train_tokens[:num_samples]

        train_perplexity = lm.compute_perplexity(lm.train_tokens)
        val_perplexity = lm.compute_perplexity(lm.val_tokens)

        train_perplexities_fraction.append(train_perplexity)
        val_perplexities_fraction.append(val_perplexity)

    plt.figure(figsize=(10, 5))
    plt.plot(fractions, train_perplexities_fraction, marker='o', label='Training Perplexity')
    plt.plot(fractions, val_perplexities_fraction, marker='o', label='Validation Perplexity')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexities vs. Fraction of Training Data')
    plt.legend()

    for (fraction, train_ppl, val_ppl) in zip(fractions, train_perplexities_fraction, val_perplexities_fraction):
        plt.text(fraction, train_ppl, f'({fraction:.2f}, {train_ppl:.2f})', ha='right', va='bottom', fontsize=5)
        plt.text(fraction, val_ppl, f'({fraction:.2f}, {val_ppl:.2f})', ha='right', va='bottom', fontsize=5)

    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # Generate frankenstein text using bigrams
    # run with python lm.py --train_file frankenstein.txt --frankenstein_file frankenstein.txt
    generated_text_bigram = lm.generate_text(n=2, length=100)
    print("Generated Text (Bigram):", ' '.join(generated_text_bigram))

    # Generate frankenstein text using trigrams
    generated_text_trigram = lm.generate_text(n=3, length=100)
    print("Generated Text (Trigram):", ' '.join(generated_text_trigram))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='lm-data/brown-train.txt')
    parser.add_argument('--frankenstein_file', default='frankenstein.txt')
    parser.add_argument('--val_file', default='lm-data/brown-val.txt')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Specify a fraction of training data to use to train the language model.')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Parameter for add-alpha smoothing.')
    parser.add_argument('--show_plot', type=bool, default=False, help='Whether to display the word frequency plot.')

    args = parser.parse_args()
    main(args)
