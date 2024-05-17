"""
Description: Sentiment Classifier
Date: January 30th, 2024
Author: Aaron Floreani
"""

import argparse
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tqdm import tqdm
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class BaselineClassifier():

    def __init__(self, args):
        """
        A baseline classifier that always predicts positive sentiment.

        Attributes
        --------------------
            train_sents -- list of training sentences as strings
            train_labels -- list of training integer (0 or 1) labels
            val_sents -- list of validation sentences as strings
            val_labels -- list of validation integer (0 or 1) labels

        """
        self.remove_stopwords = args.remove_stopwords  
        self.train_sents, self.train_labels = self.read_data(args.train_file, remove_stopwords=self.remove_stopwords)  
        self.val_sents, self.val_labels = self.read_data(args.val_file, remove_stopwords=self.remove_stopwords)

    def read_data(self, filename, remove_stopwords=False):
        """
        Extracts all the sentences and labels from the input file.

        Parameters
        --------------------
            filename    -- filename as a string

        Returns
        --------------------
            sents       -- list of sentences as strings
            labels      -- list of integer (0 or 1) labels
        """
        sents = []
        labels = []
        with open(filename) as f:
            for line in f.readlines():
                line = line.strip().split(' ', 1)  # only split once
                sentence = line[1]

                if remove_stopwords:
                    stop_words = set(stopwords.words('english'))
                    word_tokens = word_tokenize(sentence)
                    sentence = ' '.join([w for w in word_tokens if not w in stop_words])

                sents.append(sentence)
                labels.append(int(line[0]))
        return sents, labels

    def predict(self, corpus):
        """
        Always predicts a value of 1 given the input corpus.

        Parameters
        --------------------
            corpus    -- list of sentences

        Returns
        --------------------
            list of 1 for each sentence in the corpus
        """

        return [1] * len(corpus)

    def evaluate(self):
        """
        Computes and prints accuracy on training and validation predictions.
        """

        train_predictions = self.predict(self.train_sents)
        val_predictions = self.predict(self.val_sents)

        train_accuracy = np.mean(np.array(train_predictions) == np.array(self.train_labels))
        val_accuracy = np.mean(np.array(val_predictions) == np.array(self.val_labels))

        print(f"Training Accuracy: {train_accuracy:.2%}")
        print(f"Validation Accuracy: {val_accuracy:.2%}")


class NaiveBayesClassifier(BaselineClassifier):

    def __init__(self, args):
        """
        An sklearn Naive Bayes classifier with unigram features.

        Attributes
        --------------------
            train_sents -- list of training sentences as strings
            train_labels -- list of training integer (0 or 1) labels
            val_sents -- list of validation sentences as strings
            val_labels -- list of validation integer (0 or 1) labels
            vectorizer -- sklearn CountVectorizer for training data unigrams
            classifier -- sklearn MultinomialNB classifer object

        """
        super().__init__(args)

        self.vectorizer = CountVectorizer(stop_words='english')
        self.classifier = MultinomialNB()
        self.train()

    def train(self):
        """
        Trains a Naive Bayes classifier on training sentences and labels.
        """

        X_train = self.vectorizer.fit_transform(self.train_sents)
        y_train = np.array(self.train_labels)
        self.classifier.fit(X_train, y_train)

    def predict(self, corpus):
        """
        Predicts labels on the corpus using the trained classifier.

        Parameters
        --------------------
            corpus    -- list of sentences

        Returns
        --------------------
            a list of predictions
        """

        X_corpus = self.vectorizer.transform(corpus)
        predictions = self.classifier.predict(X_corpus)
        return predictions


class LogisticRegressionClassifier(NaiveBayesClassifier):

    def __init__(self, args):
        """
        An sklearn Logistic Regression classifier with unigram features.

        Attributes
        --------------------
            train_sents -- list of training sentences as strings
            train_labels -- list of training integer (0 or 1) labels
            val_sents -- list of validation sentences as strings
            val_labels -- list of validation integer (0 or 1) labels
            vectorizer -- sklearn CountVectorizer for training data unigrams
            classifier -- sklearn LogisticRegression classifer object

        """
        BaselineClassifier.__init__(self, args)

        self.vectorizer = CountVectorizer()
        self.classifier = LogisticRegression(solver=args.solver, penalty=args.penalty, C=args.C)
        self.train()


class BigramLogisticRegressionClassifier(LogisticRegressionClassifier):

    def __init__(self, args):
        """
        A Logistic Regression classifier with unigram and bigram features.

        Attributes
        --------------------
            train_sents -- list of training sentences as strings
            train_labels -- list of training integer (0 or 1) labels
            val_sents -- list of validation sentences as strings
            val_labels -- list of validation integer (0 or 1) labels
            vectorizer -- sklearn CountVectorizer for unigrams and bigrams
            classifier -- sklearn LogisticRegression classifer object

        """
        BaselineClassifier.__init__(self, args)

        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.classifier = LogisticRegression(solver=args.solver, penalty=args.penalty, C=args.C)
        self.train()


def main(args):
    # Evaluate basline classifier (i.e., always predicts positive).
    print("Baseline Evaluation:")
    baseline_classifier = BaselineClassifier(args)
    baseline_classifier.evaluate()

    # Evaluate Naive Bayes classifier with unigram features.
    print("Naive Bayes Evaluation:")
    nb_classifier = NaiveBayesClassifier(args)
    nb_classifier.evaluate()

    # Evaluate logistic regression classifier with unigrams.
    print("Logistic Regression Evaluation:")
    lr_classifier = LogisticRegressionClassifier(args)
    lr_classifier.evaluate()

    # Evaluate logistic regression classifier with unigrams + bigrams.
    print("Bigram Logistic Regression Evaluation:")
    bigram_lr_classifier = BigramLogisticRegressionClassifier(args)
    bigram_lr_classifier.evaluate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # argument to remove stop words
    parser.add_argument('--remove_stopwords', action='store_true', help='Remove stopwords from sentences.')

    parser.add_argument('--train_file', default='sentiment-data/train.txt')
    parser.add_argument('--val_file', default='sentiment-data/val.txt')
    parser.add_argument('--solver', default='liblinear', help='Optimization algorithm.')
    parser.add_argument('--penalty', default='l2', help='Regularization for logistic regression.')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength for logistic regression.')

    args = parser.parse_args()
    main(args)
