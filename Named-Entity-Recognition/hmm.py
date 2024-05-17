"""
Description: HMM with greedy and Viterbi decoding for named entity recognition.
Author: Aaron Floreani
Reference: Chen & Narasimhan
Date: 3/26/2024
"""

import argparse
import numpy as np
import loader

from sklearn.metrics import classification_report


class HMM():

    def __init__(self, dics, decode_type):
        """
        Hidden Markov Model (HMM) for named entity recognition.

        Parameters
        --------------------
            dics        -- contains 4 dicts mapping word/tags to/from indices:
                            'word_to_id'
                            'id_to_word'
                            'tag_to_id'
                            'id_to_tag'
            decode_type -- 'viterbi' or 'greedy'

        Attributes
        --------------------
            num_words -- number of words in the vocabulary
            num_tags -- number of named entity tags
            initial_prob -- 1D np array of start probabilities of each tag
            transition_prob -- 2D np array of transition probabilities
            emission_prob -- 2D np array of emission probabilities (tag x word)
            decode_type -- 'viterbi' or 'greedy'
        """
        self.num_words = len(dics['word_to_id'])
        self.num_tags = len(dics['tag_to_id'])

        # Initialize all start, emission, and transition probabilities to 1.
        self.initial_prob = np.ones([self.num_tags])
        self.transition_prob = np.ones([self.num_tags, self.num_tags])
        self.emission_prob = np.ones([self.num_tags, self.num_words])
        self.decode_type = decode_type

    def train(self, corpus):
        """
        Train a bigram HMM model using MLE estimates.
        Update self.initial_prob, self.transition_prob, & self.emission_prob.

        Parameters
        --------------------
            corpus -- a list of dictionaries of the form:
                      {'str_words': str_words,   # List of string words
                       'words': words,           # List of word IDs
                       'tags': tags}             # List of tag IDs

        Each dict's lists all have the same length as that instance's sentence.
        """
        initial_counts = np.zeros(self.num_tags)
        transition_counts = np.zeros((self.num_tags, self.num_tags))
        emission_counts = np.zeros((self.num_tags, self.num_words))
        tag_counts = np.zeros(self.num_tags)

        for sentence in corpus:
            tags = sentence['tags']
            initial_counts[tags[0]] += 1
            for i in range(len(tags)):
                tag = tags[i]
                word = sentence['words'][i]
                emission_counts[tag, word] += 1
                tag_counts[tag] += 1
                if i > 0:
                    prev_tag = tags[i - 1]
                    transition_counts[prev_tag, tag] += 1

        self.initial_prob = initial_counts / np.sum(initial_counts)
        self.transition_prob = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        self.emission_prob = emission_counts / emission_counts.sum(axis=1, keepdims=True)

    def greedy_decode(self, sentence):
        """
        Decode a single sentence in greedy fashion.
        The first step uses initial and emission probabilities per tag.
        Each word after the first uses transition and emission probabilities.

        Parameters
        --------------------
            sentence    -- list of word IDs in the given sentence

        Returns
        --------------------
            list of greedily predicted tags

        """
        tags = []
        for i, word in enumerate(sentence):
            if i == 0:
                scores = self.initial_prob * self.emission_prob[:, word]
            else:
                scores = self.transition_prob[tags[i-1], :] * self.emission_prob[:, word]
            tags.append(np.argmax(scores))
        assert len(tags) == len(sentence)
        return tags

    def viterbi_decode(self, sentence):
        """
        Decode a single sentence using the Viterbi algorithm.

        Parameters
        --------------------
            sentence    -- list of word IDs in the given sentence

        Returns
        --------------------
            list of predicted tags using Viterbi search
        
        """
        V = np.zeros((len(sentence), self.num_tags))
        backpointer = np.zeros((len(sentence), self.num_tags), dtype=int)

        V[0, :] = self.initial_prob * self.emission_prob[:, sentence[0]]

        for t in range(1, len(sentence)):
            for s in range(self.num_tags):
                trans_prob = V[t-1] * self.transition_prob[:, s]
                V[t, s] = np.max(trans_prob) * self.emission_prob[s, sentence[t]]
                backpointer[t, s] = np.argmax(trans_prob)

        tags = []
        last_tag = np.argmax(V[len(sentence)-1])
        tags.append(last_tag)

        for t in range(len(sentence)-1, 0, -1):
            tags.insert(0, backpointer[t, tags[0]])

        assert len(tags) == len(sentence)
        return tags

    def tag(self, sentence):
        """
        Tag a sentence with a trained HMM, using greedy or Viterbi search.

        Parameters
        --------------------
            sentence    -- list of word IDs in the given sentence

        Returns
        --------------------
            list of predicted tags using the specified decode type
        
        """
        if self.decode_type == 'viterbi':
            return self.viterbi_decode(sentence)
        else:
            return self.greedy_decode(sentence)


def evaluate(model, test_corpus, dics, args):
    """
    Predicts test data tags with the trained model, and prints accuracy.

    Parameters
    --------------------
        test_corpus    -- a list of dictionaries of the form:
                          {'str_words': str_words,   # List of string words
                           'words': words,           # List of word IDs
                           'tags': tags}             # List of tag IDs
        dics           -- contains 4 dicts mapping word/tags to/from indices:
                            'word_to_id'
                            'id_to_word'
                            'tag_to_id'
                            'id_to_tag'
        args           -- ArgumentParser args, including:
                          train_file (path to training data, as a string)
                          test_file (path to test data, as a string)
                          lower (boolean indicating whether to lowercase)
                          decode_type ('viterbi' or 'greedy')

    """
    y_pred = []
    y_actual = []
    for i, sentence in enumerate(test_corpus):
        tags = model.tag(sentence['words'])
        y_pred.extend(tags)
        y_actual.extend(sentence['tags'])

    target_names = [dics['id_to_tag'][i] for i in range(len(dics['tag_to_id']))]
    print(classification_report(y_actual, y_pred, target_names=target_names))


def main(args):
    # Load the training data.
    train_sentences = loader.load_sentences(args.train_file, args.lower)
    train_corpus, dics = loader.prepare_dataset(train_sentences, mode='train',
                                                lower=args.lower)

    # Train the HMM.
    model = HMM(dics, decode_type=args.decode_type)
    model.train(train_corpus)

    # Load the validation data for testing.
    test_sentences = loader.load_sentences(args.test_file, args.lower)
    test_corpus = loader.prepare_dataset(test_sentences, mode='test',
                                         lower=args.lower,
                                         word_to_id=dics['word_to_id'],
                                         tag_to_id=dics['tag_to_id'])
    print(dics.keys())
    print(args)

    # Evaluate the model on the validation data.
    evaluate(model, test_corpus, dics, args)

    # Should see 90% accuracy with greedy and 91% with Viterbi.


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='data/eng.train')
    parser.add_argument('--test_file', default='data/eng.val')
    parser.add_argument('--lower', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--decode_type', default='greedy', choices=['viterbi', 'greedy'])

    args = parser.parse_args()
    main(args)
