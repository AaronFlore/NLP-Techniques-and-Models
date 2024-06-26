"""
Description: Loads the data and creates mappings.
Authors: Chen & Narasimhan
Documentation: Dr. Korpusik
Date: 6/29/2020
"""

import os
import re
import codecs


def create_dico(item_list):
    """
    Create a frequency dictionary.

    Parameters
    --------------------
        item_list    -- list of lists of items

    Returns
    --------------------
        dictionary mapping items to frequencies

    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Item IDs are ordered by decreasing frequency.

    Parameters
    --------------------
        dico    -- frequency dictionary

    Returns
    --------------------
        tuple of two dictionaries:
            one maps items to IDs
            second maps IDs to items

    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items) if v[1] > 2}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.

    Parameters
    --------------------
        s    -- input string

    Returns
    --------------------
        same string, but with every digit replaced by a zero

    """
    return re.sub('\d', '0', s)


def load_sentences(path, lower, zeros=True):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.

    Parameters
    --------------------
        lower    -- boolean indicating whether to lowercase
        zeros    -- boolean indicating whether to replace digits with zeros

    Returns
    --------------------
        list of sentences (i.e., lists of words)

    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.

    Parameters
    --------------------
        sentences -- list of sentences (i.e., lists of words)
        lower     -- boolean indicating whether to lowercase

    Returns
    --------------------
        tuple of three dictionaries:
            one maps words to frequencies
            second maps words to IDs
            third maps IDs to words

    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words))
    )
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.

    Parameters
    --------------------
        sentences -- list of sentences (i.e., lists of words)

    Returns
    --------------------
        tuple of three dictionaries:
            one maps tags to frequencies
            second maps tags to IDs
            third maps IDs to tags

    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag



def prepare_sentence(str_words, word_to_id, lower=False):
    """
    Prepare a sentence for evaluation.

    Parameters
    --------------------
        str_words  -- list of words
        word_to_id -- dictionary mapping words to IDs
        lower      -- boolean indicating whether to lowercase

    Returns
    --------------------
        dictionary containing two keys:
            'str_words' -- list of words
            'words'     -- list of word IDs (unseen words map to <UNK>)

    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
    }


def prepare_dataset(sentences, mode=None, lower=False, word_to_id=None, tag_to_id=None):
    """
    Prepare the dataset.

    Parameters
    --------------------
        sentences  -- list of sentences (i.e., lists of words)
        mode       -- 'train' or 'test'
        lower      -- boolean indicating whether to lowercase
        word_to_id -- dictionary mapping words to IDs
        tag_to_id  -- dictionary mapping tags to IDs

    Returns
    --------------------
        tuple containing (1) a list of dictionaries with the following keys:
            'str_words' -- list of words
            'words'     -- list of word IDs
            'tags'      -- list of tag IDs

        and (2) a dictionary with the following keys:
            'word_to_id'
            'id_to_word'
            'tag_to_id'
            'id_to_tag'

    """
    assert mode == 'train' or (mode == 'test' and word_to_id and tag_to_id)

    if mode=='train':
        word_dic, word_to_id, id_to_word = word_mapping(sentences, lower)
        tag_dic, tag_to_id, id_to_tag = tag_mapping(sentences)

    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'tags': tags,
        })

    if mode == 'train':
        return data, {'word_to_id':word_to_id, 'id_to_word':id_to_word, 'tag_to_id':tag_to_id, 'id_to_tag':id_to_tag}
    else:
        return data
