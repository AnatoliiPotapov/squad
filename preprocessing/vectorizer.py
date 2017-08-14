#!/usr/bin/env python3

"""Class witch adds additional word-in-question feature and converts pos, ner, etc.
to one-hot, and embed word using fixed word embeddings.
"""

import numpy as np
from gensim.models import KeyedVectors

def word2vec(word2vec_path):
    model = KeyedVectors.load_word2vec_format(word2vec_path)

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return np.zeros(model.vector_size)

    return get_word_vector

class Vectorizer(object):

    def __init__(self, data):
        pass

    def vectorize(self, use = 'pos, ner, lemma', w2v_path = '../data/'):
        pass
