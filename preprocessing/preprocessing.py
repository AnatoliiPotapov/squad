# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _pickle as pickle
import argparse
import json
from collections import Counter
import multiprocessing
from multiprocessing import Pool

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

def word2vec(word2vec_path):
    model = KeyedVectors.load_word2vec_format(word2vec_path)

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return np.zeros(model.vector_size)

    return get_word_vector


class FeatureDict(object):

    def __init__(self):
        try:
            self.load()
        except:
            self.feature_dict = {}

    def add_data(self, data):
        for example in data:
            for token in example['question_tokens']+example['context_tokens']:
                if not (token[3] == None): self.add_feature('pos='+token[3])
                #if not (token[4] == None): self.add_feature(token[4])  # To many lemma features
                if not (token[5] == None): self.add_feature('ner='+token[5])

    def add_feature(self, feature):
        if not self.feature_dict.get(feature):
            self.feature_dict[feature] = len(self.feature_dict)

    def _to_id(self, feature):
        return self.feature_dict[feature]

    def save(self):
        with open('../data/feature_dict.pkl', 'wb') as fd:
            pickle.dump(self.feature_dict, fd)

    def load(self):
        with open('../data/feature_dict.pkl', 'rb') as f:
            self.feature_dict = pickle.load(f, encoding='iso-8859-1')

    def renumerate(self):
        keys = list(self.feature_dict.keys())
        self.feature_dictdict = {}
        for key in keys: self.feature_dictdict[key] = len(self.feature_dict)

class Vectorizer(object):

    def __init__(self, feature_dict, w2v_path, extra = True, use='pos, ner, wiq, tf, is_question', use_qc = (True, False)):
        self.word_vector = word2vec(w2v_path)
        self.dict = FeatureDict()
        self.use = use
        self.extra = extra
        self.use_qc = use_qc

        keys = list(self.dict.feature_dict.keys())

        if not 'pos' in use:
            for key in keys:
                if 'pos' in key:
                    self.dict.feature_dict.pop(key, None)

        if not 'ner' in use:
            for key in keys:
                if 'ner' in key:
                    self.dict.feature_dict.pop(key, None)

        self.dict.renumerate()

        if 'tf' in use:
            self.dict.add_feature('tf')
            self.dict.add_feature('tf_rev')

        if 'wiq' in use:
            self.dict.add_feature('in_question')
            self.dict.add_feature('in_question_uncased')
            self.dict.add_feature('in_question_lemma')


    def extra_features(self, sample):

        context_features = np.zeros(len(sample['context_tokens']), len(self.dict.feature_dict))
        question_features = np.zeros(len(sample['question_tokens']), len(self.dict.feature_dict))

        def wiq(features, question=False):

            if not question:
                q_words_cased = {w for w in sample['question']}
                q_words_uncased = {w.lower() for w in sample['question']}
                q_lemma = {w[4] for w in sample['question_tokens']} if 'lemma' in self.use else None

                for i in range(len(sample['context_tokens'])):
                    if sample['context_tokens'][i][0] in q_words_cased:
                        features[i][self.dict.feature_dict['in_question']] = 1.0
                    if sample['context_tokens'][i][0].lower() in q_words_uncased:
                        features[i][self.dict.feature_dict['in_question_uncased']] = 1.0
                    if q_lemma and sample['context_tokens'][i] in q_lemma:
                        features[i][self.dict.feature_dict['in_question_lemma']] = 1.0

        def pos(features, question=False):
            tokens = 'context_tokens'
            if question:
                tokens = 'question_tokens'
            for i, w in enumerate(sample[tokens]):
                f = 'pos=%s' % w[3]
                if f in self.dict.feature_dict:
                    features[i][self.dict.feature_dict[f]] = 1.0

        def ner(features, question=False):
            tokens = 'context_tokens'
            if question:
                tokens = 'question_tokens'
            for i, w in enumerate(sample[tokens]):
                f = 'pos=%s' % w[5]
                if f in self.dict.feature_dict:
                    features[i][self.dict.feature_dict[f]] = 1.0

        def tf(features, question=False):
            tokens = 'context_tokens'
            if question:
                tokens = 'question_tokens'
            counter = Counter([w[0].lower() for w in sample[tokens]])
            l = len(sample[tokens])
            for i, w in enumerate(sample[tokens]):
                features[i][self.dict.feature_dict['tf']] = counter[w[0].lower()] * 1.0 / l
                features[i][self.dict.feature_dict['tf_rev']] = l / (counter[w[0].lower()] + 1.0)

        if self.use_qc[0]:
            if 'pos' in self.use:
                pos(context_features)
            if 'ner' in self.use:
                ner(context_features)
            if 'tf' in self.use:
                ner(context_features)
            if 'wiq' in self.use:
                wiq(context_features)
        else:
            context_features = None

        if self.use_qc[1]:
            if 'pos' in self.use:
                pos(question_features, True)
            if 'ner' in self.use:
                ner(question_features, True)
            if 'tf' in self.use:
                ner(question_features, True)
            if 'wiq' in self.use:
                wiq(question_features, True)
        else:
            question_features = None


        return [context_features, question_features]

    def to_vector(self, sample, need_answer = True):

        context_vecs = [self.word_vector(token[0]) for token in sample['context_tokens']]
        context_vecs = np.vstack(context_vecs).astype(np.float32)

        question_vecs = [self.word_vector(token[0]) for token in sample['question_tokens']]
        question_vecs = np.vstack(question_vecs).astype(np.float32)


        if self.extra:
            context_extra, question_exta = self.extra_features(sample)
        if self.use_qc[0]:
            context_vecs = np.hstack(context_vecs, context_extra)
        if self.use_qc[1]:
            question_vecs = np.hstack(question_vecs, question_exta)

        if need_answer:

            context_char_offsets = [token[2] for token in sample['context_tokens']]

            try:
                answer_start, answer_end = sample['answer_start'], sample['answer_end']

                answer_start = [answer_start >= s and answer_start < e
                                for s, e in context_char_offsets].index(True)
                answer_end = [answer_end >= s and answer_end < e
                              for s, e in context_char_offsets].index(True)
            except ValueError:
                return None

            return [[context_vecs, question_vecs], [answer_start, answer_end]]

        else:
            return [context_vecs, question_vecs]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Preprocessor(object):

    def __init__(self, w2v_path, use, use_qc, cpus=4, need_answers=True):
        self.vectorizer = Vectorizer(feature_dict=FeatureDict(), w2v_path=w2v_path, extra = True, use=use, use_qc = use_qc)
        self.cpus = cpus

    def worker(self, arr):
        return [self.vectorizer.to_vector(sample) for sample in arr]

    def preprocess(self, samples):

        if len(samples) < 10000:
            return [self.worker(samples)]
        else:
            chunked = chunks(samples, round(len(samples) / self.cpus))
        p = Pool(self.cpus)
        nested_list = p.map(self.worker, chunked)
        samples = [val for sublist in nested_list for val in sublist]

        # Transpose
        data = [[[], []],
                [[], []]]

        for sample in samples[0]:
            data[0][0].append(sample[0][0])
            data[0][1].append(sample[0][1])
            data[1][0].append(sample[1][0])
            data[1][1].append(sample[1][1])

        return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str,
                        default='data/word2vec_from_glove_300.vec',
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/tmp.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('--data', type=str, help='Data json')
    parser.add_argument('--use', default='pos, ner, wiq, tf', help='Which additional features to use', type=str)

    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
        args.outfile += '.pkl'

    print('Reading SQuAD data... ', end='')
    with open(args.data) as fd:
        samples = json.load(fd)
    print('Done!')

    print('Making feature dict... ', end='')
    feature_dict = FeatureDict()
    feature_dict.add_data(samples)
    feature_dict.save()
    print('Done!')

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default

    print('Processing SQuAD data... ', end='')
    prepro = Preprocessor(w2v_path=args.word2vec_path, cpus=cpus, use=args.use, use_qc=(True, True))
    data = prepro.preprocess(samples)
    print('Done!')

    print('Writing to file {}... '.format(args.outfile), end='')
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd)
    print('Done!')




