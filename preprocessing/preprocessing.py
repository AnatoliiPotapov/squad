# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _pickle as pickle
import argparse
import json
import multiprocessing
from multiprocessing import Pool

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

# from stanfordcorenlp import StanfordCoreNLP
from preprocessing.tokenizer import CoreNLPTokenizer


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str, 
                        default='data/word2vec_from_glove_300.vec',
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/tmp.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('data', type=str, help='Data json')
    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
        args.outfile += '.pkl'

    print('Reading SQuAD data... ', end='')
    with open(args.data) as fd:
        samples = json.load(fd)
    print('Done!')


    print('Tokenizing dataset with CoreNLP using pool of workers')
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default


    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    class Tokenizer(object):
        def __init__(self, cpus):
            self.cpus = cpus

        def worker(self, arr):
            t = CoreNLPTokenizer(classpath='/home/anatoly/stanford-corenlp-full-2017-06-09/*')
            return [t.tokenize(sample) for sample in arr]

        def tokenize(self, arr):
            chunked = chunks(arr, round(len(arr) / self.cpus))
            p = Pool(self.cpus)
            nested_list = p.map(self.worker, chunked)
            return [val for sublist in nested_list for val in sublist]


    t = Tokenizer(cpus)
    context_tokens = t.tokenize([sample['context'] for sample in samples])
    question_tokens = t.tokenize([sample['question'] for sample in samples])
    print('Done!')

    print('Reading word2vec data... ', end='')
    word_vector = word2vec(args.word2vec_path)
    print('Done!')


    def parse_sample(context_t, question_t, context, question, answer_start, answer_end, **kwargs):
        tokens, char_offsets = context_t
        try:
            answer_start = [answer_start >= s and answer_start < e
                            for s, e in char_offsets].index(True)
            answer_end   = [answer_end   >= s and answer_end   < e
                            for s, e in char_offsets].index(True)
        except ValueError:
            return None

        context_vecs = [word_vector(token) for token in tokens]
        context_vecs = np.vstack(context_vecs).astype(np.float32)

        tokens, char_offsets = question_t
        question_vecs = [word_vector(token) for token in tokens]
        question_vecs = np.vstack(question_vecs).astype(np.float32)
        return [[context_vecs, question_vecs],
                [answer_start, answer_end]]

    print('Parsing samples... ', end='')
    samples = [parse_sample(context_tokens[i], question_tokens[i], **sample) for i, sample in tqdm(enumerate(samples))]
    samples = [sample for sample in samples if sample is not None]
    print('Done!')

    # Transpose
    data = [[[], []], 
            [[], []]]
    for sample in samples:
        data[0][0].append(sample[0][0])
        data[0][1].append(sample[0][1])
        data[1][0].append(sample[1][0])
        data[1][1].append(sample[1][1])

    print('Writing to file {}... '.format(args.outfile), end='')
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd)
    print('Done!')
    

