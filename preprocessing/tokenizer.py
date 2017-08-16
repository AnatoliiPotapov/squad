#!/usr/bin/env python3

"""Simple wrapper around the Stanford CoreNLP pipeline.
Serves commands to a java subprocess running the jar. Requires java 8.
"""

import os
import copy
import json
import pexpect
import argparse
import multiprocessing
from multiprocessing import Pool

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Tokenizer(object):
    def __init__(self, cpus, annotators, corenlp_classpath):
        self.cpus = cpus
        self.annotations = annotators
        self.classpath = corenlp_classpath

    def worker(self, arr):
        t = CoreNLPTokenizer(classpath=self.classpath, annotators=self.annotations)
        return [t.tokenize(sample) for sample in arr]

    def tokenize(self, arr):
        if len(arr) < 10000:
            return [self.worker(arr)]
        else:
            chunked = chunks(arr, round(len(arr) / self.cpus))
        p = Pool(self.cpus)
        nested_list = p.map(self.worker, chunked)
        return [val for sublist in nested_list for val in sublist]

class CoreNLPTokenizer(object):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            classpath: Path to the corenlp directory of jars
            mem: Java heap memory
        """
        self.classpath = kwargs.get('classpath')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.mem = kwargs.get('mem', '2g')
        self._launch()

    def _launch(self):
        """Start the CoreNLP jar with pexpect."""
        annotators = ['tokenize', 'ssplit']
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete',
                            'invertible=true'])
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath,
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
               annotators, '-tokenize.options', options,
               '-outputFormat', 'json', '-prettyPrint', 'false']

        # We use pexpect to keep the subprocess alive and feed it commands.
        # Because we don't want to get hit by the max terminal buffer size,
        # we turn off canonical input processing to have unlimited bytes.
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing
        # the NLP> prompt. Hacky!
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return data

        # Minor cleanup before tokenizing.
        clean_text = text.replace('\n', ' ')

        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{"sentences":')
        output = json.loads(output[start:].decode('utf-8'))

        data = []
        token_arr = []
        offset_arr = []

        tokens = [t for s in output['sentences'] for t in s['tokens']]
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i]['characterOffsetBegin']
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1]['characterOffsetBegin']
            else:
                end_ws = tokens[i]['characterOffsetEnd']

            data.append((
                self._convert(tokens[i]['word']),
                text[start_ws: end_ws],
                (tokens[i]['characterOffsetBegin'],
                 tokens[i]['characterOffsetEnd']),
                tokens[i].get('pos', None),
                tokens[i].get('lemma', None),
                tokens[i].get('ner', None)
            ))


        return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotators', default='', help='List of annotators for CoreNLP', type=str)
    parser.add_argument('--load_data', default=None, help='Load data from .json file', type=str)
    parser.add_argument('--outfile', default=None, help='Path to save .json file', type=str)
    args = parser.parse_args()

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default

    try:
        corenlp_path = os.environ["CORENLP_CLASSPATH"]
    except KeyError:
        print("$CORENLP_CLASSPATH not found, using default.")
        corenlp_path = '/home/anatoly/stanford-corenlp-full-2017-06-09/*'

    try:
        with open(args.load_data) as fd:
            test_data = json.load(fd)
    except:
        print('Using example data')
        test_data = [{
            'id':1,
            'question': 'How are U?',
            'answer':   'Well',
            'answer_start': 2,
            'answer_end':   3,
            'context': 'I am always well.',
            'topic': 'general'}]

    questions = [sample['question'] for sample in test_data]
    contexts = [sample['context'] for sample in test_data]

    t = Tokenizer(cpus, annotators=args.annotators, corenlp_classpath=corenlp_path)

    tokenized_questions = t.tokenize(questions)
    tokenized_contexts = t.tokenize(contexts)

    for i, sample in enumerate(test_data):
        sample['question_tokens'] = tokenized_questions[i]
        sample['context_tokens'] = tokenized_questions[i]

    if not args.outfile==None:
        with open(args.outfile, 'w') as fd:
            json.dump(test_data, fd)
    else:
        print(test_data)



