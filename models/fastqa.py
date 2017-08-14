# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from keras import backend as K
import keras
from keras.models import Model
from keras.layers import Input, Dense, RepeatVector, Masking, Dropout, Flatten, Activation, Reshape, Lambda, Permute, merge, multiply, concatenate
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras.layers.pooling import GlobalMaxPooling1D


class FastQA(Model):
    def __init__(self, inputs=None, outputs=None,
                 N=None, M=None, unroll=False,
                 hdim=300, word2vec_dim=300, dropout_rate=0.2,
                 **kwargs):
        # Load model from config
        if inputs is not None and outputs is not None:
            super(FastQA, self).__init__(inputs=inputs,
                                         outputs=outputs,
                                         **kwargs)
            return

        '''Dimensions'''
        B = None
        H = hdim
        W = word2vec_dim

        '''Inputs'''
        P = Input(shape=(N, W), name='P')
        Q = Input(shape=(M, W), name='Q')

        '''Word in question binary'''

        def wiq_feature(P, Q):
            '''
            Binary feature mentioned in the paper.
            For each word in passage returns if that word is present in question.
            '''
            slice = []
            for i in range(N):
                word_sim = K.tf.equal(W, K.tf.reduce_sum(
                    K.tf.cast(K.tf.equal(K.tf.expand_dims(P[:, i, :], 1), Q), K.tf.int32), axis=2))
                question_sim = K.tf.equal(M, K.tf.reduce_sum(K.tf.cast(word_sim, K.tf.int32), axis=1))
                slice.append(K.tf.cast(question_sim, K.tf.float32))

            wiqout = K.tf.expand_dims(K.tf.stack(slice, axis=1), 2)
            return wiqout

        wiq_p = Lambda(lambda arg: wiq_feature(arg[0], arg[1]))([P, Q])
        wiq_q = Lambda(lambda q: K.tf.ones([K.tf.shape(Q)[0], M, 1], dtype=K.tf.float32))(Q)

        passage_input = P
        question_input = Q
        # passage_input = Lambda(lambda arg: concatenate([arg[0], arg[1]], axis=2))([P, wiq_p])
        # question_input = Lambda(lambda arg: concatenate([arg[0], arg[1]], axis=2))([Q, wiq_q])

        '''Encoding'''
        encoder = Bidirectional(LSTM(units=W,
                                     return_sequences=True,
                                     dropout=dropout_rate,
                                     unroll=unroll))

        passage_encoding = passage_input
        passage_encoding = encoder(passage_encoding)
        passage_encoding = TimeDistributed(
            Dense(W,
                  use_bias=False,
                  trainable=True,
                  weights=np.concatenate((np.eye(W), np.eye(W)), axis=1)))(passage_encoding)

        question_encoding = question_input
        question_encoding = encoder(question_encoding)
        question_encoding = TimeDistributed(
            Dense(W,
                  use_bias=False,
                  trainable=True,
                  weights=np.concatenate((np.eye(W), np.eye(W)), axis=1)))(question_encoding)

        '''Attention over question'''
        # compute the importance of each step
        question_attention_vector = TimeDistributed(Dense(1))(question_encoding)
        question_attention_vector = Lambda(lambda q: keras.activations.softmax(q, axis=1))(question_attention_vector)

        # apply the attention
        question_attention_vector = Lambda(lambda q: q[0] * q[1])([question_encoding, question_attention_vector])
        question_attention_vector = Lambda(lambda q: K.sum(q, axis=1))(question_attention_vector)
        question_attention_vector = RepeatVector(N)(question_attention_vector)

        '''Answer span prediction'''

        # Answer start prediction
        answer_start = Lambda(lambda arg:
                              concatenate([arg[0], arg[1], arg[2]]))([
            passage_encoding,
            question_attention_vector,
            multiply([passage_encoding, question_attention_vector])])

        answer_start = TimeDistributed(Dense(W, activation='relu'))(answer_start)
        answer_start = TimeDistributed(Dense(1))(answer_start)
        answer_start = Flatten()(answer_start)
        answer_start = Activation('softmax')(answer_start)

        # Answer end prediction depends on the start prediction
        def s_answer_feature(x):
            maxind = K.argmax(
                x,
                axis=1,
            )
            return maxind

        x = Lambda(lambda x: K.tf.cast(s_answer_feature(x), dtype=K.tf.int32))(answer_start)
        start_feature = Lambda(lambda arg: K.tf.gather_nd(arg[0], K.tf.stack(
            [K.tf.range(K.tf.shape(arg[1])[0]), K.tf.cast(arg[1], K.tf.int32)], axis=1)))([passage_encoding, x])
        start_feature = RepeatVector(N)(start_feature)

        # Answer end prediction
        answer_end = Lambda(lambda arg: concatenate([
            arg[0],
            arg[1],
            arg[2],
            multiply([arg[0], arg[1]]),
            multiply([arg[0], arg[2]])
        ]))([passage_encoding, question_attention_vector, start_feature])

        answer_end = TimeDistributed(Dense(W, activation='relu'))(answer_end)
        answer_end = TimeDistributed(Dense(1))(answer_end)
        answer_end = Flatten()(answer_end)
        answer_end = Activation('softmax')(answer_end)

        input_placeholders = [P, Q]
        inputs = input_placeholders
        outputs = [answer_start, answer_end]

        super(FastQA, self).__init__(inputs=inputs,
                                     outputs=outputs,
                                     **kwargs)

if __name__ == "__main__":
    model = FastQA(hdim=300, N=300, M=30, dropout_rate=0.2)
