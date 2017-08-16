# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import argparse
import requests

import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from models.fastqa import FastQA
from preprocessing.batch_generator import BatchGen, load_dataset

import sys
sys.setrecursionlimit(100000)

np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='fastqa', help='Model to evaluate', type=str)
parser.add_argument('--hdim', default=300, help='Model to evaluate', type=int)
parser.add_argument('--batch_size', default=64, help='Batch size', type=int)
parser.add_argument('--nb_epochs', default=50, help='Number of Epochs', type=int)
parser.add_argument('--optimizer', default='Adam', help='Optimizer', type=str)
parser.add_argument('--lr', default=0.001, help='Learning rate', type=float)
parser.add_argument('--name', default='', help='Model dump name prefix', type=str)
parser.add_argument('--loss', default='categorical_crossentropy', help='Loss', type=str)

parser.add_argument('--dropout', default=0, type=float)

parser.add_argument('--train_data', default='data/train_data.pkl', help='Train Set', type=str)
parser.add_argument('--valid_data', default='data/valid_data.pkl', help='Validation Set', type=str)

# parser.add_argument('model', help='Model to evaluate', type=str)
args = parser.parse_args()

print('Creating the model...', end='')
model = FastQA(hdim=args.hdim, dropout_rate=args.dropout, N=300, M=30)
print('Done!')

print('Compiling Keras model...', end='')
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr} if args.lr else {}}
model.compile(optimizer=optimizer_config,
              loss=args.loss,
              metrics=['accuracy'])
print('Done!')

print('Loading datasets...', end='')
train_data = load_dataset(args.train_data)
valid_data = load_dataset(args.valid_data)
print('Done!')

print('Preparing generators...', end='')
train_data_gen = BatchGen(*train_data, batch_size=args.batch_size, shuffle=False, group=True, maxlen=[300, 30])
valid_data_gen = BatchGen(*valid_data, batch_size=args.batch_size, shuffle=False, group=True, maxlen=[300, 30])
print('Done!')

print('Training...', end='')

path = 'checkpoints/' + args.name + '{epoch}-t{loss}-v{val_loss}.model'


class NBatchLogger(Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            # you can access loss, accuracy in self.params['metrics']
            print(logs)


def OffitialEvaluator(object):
    def __init__(self):
        pass

class PastalogLogger(Callback):
    def __init__(self, display, service_ip, model_name = 'FastQa', log = 'acc'):
        self.seen = 0
        self.display = display
        self.ip = service_ip
        self.model_name = model_name
        self.log = log

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            payload = {"modelName": self.model_name,
                       "pointType": self.log,
                       "pointValue": logs[self.log],
                       "globalStep": self.display}

            r = requests.post(self.url, json=payload)

    def on_epoch_end(self, epoch, logs=None):
        pass





model.fit_generator(generator=train_data_gen,
                    steps_per_epoch=train_data_gen.steps(),
                    validation_data=valid_data_gen,
                    validation_steps=valid_data_gen.steps(),
                    epochs=args.nb_epochs,
                    callbacks=[
                        ModelCheckpoint(path, verbose=1, save_best_only=True),
                        NBatchLogger(10),
                        PastalogLogger(10, 'http://34.232.73.156:8120/', log='loss')
                    ])
print('Done!')
