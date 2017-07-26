# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class SimpleAttn(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SimpleAttn, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        self.v_q = self.add_weight(name='attention_weight',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)

        super(SimpleAttn, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
