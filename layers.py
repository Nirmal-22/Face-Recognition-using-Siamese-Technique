import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras import backend as K

class L2Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, vects):
        x,y = vects
        sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))