from utils.hyperparameters import *
from configuration.config import *

from abc import ABC, abstractmethod

class QNetwork(ABC):
    # The Action Scorer - Abstract Class - of Focused Crawler

    @abstractmethod
    def predict(self, action):
        ''' 
            Method for calculating the Q-value of given state-action

            Parameters:
                actions:   StateAction
        '''
        pass

    @abstractmethod
    def fit(self, X, y):
        ''' 
            Training the model
        '''
        pass

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from keras.models import load_model

class ActionScorerBaseline(QNetwork):

    def __init__(self, input_dim=INPUT_DIM, lr=LEARNING_RATE):
        self.input_dim = input_dim
        self.lr = lr
        self.lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return

    def build(self):
        ''' 
            Initialize model
        '''
        self.model = tf.keras.Sequential([
            Input((self.input_dim,)),
            Dense(30, activation=self.lrelu),
            Dense(15, activation=self.lrelu),
            Dense(1)
        ])
        self.model.compile(loss='mse', optimizer=OPTIMIZER)
        return self.model
    
    def fit(self, train_ds, epochs=EPOCHS):
        ''' 
            Training model
        '''
        history = self.model.fit(train_ds, epochs=epochs, verbose=QNETWORK_VERBOSE_TRAIN)
        return history

    def predict(self, action):
        '''
            Predict the Q-value of the given (state-)action
        '''
        return self.model(action)

