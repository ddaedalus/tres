import tensorflow as tf
import numpy as np

from utils.hyperparameters import *
from configuration.config import *

class ReplayBuffer:
    # The class of the classic Experience Replay (ER) buffer that is used for a DQN-like focused crawler agent.

    def __init__(self, batch_size=BATCH_SIZE, capacity=BUFFER_CAPACITY):
        '''
            Parameters:
                batch_size:     int
                capacity:       int
        '''
        self.batch_size = batch_size
        self.capacity = capacity
        self.buffer_size = 0
        self.count = 0              # count % capacity; the count idx that a new record would utilize in order to be inserted
        # Replay Buffer content     <x(s,a), a(id), r>
        self.rewards = np.empty(capacity)
        self.actions = np.empty(capacity)
        self.states_actions = np.empty((capacity, INPUT_DIM))
        return

    def size(self):
        '''
            Returns the buffer size
        '''
        return self.buffer_size

    def updateCount(self):
        '''
            Update the count number after each self.insert()
        '''
        self.count = (self.count + 1) % self.capacity
        return 

    def insert(self, record):
        '''
            Inserts a record in the buffer.
        
            Parameters:
                record:     tuple of (StateAction:Array, Action:int, Reward:float)
        '''
        # Get state-action array representation, action num and reward 
        x, action, reward = record
        reward = tf.cast(reward, tf.float32)

        # Increment buffer size counter
        if self.capacity > self.buffer_size: self.buffer_size += 1
        
        # Insert the new record to the corresponding index (self.count) into the arrays
        self.states_actions[self.count] = x
        self.actions[self.count] = action
        self.rewards[self.count] = reward

        # Update the count idx
        self.updateCount()
        return

    def get_next(self):
        '''
            Returns a random batch of records from the buffer.
        
            Returns:
                states_actions:     Array
                actions:            Array
                rewards:            Array
        '''
        max_size = self.buffer_size if self.buffer_size < self.batch_size else self.batch_size
        batch_idxs = np.random.choice(range(self.buffer_size), max_size, replace=False)
        return self.states_actions[batch_idxs], self.actions[batch_idxs], self.rewards[batch_idxs]

    def clear(self):
        '''
            Deletes the records of the buffer.
        '''
        self.batch_size = 0
        self.rewards = np.empty(capacity)
        self.actions = np.empty(capacity)
        self.states_actions = np.empty((capacity, INPUT_DIM))
        return        

