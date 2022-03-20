from numpy.linalg import norm
import math
import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM, Layer, Masking, Dropout, Concatenate, GlobalAveragePooling1D, GlobalMaxPool1D
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Colours
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def get_files(dir):
    files = _get_files(dir)
    for i,file in enumerate(files):
        files[i] = dir + file
    return files

def _get_files(dir):
    for root, dirs, files in os.walk(dir):
        return files

def normalize(word_vec):
    norm=np.linalg.norm(word_vec)
    if norm == 0: 
       return word_vec
    return word_vec/norm

def interception(l1, l2):
    '''
        Find the interception of two lists
    '''
    return list(set(l1) & set(l2))

def convert_arr_bins(arr):
    '''
        From not one-hot to one-hot encoding 
    '''
    for row in range(len(arr)):
        size = len(arr[row,:])
        for i in range(size)[::-1]:
            if i == 2: break
            if arr[row,i] >= 1:
                try:
                    arr[row,3:i] = 0
                    break
                except:
                    break
    return arr

def from_list_to_dict(l):
    '''
        Given a list, returns a dict[list(element)]=0
    ''' 
    d = {}
    for item in l:
        d[item] = 0
    return d


def inverse_dict(d):
    '''
        Given a dict[key]=item, returns a dict[item]=key
    '''
    inverse_d = {}
    for key in list(d.keys()):
        inverse_d[d[key]] = key
    return inverse_d 


def cosine_similarity(a, b):
    '''
        Performs the cosine similarity between a and b

        - Parameters:
            -a:     array
            -b:     array
    '''
    return ( a @ b.T ) / ( norm(a) * norm(b) )


def find_maxseqlen(l):
    '''
        Given a list l of matrixes find the max length of these matrixes
    '''
    lengths = []
    for matrix in l:
        lengths.append(len(matrix))
    return max(lengths) 


def dict_values_to_numpy(d):
    '''
        Given a dict, cast its values to numpy array
    '''
    values = list(d.values())
    return np.array(values)


def array_decrease_by_unit(arr):
    '''
        Given an array decrease all its elements by 1
    '''
    return arr - 1


def getTrain(X_dataset, y_dataset, idx_dict, label_dict, pool_dict, concat=True):
    '''
        - X_dataset:    array of 49 docs
        - y_dataset:    array of 49 docs
        - idx_dict:     dict, d[idx]=url
        - label_dict:   dict, d[url]=label
        - pool_dict:    dict, d[idx]=array
    '''
    inverse_idx_dict = inverse_dict(idx_dict)       # d[url] = idx
    train = []
    labels = []
    for url in list(label_dict.keys()):
        idx = inverse_idx_dict[url]
        labels.append(label_dict[url])
        train.append(pool_dict[idx])
    X = np.array(train) 
    y = np.array(labels)    
    if concat:
        X = np.concatenate((X_dataset, X))
        y = np.concatenate((y_dataset, y))
    return X,y


def create_re_pattern(s):
    '''
        s:  String
    '''
    return re.compile(s)
    

def array_to_max_mean(arr):
    '''
        - arr:    Array of shape (docs, timesteps, features)
    '''
    _, timesteps, features = arr.shape
    # Max Pooling
    model1_in = Input(shape=(timesteps,features))
    model1 = Masking(mask_value=0.0, input_shape=(timesteps, features))(model1_in)
    model1_out = GlobalMaxPool1D()(model1)
    model1 = Model(model1_in, model1_out)
    model1.compile()
    o1 = model1(arr)
    max_pool = np.array(o1)

    # Mean Pooling
    model1_in = Input(shape=(timesteps,features))
    model1 = Masking(mask_value=0.0, input_shape=(timesteps, features))(model1_in)
    model1_out = GlobalAveragePooling1D()(model1)
    model1 = Model(model1_in, model1_out)
    model1.compile()
    o1 = model1(arr)
    mean_pooling = np.array(o1)

    # Return array of shape (mean+max,)
    return np.concatenate((max_pool, mean_pooling), axis=1)


def quantize(num, quantum_list=[0.00001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    if num == 1.0: return 1.0
    if num == 0.0: return 0.0
    for i in range(len(quantum_list)):
        try:
            if num >= quantum_list[i] and num < quantum_list[i+1]:
                return quantum_list[i]
            else:
                continue
        except:
            return quantum_list[i]

