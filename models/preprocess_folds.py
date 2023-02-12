from utils.hyperparameters import *
from configuration.config import *
from crawling.textReprGenerator import TextReprGenerator
from keywords.keyword_filter import *
from configuration.taxonomy import taxonomy_keywords, taxonomy_phrases

import pickle
import numpy as np
import gc
import time
import random
import math
random.seed(32)

path = "./files/"

t1 = time.time()
domains = ['Arts', 'Business', 'Computers', 'Health',  'News', 
           'Recreation', 'Reference', 'Science', 'Society', 'Sports']

domain_num = -1
if domain in domains:
    for i,name in enumerate(domains):
        if domain == name: domain_num = i
if domain_num == -1:
    # Not found in domains
    domains.append(domain)
    domain_num = len(domains) - 1

domains_lists_data = []

t1 = time.time()
for i,domain in enumerate(domains): 
    with open(path + f'{domain}.pickle', 'rb') as fp:
        d = pickle.load(fp) 
    fp.close()
    print(f"len {domain}: {len(d)}")
    keys = list(d.keys())[:MAX_LIMIT]
    values = list(d.values())[:MAX_LIMIT]
    data = list(zip(keys,values))
    random.shuffle(data)
    domains_lists_data.append(data)

t2 = time.time()
print(f"{t2-t2} secs")

# Seeds
try:
    with open(path + 'seeds_dict_bodies_' + domain + '.pickle', 'rb') as handle:
        d = pickle.load(handle)
    handle.close()
except:
    d = {}

keys = list(d.keys())
values = list(d.values())
data = list(zip(keys,values))
random.shuffle(data)
domains_lists_data[domain_num] += data
random.shuffle(domains_lists_data[domain_num])

print(f"len total {domain} docs: {len(domains_lists_data[domain_num])}")

### Create folds

X = []
y = []
for i in range(len(domains_lists_data)):
    X += domains_lists_data[i]
    y += [i]*len(domains_lists_data[i])

print(len(X))
print(len(y))
print()

folds = FOLDS
train_folds = [None]*folds
val_folds = [None]*folds

train_folds_labels = [None]*folds
val_folds_labels = [None]*folds

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for j, (train, val) in enumerate(skf.split(X, y)):
    print(len(train))
    print(len(val))
    train_folds[j] = [X[i] for i in train]      # list of list of tuple (url, body)
    val_folds[j] = [X[i] for i in val]          # list of list of tuple (url, body)
    train_folds_labels[j] = [y[i] for i in train]
    val_folds_labels[j] = [y[i] for i in val]
    
y = np.array(train_folds_labels[0])
print(f"Relevant samples: {len(y[y==domain_num])}")
print(f"Irrelevant samples: {len(y[y!=domain_num])}")
print(f"All samples: {len(y)}")

oversample_threshold = OVERSAMPLING_RATIO     # relevant to be oversample_threshold % of irrelevant
rel_fold = len(y[y==domain_num])
irr_fold = len(y[y!=domain_num])

oversample = int(oversample_threshold * irr_fold - rel_fold)
print("If OVERSAMPLING:")
print(f"oversample (add) per fold: {oversample}")
print(f"after oversampling relevant per fold: {oversample + rel_fold}")


### Main Functions
import numpy as np

with open(path + 'new_keywords_' + domain + '.pickle', 'rb') as fp:
    new_keywords = pickle.load(fp) 

keyword_filter = KeywordFilter(taxonomy_keywords=taxonomy_keywords, new_keywords=new_keywords,
                        taxonomy_phrases=taxonomy_phrases)
trg = TextReprGenerator(keyword_filter=keyword_filter)

def parse_element(element, oversampling_flag=True):
    content_data = element[0]
    label = element[1]
    if label == domain_num: label = 0   # relevant
    else: label = 1                     # irrelevant

    url = content_data[0]
    body = content_data[1]
    body_l = trg.preprocess_documents(body)[0]

    if oversampling_flag:
        # Oversampling using word2vec
        body_l = trg.oversample(body_l)

    # Keywords
    keyword_bins_layer = trg.keywordBinsLayer(body)
    keyword_found_layer = trg.keywordFoundLayer(body, url)    

    # Embeddings 
    embedding_layer = trg.embeddingLayer(body_l)
    return embedding_layer, keyword_bins_layer, keyword_found_layer, label, url

def oversampling(ds_l, oversample=oversample):
    '''
        Returns an oversampled fold dataset

        Params:
            -oversample:     int; how many samples to add in relevant dataset
    '''
    rel_ds = [ rec for rec in ds_l if rec[1] == domain_num ]
    irr_ds = [ rec for rec in ds_l if rec[1] != domain_num ]
    rel_size = len(rel_ds)
    oversample_total_size = rel_size + oversample
    oversample_rel_ds = (oversample_total_size // rel_size) * rel_ds
    oversample_rel_ds += rel_ds[: (oversample_total_size % rel_size)]
    ds = oversample_rel_ds + irr_ds
    random.seed(None)
    random.shuffle(ds)
    return ds

def create_list_dataset(fold, train=True):
    if train:
        ds_l = list(zip(train_folds[fold], train_folds_labels[fold])) 
    else:
        ds_l = list(zip(val_folds[fold], val_folds_labels[fold])) 
    return ds_l

def get_batches(ds_l, batch_size=64, shuffle=True):
    '''
        Returns a list (batch) of instances
    '''
    if shuffle:
        random.seed(None)
        random.shuffle(ds_l)
    batches = [ ds_l[i:i+batch_size] for i in range(0, len(ds_l), batch_size) ]
    return batches

def get_stratified_batches(ds_l, batch_size=64, shuffle=True):
    rel_ds = [ rec for rec in ds_l if rec[1] == domain_num ]
    irr_ds = [ rec for rec in ds_l if rec[1] != domain_num ]
    rel_size = len(rel_ds)
    irr_size = len(irr_ds)
    rel_per_batch = math.ceil( ( rel_size / (rel_size + irr_size) ) * batch_size )
    irr_per_batch = batch_size - rel_per_batch
    if shuffle:
        random.seed(None)
        random.shuffle(rel_ds)
        random.shuffle(irr_ds)
    rel_batches = [ rel_ds[i:i+rel_per_batch] for i in range(0, len(rel_ds), rel_per_batch) ]
    irr_batches = [ irr_ds[i:i+irr_per_batch] for i in range(0, len(irr_ds), irr_per_batch) ]
    batches = []
    for i in range(len(rel_batches)):
        l = rel_batches[i] + irr_batches[i]
        random.shuffle(l)
        batches.append(l)
    return batches

def map_batch(batch, oversampling=True):
    '''
        Returns a list (batch) modified by fn: parse_element
    '''
    return list(map(lambda x: parse_element(x, oversampling_flag=oversampling), batch))

def map_batch_get_data(batch):
    def get_X():
        return np.array(list(map(lambda x:x[0], batch)))
    def get_X_keyword_bins():
        return np.array(list(map(lambda x:x[1], batch)))
    def get_X_keyword_found():
        return np.array(list(map(lambda x:x[2], batch)))
    def get_y():
        return np.array(list(map(lambda x:x[3], batch)))
    def getURLs():
        return list(map(lambda x:x[4], batch))
    return get_X(), get_X_keyword_bins(), get_X_keyword_found(), get_y(), getURLs()
    

