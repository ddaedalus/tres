import numpy as np
from numpy import zeros, ones, mean
import time
import math
import random

from utils.utils import *
from .textPreprocessor import *
from .webpage import *
from utils.hyperparameters import *
from configuration.config import *

class TextReprGenerator(TextPreprocessor):

    def __init__(self, corpus=None, train_and_save=False, embedding="word2vec", 
                 pre_trained_model="wiki", keyword_filter=None, bins_size=SHORTCUT1, 
                 word_dim=WORD_DIM, maxseqlen=MAXSEQLEN):
        '''
            The Text Representation Generator (TRG) of Focused Crawler

            Parameters:
                corpus:             list of String, should be given if train_and_save=True
                train_and_save:     Boolean, if true then train and save the model on corpus using the given path
                                             else load from path
                embedding:          String, "word2vec" | "Glove"
                pre_trained_model:  String, "bio.nlplab" | "GoogleNews" | "wiki" | "wv-giga",
                keyword_filter:     KeywordFilter

            Important Features from TextPreprocessor class:
                w2v_corpus:         word2vec, trained on corpus (optional)
                w2v:                word2vec, pre-trained       
        '''
        TextPreprocessor.__init__(self, corpus=corpus, embedding=embedding, pre_trained_model=pre_trained_model,
                                  train_and_save=train_and_save, keyword_filter=keyword_filter)
        self.embedding = embedding
        self.pre_trained_model = pre_trained_model
        self.taxonomy_keywords = self.keyword_filter.taxonomy_keywords
        self.new_keywords = self.keyword_filter.new_keywords
        self.taxonomy_phrases = self.keyword_filter.taxonomy_phrases
        self.tax_size = len(self.taxonomy_keywords)
        self.bins_size = bins_size
        self.word_dim = word_dim
        self.maxseqlen = maxseqlen
        try:
            with open('dict_most_similar' + '.pickle', 'rb') as handle:
                self.dict_most_similar = pickle.load(handle)
        except:
            self.dict_most_similar = {}
        print(len(self.dict_most_similar))

        self.d_keywords = {}
        count = 0
        for key in list(self.taxonomy_keywords.keys()):
            self.d_keywords[key] = count
            count += 1 
        for key in list(self.new_keywords.keys()):
            self.d_keywords[key] = count
            count += 1 
        return

    def embeddingLayer(self, doc, padding=True, pooling=""):
        ''' 
            The Word Embeddings Layer for a single document.

            - Parameters:
                - doc:                  list of String
                - padding:              Boolean, True for padding, else False
                - pooling:              String, if "mean" | "max-mean" pooling,
                                                else (eg. "") no pooling option is followed
        
            - Returns:
                - word_embeddings:      array, word embeddings matrix of the given text, 
                                        shape(maxseqlen, word_dim)
        
            Word Embeddings Layer (for each document):
            ------------------------------------------------------------

                word_dim                mean pooling OR max-mean pooling [OR no pooling]
            for pre-trained             word_dim OR
            word embeddings             word_dim * 2
            |  | ... |  |       =>      |  | ... |  |         

        '''
        # Max Sequence Length
        doc_length = len(doc) 
        maxseqlen = doc_length if padding == False else self.maxseqlen        
        found_pretrained = doc_length       # how many words in doc have pre-trained word embeddings

        # Creating Embeddings
        word_embeddings = np.zeros((maxseqlen, self.word_dim))
        i = 0
        for word in doc:
            try:
                # if word is present in pre-trained word embeddings
                self.w2v[self.to_TAG(word)] 
                word_embeddings[i,:] = self.w2v[self.to_TAG(word)]
                i += 1  
            except:
                found_pretrained -= 1
                continue

        # Pooling
        if pooling == "mean":
            # mean pooling
            word_embeddings = ( np.sum(word_embeddings, axis=0) / found_pretrained ) if found_pretrained != 0 else zeros(self.word_size)      
            return word_embeddings      # shape (word_size,)

        if pooling == "max-mean":
            # max-mean pooling (for SVM)
            word_embeddings = np.array([word_embeddings])
            word_embeddings = np.squeeze( array_to_max_mean(word_embeddings) )     # shape (word_size,)
            return word_embeddings

        # No pooling
        return word_embeddings      # shape (maxseqlen, word_size)

    def to_bins(self, count):
        '''
            [0][1][2] ... [>mean_keywords_found]

            Not one-hot. 
            The [0] is 1 only when count=0
            The [1],...,[idx] are 1 when count >= idx 
        '''
        layer = np.zeros(self.bins_size)
        if count == 0: 
            layer[0] = 1
            return layer
        for i in range(1,self.bins_size):
            if count >= i: layer[i] = 1
            else: break
        return layer

    def to_one_hot_bins(self, count):
        '''
            [0][1][2] ... [>mean_keywords_found]

            one-hot. 
            The [0] is 1 only when count=0
            The [1],...,[idx] are 1 when count >= idx 
        '''
        layer = np.zeros(self.bins_size)
        if count == 0: 
            layer[0] = 1
            return layer
        try:
            layer[count] = 1
        except:
            layer[-1] = 1
        return layer

    def keywordOneHotBinsLayer(self, body):
        '''
            Size:   [bins_size]
            Param:
                - body:     String
                - url:      String  
        '''  
        # Search for keywords
        d = self.keyword_filter.find_keywords(body)      # dict[keyword] = times found > 0
        count = 0
        for found in list(d.values()):
            count += found

        # bins layer from body
        layer_bins = self.to_one_hot_bins(count)
        return layer_bins

    def keywordBinsLayer(self, body):
        '''
            Size:   [bins_size + percent_of_keywords_found_in_body + binary_keywords_found_in_url]
            Param:
                - body:     String
        '''  
        # Search for keywords
        d = self.keyword_filter.find_keywords(body)      # dict[keyword] = times found > 0
        count = 0
        for found in list(d.values()):
            count += found

        # bins layer from body
        layer_bins = self.to_bins(count)
        return layer_bins

    def keywordFoundLayer(self, body, url, count=-1):
        '''
             [percent_of_keywords_found_in_body + binary_keyphrases_found_in_body + binary_keywords_found_in_url]
        '''
        # body and URL keywords found
        if count == -1:
            count = 0
            d = self.keyword_filter.find_keywords(body)      # dict[keyword] = times found > 0
            for found in list(d.values()):
                count += found
        body_size = len( self.preprocess_documents(body)[0] )
        if body_size == 0: body_size = 1
        keys_in_url = self.keyword_filter.find_keywords_bin(url)
        found_phrase = self.keyword_filter.find_keyphrases(body)
        layer = np.array([count / body_size, found_phrase, keys_in_url])
        return layer

    def keywordOneHotLayer(self, body):
        '''
            Size:   [one-hot-all-keywords]
            Param:
                - body:      String
        '''
        # Search for keywords
        d = self.keyword_filter.find_keywords(body)      # dict[keyword] = times found > 0
        count_keys = 0
        layer = np.zeros(len(self.new_keywords) + len(self.taxonomy_keywords))
        if len(d) == 0: return layer

        for key in list(d.keys()): 
            idx = self.d_keywords[key]
            layer[idx] = 1  # one-hot

        return layer

    def keywordConcatLayerBinsOneHot(self, doc):
        layer1 = self.keywordOneHotLayer(doc)
        layer2 = self.keywordBinsLayer(doc)
        layer = np.concatenate((layer1, layer2))
        return layer

    def keywordLayer(self, doc):
        '''
            Param:
                - doc:      String
        '''
        # Find doc length
        words = doc.split()
        doc_size = len(words)

        # Search for keywords
        d = self.keyword_filter.find_keywords(doc)      # dict[keyword] = times found > 0
        count_tax = 0
        count_new = 0
        layer = np.zeros(3 * self.tax_size + 2 + 1)

        if len(d) == 0: return layer

        # Search in taxonomy keywords and new keywords
        for i,tax_key in enumerate(list(self.taxonomy_keywords.keys())):
            cos_l = []
            for key in list(d.keys()): 
                cos = self.cos_sim(key, tax_key)
                cos_l.append(cos)
            idx = i * 3
            layer[idx] = np.max(cos_l)
            layer[idx+1] = np.mean(cos_l)
            layer[idx+2] = np.min(cos_l)
        
        # Count taxonomy keywords founds
        for key in list(d.keys()): 
            if key in self.taxonomy_keywords:
                count_tax += d[key]
            else:
                count_new += d[key]
        layer[-3] = np.tanh(count_tax / doc_size)
        layer[-2] = np.tanh(count_new / doc_size)

        # Check whether taxonomy keyphrase found (1 or 0)
        found_keyphrase = self.keyword_filter.find_keyphrases(doc)
        layer[-1] = found_keyphrase
        return layer

    def oversample(self, l_words, threshold=0.55, MAXSEQLEN=MAXSEQLEN):
        '''
            l_words:    list of String
        '''
        new_l = l_words.copy()
        for i,word in enumerate(l_words):
            if i > MAXSEQLEN: break
            if word in self.taxonomy_keywords: continue
            if word in self.new_keywords: continue
            if np.random.rand(1)[0] < threshold: continue
            try:
                if word in self.dict_most_similar: 
                    new_l[i] = self.dict_most_similar[word]
                    continue
                # Get top 3 similar words and pick one at random
                similar_words = trg.w2v.most_similar(trg.to_TAG(word), topn=3)
                random.shuffle(similar_words)
                new_l[i] = similar_words[0][0].split("_")[0]
                self.dict_most_similar[word] = new_l[i]
                with open('dict_most_similar' + '.pickle', 'wb') as handle:
                     pickle.dump(self.dict_most_similar, handle)
            except:
                pass
        return new_l
 
    def create_instance_repr(self, doc, url):
        '''
            Returns a tuple of the representation of the instance that is needed to be fed into clf.
        '''
        doc_l = self.preprocess_documents(doc)[0]
        if len(doc_l) == 0: 
            emb =  -10 * np.ones((1,self.maxseqlen,self.word_dim))
        else:
            emb = self.embeddingLayer(doc_l, pooling="no pooling", padding=True)
            emb = np.reshape(emb, (1,emb.shape[0],emb.shape[1]))
        key1 = self.keywordBinsLayer(doc)
        key1 = np.reshape(key1, (1,-1))
        # Find the number of keywords found in body
        if key1[0,0] == 1: count = 0
        else:
            count = 1
            for i in range(2, len(key1[0])):
                if key1[0,i] == 0: break
                count = i
        key2 = self.keywordFoundLayer(doc, url, count)
        key2 = np.reshape(key2, (1,-1))
        return (emb, key1, key2)
 
