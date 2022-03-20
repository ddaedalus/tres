import gensim
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.metrics.pairwise import cosine_similarity
from numpy import squeeze
import string
import re

from utils.utils import *
from .stopwords import *
from keywords.keyword_filter import KeywordFilter
from utils.hyperparameters import *
from configuration.config import *

class TextPreprocessor:
    # A class for document processing with word2vec embeddings
    def __init__(self, corpus=None, stopwords=stopwords, embedding="word2vec", 
                 pre_trained_model="wiki", train_and_save=False, keyword_filter=KeywordFilter()):
        '''
            corpus:                 list of String, should be given if train_and_save=True
            embedding:              String, "Word2Vec" only available
            pre_trained_model:      String, "bio.nlplab" | "GoogleNews" | "wiki" | "wv-giga", 
                                    the "GoogleNews" option may only require a given corpus  
            train_and_save:         Boolean, if true then train and save the model on corpus using the given path
                                             else load from path
            keyword_filter:         KeywordFilter
        '''
        self.embedding = embedding
        self.stopwords = [".", ",", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "s", "d",
                          "f", "g", "h", "j", "k", "l", "z", "x", "c", "v", "b", "n", "m", 
                          "ii", "iii", "iv"] + stopwords
        self.stopwords = from_list_to_dict(self.stopwords)      # stopwords dict from list
        self.corpus = corpus
        self.pre_trained_model = pre_trained_model
        self.pattern1 = re.compile("[:;\"\'\(\)\*\+\]\[\^%<=>{}$\{!\}@`~\|#&,\/\\\_-]")
        self.pattern2 = re.compile("\d+")
        self.keyword_filter = keyword_filter

        if embedding == "word2vec":
            # Word2Vec embeddings #
            if self.pre_trained_model == "GoogleNews":
                # Google News pretrained model #
                # Load Google's pre-trained word2vec model.
                path_pre_trained = '/content/drive/My Drive/Crawling/GoogleNews-vectors-negative300.bin'
                self.w2v = gensim.models.KeyedVectors.load(path_pre_trained) 

                if train_and_save:
                    # Preprocess corpus
                    self.corpus = self.preprocess_corpus()
                    path = '/content/drive/My Drive/Crawling/GoogleNews_modified.bin'
                    # Train word embeddings on it
                    self.w2v_corpus = self.trainCorpus()  # word embeddings trained on corpus
                    # Save the word embeddings model that was trained on corpus
                    path = get_tmpfile(path)
                    self.w2v_corpus.wv.save(path)
                elif not train_and_save:
                    pass
                    # Load model
                    # self.w2v_corpus = gensim.models.KeyedVectors.load(path, mmap='r')
            
            elif self.pre_trained_model == "bio.nlplab":
                # Biomedical natural language processing pretrained model #
                path_pre_trained = "/content/drive/My Drive/wikipedia-pubmed-and-PMC-w2v.bin"
                self.w2v = gensim.models.KeyedVectors.load_word2vec_format(path_pre_trained, binary=True)

            elif self.pre_trained_model == "wv-giga":
                path_pre_trained = "/content/drive/My Drive/Crawling/wv-giga.bin"
                self.w2v = gensim.models.KeyedVectors.load_word2vec_format(path_pre_trained, binary=True)
    
            elif self.pre_trained_model == "wiki":
                path_pre_trained = "./files/wiki.bin"
                self.w2v = gensim.models.KeyedVectors.load_word2vec_format(path_pre_trained, binary=True)
                
        elif embedding == "glove":
            glove2word2vec(glove_input_file="vectors.txt", word2vec_output_file="gensim_glove_vectors.txt")
            self.w2v = gensim.models.KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
        return

    def preprocess_corpus(self):
        '''
            Preprocesses seed corpus. 
        
            Returns:
                self.corpus:    list of list of String
        '''
        new_corpus = []
        for document in self.corpus: 
            try:
                # Lowercase, filter punctuations, numbers, symbols, repeative spaces and synonyms
                document = re.sub(self.pattern1, "", document.lower())
                document = re.sub(self.pattern2, "", document)
                document = re.sub("\.", "", document)
                document = re.sub("  ", " ", document)

                # Tokenize sentences and transform corpus of sentences
                sentences = sent_tokenize(document)
                for i, sentence in enumerate(sentences):
                    # Remove stopwords
                    sentence = word_tokenize(sentence)
                    sentence = self.removeStopwords(sentence)
                    # Apply synonyms
                    for i, word in enumerate(sentence):
                        try:
                            self.synonyms[word]
                            sentence[i] = self.synonyms[word]
                        except:
                            continue
                    new_corpus.append(sentence)
            except:
                continue

        self.corpus = new_corpus
        return self.corpus

    def removeStopwords(self, sentence):
        '''
            Remove all stopwords from sentence:     list of String

            Parameters:
                sentence:   list of String
        '''
        filtered_sentence = []
        filtered_sentence = [ word for word in sentence if word not in self.stopwords ]
        return filtered_sentence

    def trainCorpus(self):
        '''
            Method for training word embeddings on corpus
        '''
        model = None
        if self.embedding == "word2vec":
            model = Word2Vec(min_count=1, size=300, workers=4, window=5)
            model.build_vocab(self.corpus)
            model.train(self.corpus, total_examples=len(self.corpus), epochs=80) 
        return model

    def find_keywords(self, doc):
        '''
            doc:    String

            Returns a list[binary var, binary var, ...]; it depends on KeywordFilter used.
        '''
        return self.keyword_filter.find_keywords(doc)

    def preprocess_documents(self, docs):
        '''
            Preprocesses documents using only words that have a word embedding.
        
            Parameters:
                docs:   list of String
        
            Returns:
                docs:   list of list of String
        '''
        if type(docs) == str:
            docs = [docs]

        for i,document in enumerate(docs): 
            # Lowercase, filter punctuations, numbers, symbols and repeative spaces
            try:
                document = document.lower()
            except:
                document = default_abstract.lower()
            document = re.sub(self.pattern1, "", document)
            document = re.sub(self.pattern2, "", document)
            document = re.sub("\.", "", document)
            document = re.sub("   ", " ", document)
            document = re.sub("  ", " ", document)

            # Tokenize into words
            words = nltk.word_tokenize(document)

            # Remove stopwords
            words = self.removeStopwords(words)

            # Keep the words that have either/both word embeddings
            filtered_doc = []
            for word in words:
                try:
                    self.w2v[self.to_TAG(word)]
                    filtered_doc.append(word)
                except:
                    try:
                        self.w2v_corpus[self.to_TAG(word)]
                        filtered_doc.append(word)
                    except:
                        continue    
            docs[i] = filtered_doc
        return docs

    def to_TAG(self, w):
        if self.pre_trained_model != "wiki":
            return w
        try:         
            ww = w + "_NOUN"
            self.w2v[ww]
            return ww
        except:      
            try: 
                ww = w + "_ADV"
                self.w2v[ww]
                return ww
            except:
                try:
                    ww = w + "_PROPN"
                    self.w2v[ww]
                    return ww
                except:
                    try:
                        ww = w + "_VERB"
                        self.w2v[ww]
                        return ww
                    except:
                        try:
                            ww = w + "_ADJ"
                            self.w2v[ww]
                            return ww
                        except:
                            return w
        return w

    def hasEmbedding(self, w):
        '''
            Check if word w has a word embedding vector

            w:      String
        '''
        ww = self.to_TAG(w)
        if ww == w: return False
        return True

    def cos_sim(self, w1, w2):
        w1 = self.to_TAG(w1)
        w2 = self.to_TAG(w2)
        try:
            cos = cosine_similarity(np.reshape(normalize(self.w2v[w1]), (1,-1)), 
                                    np.reshape(normalize(self.w2v[w2]), (1,-1)))[0][0]
        except:
            cos = 0
        return cos

