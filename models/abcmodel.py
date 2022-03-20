from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator, ClassifierMixin
import gc
import math

import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM, Layer, Bidirectional, Masking, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import sparse_categorical_crossentropy
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef as mcc
from keras.models import load_model
from keras.layers.merge import concatenate
from tensorflow.keras.models import Model
import keras
import pickle
import time

from utils.hyperparameters import *
from configuration.config import *
from .preprocess_folds import get_stratified_batches, map_batch, map_batch_get_data, get_batches

import tensorflow as tf


class ABCModel(ABC, BaseEstimator, ClassifierMixin):
    '''
        The abstract class of models that will be used for active learning
    '''
    def __init__(self):
        self.CONSTANT = 0.00000001      # To prevent nan loss

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def save_model(self, path, round=None):
        pass
    
    @abstractmethod
    def load_model(self, path):
        pass


class KwBiLSTM(ABCModel):
    '''     
        Network Architecture:
            - Input:    shape (batch_size, maxseqlen, input_dim)
            - Masking 
            - LSTM
            - Dropout
            - Mean Pooling
            - Dropout
            - Fully-Connected
            - Dropout
            - Fully-Connected
            - Dropout
            - Fully-Connected
            - Dropout
            - Shortcut concatenation (terms / keyword information)
            - Fully-Connected
    '''
    def __init__(self, maxseqlen=MAXSEQLEN, hidden_state_dim=300, input_dim=300, mask=0.0, 
                 shortcut_dim1=SHORTCUT1, shortcut_dim2=SHORTCUT2, output_dim=2, path="", best_score=0.0, save=CLASSIFICATION_MODEL_SAVE):
        '''
            input_dim:          int, size of features of the input of LSTM
            maxseqlen:          int, size of timesteps of the input of LSTM
            hidden_state_dim:   int, size of output dimension of LSTM    
            mask:               int, the mask value in case that we had zero-padded the input earlier
            optimizer:          tensorflow.keras.optimizers
            best_score:         float, for saving the model with a score >= best_score
            save:               bool
        '''
        ABCModel.__init__(self)
        self.best_score = best_score
        self.best_score_fold = 0
        self.path = path
        self.setPathID()    # self.path_id
        self.shortcut_dim1 = shortcut_dim1
        self.save = save

        # Shortcut1
        shortcut1 = Input(shape=(shortcut_dim1,))

        # Shortcut2
        shortcut2 = Input(shape=(shortcut_dim2,))

        # LSTM
        lstm_input_layer = Input(shape=(maxseqlen, input_dim))
        masking_layer = Masking(mask_value=mask, input_shape=(maxseqlen, input_dim))(lstm_input_layer)
        lstm = Bidirectional(LSTM(hidden_state_dim, return_sequences=True,
                             	  dropout=0.1, recurrent_dropout=0.))(masking_layer)
        
        # Mean Pooling
        mean_pooling = GlobalAveragePooling1D(name="mean_pooling")(lstm)
        
        # Fully Connected
        dense = Dropout(0.5)(mean_pooling) 
        dense = Dense(300, activation='relu')(dense)
        dense = Dropout(0.5)(dense) 
        dense = Dense(150, activation='relu')(dense)
        dense = Dropout(0.5)(dense) 
        dense = Dense(85, activation='relu')(dense)
        dense = Dropout(0.5)(dense) 
        dense = Dense(45, activation='relu')(dense)
        dense = Dropout(0.5)(dense)

        # Concat keywordBinsLayer
        concat_shortcut_layer_1 = concatenate([dense, shortcut1], name="concat_shortcut_1")
        concat_shortcut_layer_1 = concatenate([concat_shortcut_layer_1, shortcut2])

        # Fully Connected  
        dense = Dense(50, activation='relu')(concat_shortcut_layer_1)
        dense = Dropout(0.5)(dense)
        dense = Dense(10, activation='relu')(dense)
        dense = Dropout(0.5)(dense)

        # Concat keywordFoundLayer
        concat_shortcut_layer_2 = concatenate([dense, shortcut2], name="concat_shortcut_2")

        # Output layer
        out = Dense(output_dim, activation='softmax', name="output")(concat_shortcut_layer_2)
        self.model = Model([lstm_input_layer, shortcut1, shortcut2], out)
        self.model.compile(
                            optimizer=Adam(learning_rate=0.001), 
                            loss=sparse_categorical_crossentropy, 
                            metrics=["accuracy"]
                           )
        self.model.summary()
        return

    def getBestScore(self):
        return self.best_score

    def getBestScoreFold(self):
        return self.best_score_fold

    def setPathID(self):
        self.path = "./KwBiLSTM/"
        return 

    def score_ds(self, test_ds, report=True):
        '''
            Classification Report and Scoring metrics on test dataset

            Params:
            - test_ds:      tf.data.Dataset
        '''
        y_predict = 0
        y_true = 0
        batches = get_batches(test_ds, batch_size=1024)
        for batch in batches:
            m_batch = map_batch(batch, oversampling=False)   # embedding, keyword, label, url
            X, X_shortcut1, X_shortcut2, y, urls = map_batch_get_data(m_batch)
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            X_shortcut1 = tf.convert_to_tensor(X_shortcut1, dtype=tf.float32)
            X_shortcut2 = tf.convert_to_tensor(X_shortcut2, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.int32)
            true = np.reshape(y, (-1,1))
            try:
                if type(y_true) == int: y_true = true
                else: y_true = np.concatenate((y_true, true))
            except:
                print("Exception happened in score_ds")
                print(y_true, y_true.shape)
                print(true, true.shape)
            pred = np.reshape(self.predict(X, X_shortcut1, X_shortcut2), (-1,1))
            if type(y_predict) == int: y_predict = pred
            else: y_predict = np.concatenate((y_predict, pred))
        if report:
            class_report = classification_report(y_true, y_predict)
            print(class_report)
        else: class_report = None

        # Matthews correlation coefficient             
        mcc_score = mcc(y_true, y_predict) 

        report = classification_report(y_true, y_predict, output_dict=True)
        f1_score = report["0"]["f1-score"]          # only for relevant class -> "0"
        prec_positive = report["0"]["precision"]
        rec_positive = report["0"]["recall"]
        prec_negative = report["1"]["precision"]
        rec_negative = report["1"]["recall"]
        y_true_one_d = np.reshape(y_true, (-1,))

        # Confusion matrix metrics
        tp = rec_positive * len(y_true_one_d[y_true_one_d==0])
        tn = rec_negative * len(y_true_one_d[y_true_one_d==1])
        try: fp = (tp / prec_positive) - tp
        except: fp = 0
        try: fn = (tp / rec_positive) - tp
        except: fn = 0

        # FPR (fall-out), TPR (recall)
        try: fpr = fp / (fp + tn)
        except: fpr = 0 
        tnr = 1 - fpr
        tpr = rec_positive

        # Balanced Accuracy (BA)
        ba = (tpr + tnr) / 2

        print(f"MCC:            {str(mcc_score)[0:5]}\t tp: {tp}")
        print(f"TPR (Recall):   {str(tpr)[0:5]}\t fp: {fp}")
        print(f"FPR (fall-out): {str(fpr)[0:5]}\t tn: {tn}")
        print(f"BA:             {str(ba)[0:5]}\t fn: {fn}")
        return mcc_score, tpr, fpr, ba, class_report

    def __call__(self, X, X_shortcut1, X_shortcut2, batch_size=1):
        '''
            Return the softmax probabilities
        '''
        return self.model.predict([X, X_shortcut1, X_shortcut2], batch_size=batch_size)

    @staticmethod
    def sqrt_class_weight(class_weight):
        '''
            class_weight:   dict
        '''
        for key in list(class_weight.keys()):
            class_weight[key] = math.sqrt( class_weight[key] )
        return class_weight

    def fit(self, X_train, X_shortcut1, X_shortcut2, y_train, class_weight=None, 
            class_weight_method="sqrt", batch_size=32, epochs=1):
        '''
            Training the model
        '''
        if class_weight is not None:
            if class_weight_method == "sqrt":
                class_weight = self.sqrt_class_weight(class_weight)
        # print(class_weight)
        y_train = np.reshape(y_train, (y_train.shape[0],1))
        return self.model.fit([X_train, X_shortcut1, X_shortcut2], y_train, class_weight=class_weight, 
                              batch_size=batch_size, epochs=epochs, shuffle=False, verbose=0)
        

    def fit_ds(self, train_ds, val_ds, class_weight=None, class_weight_method="sqrt", 
               batch_size=512, epochs=1):
        '''
            Training the model

            Params:
            - train_ds:     list
            - val_ds:       list

            Returns:
            - histories:    dict["metric"] = list [metric value]
        '''
        histories = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }   # dict["metric"] = [ metric value ] 
        history_train_loss = [ 0 for epoch in range(epochs) ]
        history_train_acc = [ 0 for epoch in range(epochs) ] 
        history_val_loss = [ 0 for epoch in range(epochs) ]
        history_val_acc = [ 0 for epoch in range(epochs) ] 
        best_report = None
        self.best_score_fold = 0
        for epoch in range(epochs):
            t1 = time.time()
            batches = get_stratified_batches(train_ds, batch_size=batch_size)
            batch_hist_train_loss = []
            batch_hist_train_acc = []
            for batch in batches:
                m_batch = map_batch(batch)   # embedding, keyword, label, url
                X_train, X_shortcut1, X_shortcut2, y, urls = map_batch_get_data(m_batch)
                X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
                X_shortcut1 = tf.convert_to_tensor(X_shortcut1, dtype=tf.float32)
                X_shortcut2 = tf.convert_to_tensor(X_shortcut2, dtype=tf.float32)
                y = tf.convert_to_tensor(y, dtype=tf.int32)
                class_weight = None
                history = self.fit(X_train, X_shortcut1, X_shortcut2, y, 
                                   class_weight=class_weight, batch_size=batch_size, 
                                   class_weight_method=class_weight_method, epochs=1)
                batch_hist_train_loss.append(history.history["loss"])
                batch_hist_train_acc.append(history.history["accuracy"])

            del batches
            del X_train
            del X_shortcut1
            del X_shortcut2
            del y
            gc.collect()

            # Train metrics
            mean_loss_epoch = np.mean(batch_hist_train_loss)
            mean_acc_epoch = np.mean(batch_hist_train_acc)
            history_train_loss.append( mean_loss_epoch )
            history_train_acc.append( mean_acc_epoch )

            print(f"Epoch {epoch+1}:") 
            mcc_score, tpr, fpr, ba, report = self.score_ds(val_ds)
            print(f"train_loss: {mean_loss_epoch}, train_acc: {mean_acc_epoch}")

            if self.best_score_fold < mcc_score:
                best_report = report
                self.best_score_fold = mcc_score

            # save model
            if mcc_score > self.best_score:
                self.best_score = mcc_score
                if self.save:
                    self.save_model()
                    print("Model saved.")
            print()  

            t2 = time.time()
            print(f"Epoch {epoch+1} finished: {(t2-t1)/60} min")
            print()
            
        # Return histories
        histories["train_loss"] = history_train_loss
        histories["train_acc"] = history_train_acc
        histories["val_loss"] = history_val_loss
        histories["val_acc"] = history_val_acc
        return histories, best_report

    def predict(self, X_test, X_shortcut1, X_shortcut2):
        '''
            Predict on X_test
        '''
        tf.compat.v1.reset_default_graph()
        return np.argmax(self.model.predict([X_test, X_shortcut1, X_shortcut2]), axis=-1)     

    def evaluate(self, X_test, X_shortcut1, X_shortcut2, y_true):
        '''
            Evaluate the model
        '''
        y_true = np.reshape(y_true, (y_true.shape[0],1))
        return self.model.evaluate([X_test, X_shortcut1, X_shortcut2], y_true)

    def evaluate_ds(self, test_ds, batch_size=32):
        '''
            Evaluate the model
        
            Params:
            - test_ds:      tf.data.Dataset

            Returns:
            - np.mean(loss)
            - np.mean(accuracy)
        '''
        ds = test_ds.batch(batch_size)
        metrics = []
        for batch in ds.as_numpy_iterator():
            gc.collect()
            X_test = batch[0] + self.CONSTANT
            X_shortcut = batch[1] 
            X_shortcut = X_shortcut.copy() + self.CONSTANT  
            X_shortcut1 = X_shortcut[:,:-self.shortcut_dim2]
            X_shortcut2 = X_shortcut[:,-self.shortcut_dim2:]
            y_true = batch[2]
            y_true = np.reshape(y_true, (y_true.shape[0],1))
            metrics.append( self.model.evaluate([X_test, X_shortcut1, X_shortcut2], y_true) )
        del X_test
        del X_shortcut
        gc.collect()
        losses = []
        accs = []
        for m in metrics:
            loss, acc = m
            losses.append(loss)
            accs.append(acc)
        return np.mean(losses), np.mean(accs)

    def save_model(self, path="model"):
        self.model.save(self.path)
        return

    def load_model(self, path="model"):
        self.model = load_model(self.path + path, compile=False)
        return



