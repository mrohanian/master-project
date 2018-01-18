# coding: utf-8

import pickle, json
import numpy as np 
# fix random seed for reproducibility
np.random.seed(7)
import keras.backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, concatenate, Conv1D, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import MaxPooling1D
from keras import regularizers
import tensorflow as tf

from layers import ChainCRF

### reading train and test
train = pickle.load(open('../stringSequences.pkl', 'rb')) 
X_train = [[x[0].lower() for x in elem] for elem in train]
y_train = [[x[1] for x in elem] for elem in train]

test = pickle.load(open('../testStringSequences.pkl', 'rb'))
X_test = [[x[0].lower() for x in elem] for elem in test]
y_test = [[x[1] for x in elem] for elem in test]
###

words = list(set([elem for sublist in (X_train+X_test) for elem in sublist]))
vocab_size = len(words) + 1
n_classes = len(set([elem for sublist in (y_train+y_test) for elem in sublist])) + 1 # add 1 because of zero padding


"""integer encode the sentences"""
def encode(sents):
    t = Tokenizer(filters='\t\n', lower=False)
    t.fit_on_texts([" ".join(sent) for sent in sents])
    return t.word_index

# assign a unique integer to each word/label 
w2idx = encode(X_train+X_test)
l2idx = encode(y_train+y_test)

# add 0 for zero-padding
# w2idx[0] = 0
l2idx[0] = 0

# keep the reverse to decode back, if necessary
idx2w = {v: k for k, v in w2idx.items()}
idx2l = {v: k for k, v in l2idx.items()}

X_train_enc = [[w2idx[w] for w in sent] for sent in X_train]
X_test_enc = [[w2idx[w] for w in sent] for sent in X_test]

y_train_enc = [[l2idx[l] for l in labels] for labels in y_train]
y_test_enc = [[l2idx[l] for l in labels] for labels in y_test]

# zero-pad all the sequences 
max_length = len(max(X_train+X_test, key=len))

X_train_enc = pad_sequences(X_train_enc, maxlen=max_length, padding='post')
X_test_enc = pad_sequences(X_test_enc, maxlen=max_length, padding='post') 

y_train_enc = pad_sequences(y_train_enc, maxlen=max_length, padding='post')
y_test_enc = pad_sequences(y_test_enc, maxlen=max_length, padding='post')

# one-hot encode the labels 
idx = np.array(list(idx2l.keys()))
vec = to_categorical(idx)
one_hot = dict(zip(idx, vec))
inv_one_hot = {tuple(v): k for k, v in one_hot.items()} # keep the inverse dict

y_train_enc = np.array([[one_hot[l] for l in labels] for labels in y_train_enc])
y_test_enc = np.array([[one_hot[l] for l in labels] for labels in y_test_enc])


########## Access pre-trained embedding for the words list [START] ####
from gensim.models import *

wvmodel = KeyedVectors.load_word2vec_format("../wv/dim300_min50vecs")

embedding_dimension = 300
embedding_matrix = np.zeros((vocab_size, embedding_dimension))

for word, i in w2idx.items():
	try:
		embedding_vector = wvmodel.wv[word]
	except KeyError:
		embedding_vector = None
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector[:embedding_dimension]
 
embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            name='embed_layer')

####### Constructing our own embedding for the words list [END] #######

def model_1():
    visible = Input(shape=(133,))
    embed = embedding_layer(visible)
    # conv2 = Conv1D(200, 2, activation="relu", padding="same", name='conv2', 
    #                     kernel_regularizer=regularizers.l2(0.001))(embed)
    # conv3 = Conv1D(200, 3, activation="relu", padding="same", name='conv3', 
    #                     kernel_regularizer=regularizers.l2(0.001))(embed)
    # conc = concatenate([conv2, conv3])
    lstm = Bidirectional(LSTM(100,return_sequences=True, name='lstm', dropout=0.5))(embed)
    dense = TimeDistributed(Dense(100, activation='relu', name='dense_tanh'))(embed) 
    dense = TimeDistributed(Dense(n_classes, name='dense'))(dense) # This layer should not have any activation. 
    crf = ChainCRF()
    crf_output = crf(dense)
    model = Model(inputs=visible, outputs=crf_output) 
    model.compile(loss=crf.loss, optimizer='adam', metrics=['mae', 'acc'])
    print(model.summary())
    return model

def model_2():
    visible = Input(shape=(133,))
    embed = embedding_layer(visible)
    conv2 = Conv1D(300, 2, activation="relu", padding="same", name='conv2', 
                        kernel_regularizer=regularizers.l2(0.001))(embed)
    lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5))(conv2)
    dense = TimeDistributed(Dense(300, activation="tanh"))(lstm)
    dense = TimeDistributed(Dense(n_classes, name='dense'))(dense) # This layer should not have any activation. 
    crf = ChainCRF()
    crf_output = crf(dense)
    model = Model(inputs=visible, outputs=crf_output) 
    model.compile(loss=crf.loss, optimizer='adam', metrics=['mae', 'acc'])
    print(model.summary())
    return model

def model_3():
    visible = Input(shape=(133,))
    embed = embedding_layer(visible)
    conv2 = Conv1D(300, 2, activation="relu", padding="same", name='conv2', 
                        kernel_regularizer=regularizers.l2(0.001))(embed)
    lstm = Bidirectional(LSTM(100,return_sequences=True, name='lstm', dropout=0.5))(conv2)
    dense = TimeDistributed(Dense(100, activation='relu', name='dense_tanh'))(lstm) 
    dense = TimeDistributed(Dense(n_classes, name='dense'))(dense) # This layer should not have any activation. 
    crf = ChainCRF()
    crf_output = crf(dense)
    model = Model(inputs=visible, outputs=crf_output) 
    model.compile(loss=crf.loss, optimizer='adam', metrics=['mae', 'acc'])
    print(model.summary())
    return model    


model = model_2()


# since we are not using early stopping, we set validation split to 0
model.fit(X_train_enc, y_train_enc, validation_split=0, batch_size=16, epochs=30)

evl = model.evaluate(X_test_enc, y_test_enc, batch_size=16, verbose=1)

# returns loss, means squared error, and accuracy
print('evaluation result:')
print(evl)
print('[loss, accuracy]')
print('####################')

# predict labels for the test data
preds = []
for i in range(len(X_test_enc)):
    pred = model.predict(X_test_enc[i].reshape(1, -1))
    pred = np.argmax(pred,-1)[0] # this turns probabilities into 1, 0. needs to be studied more 
    pred = [idx2l[p] for p in pred]
    preds.append(pred)

# save the predicted labels to a list
with open('test_experimental.pkl', 'wb') as f:
    pickle.dump(preds, f)

#evaluation
import pickle, subprocess
from evaluate import labels2Parsemetsv

with open('test_experimental.pkl', 'rb') as f:
    labels = pickle.load(f)

labels2Parsemetsv(labels, '../ES_PARSEME/test.blind.parsemetsv')
print(subprocess.check_output(["../bin/evaluate.py","../ES_PARSEME/test.parsemetsv", "../system.parsemetsv"]).decode())