import os, sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../common')

from keras.layers import Dense, Activation, Dropout, Highway, Input, merge
from keras.optimizers import RMSprop, Adagrad, Adadelta, Adam
from keras.layers.core import Flatten, Reshape, RepeatVector
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Convolution1D
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2, activity_l2
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.utils.data_utils import get_file
from nltk.tokenize import RegexpTokenizer
from keras.engine.topology import Merge
from keras.layers.core import Lambda
from keras.callbacks import History
from gensim.models import word2vec
from collections import Counter
from keras.layers import LSTM
import keras.backend as be
import tensorflow as tf
import numpy as np
import pymystem3
import tarfile
import pickle
import random
import codecs
import uuid
import gzip
import math
import sys
import re
import gc


input_len = 20
emb_model_name = '/var/model-data/word2vec_model-lit-50'

# Load w2v
emb_model = word2vec.Word2Vec.load(emb_model_name)


class MyStemmerExtractor:
    mystem_gr_tokens =         'A|ADV|ADVPRO|ANUM|APRO|COM|CONJ|INTJ|NUM|PART|PR|S|SPRO|V' +         '|наст|непрош|прош' +         '|им|род|дат|вин|твор|пр|парт|местн|зват' +         '|ед|мн' +         '|деепр|инф|прич|изъяв|пов' +         '|кр|полн|притяж' +         '|прев|срав' +         '|1-л|2-л|3-л' +         '|муж|жен|сред' +         '|несов|сов' +         '|пе|нп' +         '|вводн|гео|затр|имя|искаж|мж|обсц|отч|прдк|разг|редк|сокр|устар|фам'

    def __init__(self):
        self.mystem_gr_vocab = self.mystem_gr_tokens.split('|')
        self.mystemmer = pymystem3.Mystem()
        self.mystemmer_cache = {}
        self.mystem_gr_tokenizer = RegexpTokenizer(self.mystem_gr_tokens)
        self.mystem_gr_vectorizer = CountVectorizer(                    tokenizer=self.mystem_gr_tokenizer.tokenize,                     vocabulary=self.mystem_gr_vocab,                     binary=True)

    def get_word_features_vec(self, w):
        if not w in self.mystemmer_cache:
            d = self.mystemmer.analyze(w)
            if len(d) < 1 or not 'analysis' in d[0] or len(d[0]['analysis']) < 1 or not 'gr' in d[0]['analysis'][0]:
                ext_gr_string = ''
                self.mystemmer_cache[w] = [0]*len(self.mystem_gr_vocab)
            else:
                ext_gr_string = d[0]['analysis'][0]['gr']
                self.mystemmer_cache[w] = self.mystem_gr_vectorizer.fit_transform([ext_gr_string]).toarray().tolist()[0] #ext_gr_string
        return self.mystemmer_cache[w]


# **Word char encoder**

# In[11]:

class CharEncoder:
    def __init__(self):
        self.chars = [' ']
        self.chars.extend([chr(c) for c in range(ord('а'), ord('е')+1)])
        self.chars.append('ё')
        self.chars.extend([chr(c) for c in range(ord('ж'), ord('я')+1)])
        self.chars.append('\n')

        self.char2idx = {c: i for i, c in enumerate(self.chars)}

        self.max_wlen = 12

        self.embeddings = np.zeros((len(self.chars), len(self.chars)), dtype=np.float)
        for i, c in enumerate(self.chars):
            self.embeddings[i][i] = 1.

        self._cache = {}

    def encode_chars(self, w):
        if len(w) > self.max_wlen:
            w = w[-self.max_wlen:]
        w = w.rjust(self.max_wlen)

        if w in self._cache:
            return self._cache[w]

        result = [self.char2idx[c] for c in w]
        self._cache[w] = result
        return result


# **A helper function for input sample generation**

# In[12]:

grammar_feature_extractor = MyStemmerExtractor()
char_encoder = CharEncoder()

def prepare_input(word_idx):
    return [
        [word_idx],
        grammar_feature_extractor.get_word_features_vec(idx_to_word[word_idx]),
        char_encoder.encode_chars(idx_to_word[word_idx])
    ]


# load the model
model = load_model('/var/model-data/pushkin-iter-33.h5') #('/var/model-data/onegin-iter-30.h5')

for i, w in enumerate(model.get_weights()):
    print(i, '-', w.shape)
print('There are', sum([np.prod(w.shape) for w in model.get_weights()]), 'weights in the model')


# load the vocab
with open('/var/model-data/pushkin-iter-vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)

idx_to_word = {i: w for i, w in enumerate(vocab)}
word_to_idx = {w: i for i, w in enumerate(vocab)}

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = [p if p > 0.0 else 10**-10 for p in preds]
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# **A helper function that converts a set of word indices into a string of corresponding words**

# In[19]:

def idx_arr2str(idx_arr):
    return ' '.join([idx_to_word[w_idx] for w_idx in idx_arr])


# **Input formatter; used for batch generation and producing test output**

# In[20]:

def extract_input(X, extract_lines=None):
    def flt(a):
        if 1 == len(a):
            return a[0]
        return a
    if None is extract_lines:
        extract_lines = range(len(X))
    input_seq_size = len(X[0])

    batches = [[] for i in range(3)]
    for idx in extract_lines:
        line_inputs = [prepare_input(wi) for wi in X[idx]]
        for i in range(len(batches)):
            batches[i].append( [flt(li[i]) for li in line_inputs] )

    return [np.array(b) for b in batches]


def str2idx(s, l):
    words = [w.lower().replace('ё', 'е') for w in re.split('[^а-яё\n]+', s) if len(w) > 0]
    words_as_idx = [word_to_idx[w] if w in word_to_idx else 0 for w in words]
    #print(l, len(words_as_idx))
    if len(words_as_idx) < l:
        words_as_idx = [0]*(l - len(words_as_idx)) + words_as_idx
    elif len(words_as_idx) > l:
        words_as_idx = words_as_idx[-l:]
    return words_as_idx


def get_poetry(initial_poem):
    sentence = str2idx(initial_poem, input_len)
    result = '' #idx_arr2str(sentence) + ' ===> '

    for i in range(200):
        x = extract_input([sentence])
        preds = model.predict(x, verbose=0)[0]
        generated_w_idx = sample(preds, 1.2)
        result += ' ' + idx_to_word[generated_w_idx]
        sentence = np.append(sentence[1:], generated_w_idx)
    return result


from redisqueue import TextGenQueue
import time
import codecs

q = TextGenQueue('poem', 'redis')

while True:
    task_id, initial_string, reason_string = q.get_next_task()
    if task_id is not None:
        print('working on:', initial_string)
        p = get_poetry(initial_string)
        print('DONE:', p)
        q.store_task_result(task_id, p)
    elif 'stale' == reason_string:
        print('Stale task skipped')
    else:
        #print('sleeping...')
        time.sleep(1)
    #print()
