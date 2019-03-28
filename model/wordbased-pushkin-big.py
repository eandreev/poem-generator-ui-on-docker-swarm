
# coding: utf-8

# In[1]:

from keras.layers import Dense, Activation, Dropout, Highway, Input, merge
from keras.optimizers import RMSprop, Adagrad, Adadelta, Adam
from keras.layers.core import Flatten, Reshape, RepeatVector
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Convolution1D
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2, activity_l2
from nltk.stem.snowball import SnowballStemmer
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.utils.data_utils import get_file
from keras.utils.visualize_util import plot
from nltk.stem.porter import PorterStemmer
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
import nltk.data
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# **Settings**

# In[2]:

input_len = 20
candidate_out_size = 10
sample_output_len = 100
traing_sample_seq_step = 1
output_vocab_size = 5000
emb_model_name = '/data/word2vec_model-lit-50'
batch_size = 128
draw_loss_graph = True
save_the_model = True


# **Read the corpus and mark up the corpus**

# In[3]:

poems = []
tar = tarfile.open('poetika.tgz', 'r:gz')
for tarinfo in tar:
    if tarinfo.isreg():
        f = tar.extractfile(tarinfo)
        marker = 'a'
        if   None != re.search('poety-puskin'    , tarinfo.name): marker = 'p'
        elif None != re.search('poety-maakovskij', tarinfo.name): marker = 'm'
        poems.append({'marker': marker, 'text': codecs.decode(f.read(), 'utf-8')})
tar.close()

print('Loaded', len(poems), 'individual poems')

print('Markers:')
mcounter = Counter([p['marker'] for p in poems])
for c in mcounter:
    print('   ', c, '-', mcounter[c])

#poems = poems[:200]


# **Build a vocabulary**
# * Count word occurences
# * Sort words by popularity
# * Cerate a vocabulary that
#   * contains a newline as its first element
#   * and all of the words sorted by word count in the descending order

# In[4]:

word_counts = {}
for w in [w for w in re.split('[^а-яё]+', '\n'.join([p['text'] for p in poems]).lower()) if len(w) > 0]:
    word_counts[w] = word_counts.get(w, 0) + 1
word_counts = [(i, word_counts[i]) for i in word_counts]
word_counts = sorted(word_counts, key=lambda p: p[1], reverse=True)


# vocabulary:
# * 0 - newline
# * ... words

# In[5]:

vocab = ['\n']
for p in word_counts:
    vocab.append(p[0])

print('Unique words:', len(vocab))
print('First 100 words:', vocab[1:101])


# **Create word lookup dictionaries**

# In[6]:

idx_to_word = {i: w for i, w in enumerate(vocab)}
word_to_idx = {w: i for i, w in enumerate(vocab)}


# **Split the poetry corpus into lines, words and convert the text into a vector of word indices**

# In[7]:

poems_as_widx_seqs = []
for p in poems:
    pidx = []
    for l in p['text'].split('\n'):
        for w in re.split('[^а-яё]+', l.strip().lower()):
            len_emp = True
            if len(w) > 0:
                pidx.append(word_to_idx[w])
                len_emp = False
        pidx.append(0)
    if len(pidx) > input_len:
        poems_as_widx_seqs.append({'marker': p['marker'], 'text': pidx})

print(len(poems_as_widx_seqs), 'added for use in the training corpus')


# **Load the w2v model**

# In[8]:

emb_model = word2vec.Word2Vec.load(emb_model_name)


# **Create a vocab of word vectors collecting stats about words that are missing in the w2v model along the way**

# In[9]:

w2v_stats = [0, 0]
w2v_missing = []
emb_vocab = [[1.] + [0.]*emb_model.vector_size]
for w in vocab[1:]:
    w = w.replace('ё', 'е')
    if w in emb_model:
        emb_vocab.append([0.] + list(emb_model[w]))
    else:
        emb_vocab.append([0.]*(emb_model.vector_size + 1))
        w2v_missing.append(w)
    w2v_stats[w in emb_model] += 1
print()
print('W2v is missing for', w2v_stats[0], 'out of', sum(w2v_stats), 'words', '/ %.2f%%'%( 100.0*w2v_stats[0]/sum(w2v_stats), ))
print('Some missing words:', ', '.join(w2v_missing[:20]))


# **Grammar data extractor**

# In[10]:


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


# **Prepare the dataset**

# In[13]:

print()
print('Pre-caching the inputs...')
for wi in set([w for p in poems_as_widx_seqs for w in p['text']]):
    prepare_input(wi)


# In[14]:

def prepare_dataset(markers=[]):
    X = []
    y = []
    pos_seq_count = 0
    for p in poems_as_widx_seqs:
        marker = p['marker']
        if 0 == len(markers) or marker in markers:
            for i in range(0, len(p['text']) - input_len, traing_sample_seq_step):
                pos_seq_count += 1
                if p['text'][i + input_len] < output_vocab_size:
                    X.append(p['text'][i: i + input_len])
                    y.append(p['text'][i + input_len])
    return X, y, pos_seq_count


# **Build the model**

# In[15]:

print('Building the model...')

#emb_vocab_tensor = be.variable(emb_vocab)

#def word_idx2embedding(idx):
#    import keras.backend as be
#    return be.gather(emb_vocab_tensor, tf.to_int64(idx))


word_idx_in = Input(shape=(input_len,))
#word_vecs = Lambda(word_idx2embedding,
#                     output_shape=(input_len, emb_model.vector_size+1))(word_idx_in)

word_vecs = Embedding(len(emb_vocab), 50, trainable=True)(word_idx_in)

grammar_features_in = Input(shape=(input_len, 62), name='grammar_features_in') #TODO: len(grammar_feature_extractor...)

clstm_in = Input(shape=(char_encoder.max_wlen,), name='clstm_in')
clstm_embed = Embedding(len(char_encoder.chars), len(char_encoder.chars), weights=[char_encoder.embeddings], trainable=False)(clstm_in)
print('clstm_embed initial -', clstm_embed.get_shape())

filter_length = [5, 3, 3]
nb_filter = [196, 196, 256]
pool_length = 2

for i in range(len(nb_filter)):
    clstm_embed = Convolution1D(nb_filter=nb_filter[i],
                            filter_length=filter_length[i],
                            border_mode='same',
                            activation='relu',
                            init='glorot_normal',
                            subsample_length=1)(clstm_embed)
    print('clstm_embed', i, '-', clstm_embed.get_shape())

    clstm_embed = Dropout(0.1)(clstm_embed)
    #clstm_embed = MaxPooling1D(pool_length=pool_length)(clstm_embed)
    print('clstm_embed max pooled', '-', clstm_embed.get_shape())

c_lstm_1 = LSTM(128, unroll=True, dropout_W=0.2, dropout_U=0.2                   )(clstm_embed)
c_lstm_2 = LSTM(128, unroll=True, dropout_W=0.2, dropout_U=0.2, go_backwards=True)(clstm_embed)
clstm_out = merge([c_lstm_1, c_lstm_2], mode='concat')
c_lstm_model = Model(input=clstm_in, output=clstm_out)

chars_in = Input(shape=(input_len, char_encoder.max_wlen), name='chars_in')
c_lstm_out = TimeDistributed(c_lstm_model)(chars_in)

concatenated_input = merge([word_vecs, grammar_features_in, c_lstm_out], mode='concat')

print('word_vecs:', word_vecs.get_shape())
print('grammar_features_in:', grammar_features_in.get_shape())
print('c_lstm_out:', c_lstm_out.get_shape())
print('concatenated_input:', concatenated_input.get_shape())

lstm = LSTM(128, unroll=True, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(concatenated_input)
lstm = LSTM(128, unroll=True, dropout_W=0.2, dropout_U=0.2)(lstm)

output = Dense(output_vocab_size)(lstm)
output = Activation('softmax')(output)

model = Model(input=[word_idx_in, grammar_features_in, chars_in], output=output)

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#model = load_model('all-iter-saved.h5')
#model.load_weigths('all-iter-saved.h5')


# In[16]:

plot(model, to_file='model.png', show_shapes=True)


# **Output the model stats**

# In[17]:

for i, w in enumerate(model.get_weights()):
    print(i, '-', w.shape)
print('There are', sum([np.prod(w.shape) for w in model.get_weights()]), 'weights in the model')


# **A helper function that picks a word given a word probability vector**

# In[18]:

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


# **Keras batch generator function; it lets us save a lot of memory on one-hot encodings of expected output samples**

# In[21]:

def batch_generator(X, y, batch_size):
    number_of_batches = int(len(X)/batch_size)
    counter=0
    shuffle_index = np.arange(len(X))
    np.random.shuffle(shuffle_index)
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        y_batch = np.zeros((batch_size, output_vocab_size), dtype=np.float)
        for i, wi in enumerate([y[i] for i in index_batch]):
            y_batch[i][wi] = 1.
        counter += 1
        yield extract_input(X, extract_lines=index_batch), y_batch
        if (counter >= number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0


# **Train the model and produce sample output**

# In[22]:

text_as_idx = [ w for p in poems_as_widx_seqs for w in p['text'] + [0] ]


# In[23]:

def training_round(round_name, n_of_epochs, markers = [], context = {'loss_hist': []}):
    print('=' * 50)
    print('Preparing the dataset...')
    X, y, pos_seq_count = prepare_dataset(markers=markers)
    print('Dataset size:', len(X), 'out of', pos_seq_count, 'possible sequences')

    history = History()
    loss_hist = context['loss_hist']

    for iteration in range(1, n_of_epochs + 1):
        print()
        print('-' * 50)
        print('Iteration', iteration, 'of', n_of_epochs, '[' + round_name + ']')
        #model.fit_generator(generator=batch_generator(X, y, batch_size), samples_per_epoch=X.shape[0], nb_epoch=1, verbose=2, callbacks=[history])
        model.fit_generator(generator=batch_generator(X, y, batch_size), samples_per_epoch=len(X), nb_epoch=1, verbose=2, callbacks=[history])

        start_index = random.randint(0, len(text_as_idx) - input_len - 1)
        initial_sentence = text_as_idx[start_index: start_index + input_len]
        print()

        print('---- max:')
        sentence = list(initial_sentence)
        sys.stdout.write(idx_arr2str(sentence) + ' ===> ')
        sys.stdout.flush()
        most_probable = []
        most_probable2 = []
        for i in range(sample_output_len):
            x = extract_input([sentence])
            preds = model.predict(x, verbose=0)[0]
            generated_w_idx = np.argmax(preds)
            sys.stdout.write(' ' + idx_to_word[generated_w_idx])
            sys.stdout.flush()
            sentence = np.append(sentence[1:], generated_w_idx)

            most_probable.append([ (idx_to_word[p[0]], float('%.4f'%(p[1]))) for p in sorted(enumerate(preds), key=lambda p: p[1], reverse=1)[:candidate_out_size] ])
            most_probable2.extend([ idx_to_word[p[0]] for p in sorted(enumerate(preds), key=lambda p: p[1], reverse=1)[:candidate_out_size*4] ])
        print()
        for mp in most_probable:
            print(mp)
        print()
        print(' '.join(most_probable2))

        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('---- temperature: ', temperature)
            sentence = list(initial_sentence)
            sys.stdout.write(idx_arr2str(sentence) + ' ===> ')
            sys.stdout.flush()
            generated = ''
            for i in range(sample_output_len):
                x = extract_input([sentence])
                preds = model.predict(x, verbose=0)[0]
                generated_w_idx = sample(preds, temperature)
                sys.stdout.write(' ' + idx_to_word[generated_w_idx])
                sys.stdout.flush()
                sentence = np.append(sentence[1:], generated_w_idx)
                generated += ' ' + idx_to_word[generated_w_idx]
            print()
            print()

        if draw_loss_graph:
            # draw the loss graph
            loss_hist.append(history.history['loss'][0])
            plt.figure(figsize=(20, 6))
            plt.plot(loss_hist)
            plt.savefig('loss.png', dpi=150)
            plt.close()

        if save_the_model: #1 == iteration or 0 == iteration%100:
            # save the model
            model_files_prefix = round_name + '-iter'
            model.save(model_files_prefix + '.h5')
            with open(model_files_prefix + '-vocab.pickle', "wb" ) as f:
                pickle.dump(vocab, f)
    return {
        'loss_hist': loss_hist
    }


# In[24]:

c = training_round('all', 12)
c = training_round('pushkin', 15, markers=['p'], context=c)

print()
print('DONE')


# In[ ]:




