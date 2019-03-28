from keras.layers import Dense, Activation, Dropout, Highway
from keras.layers.core import Flatten, Reshape, RepeatVector
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils.data_utils import get_file
from nltk.tokenize import RegexpTokenizer
from keras.engine.topology import Merge
from keras.optimizers import RMSprop
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import LSTM
import numpy as np
import pymystem3
import random
import codecs
import uuid
import sys
import re
import gc


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_lr(fn, default_lr = 0.001):
	lr = default_lr
	try:
		with open(fn, 'r') as f:
			n = f.read().strip()
			lr = float(n)
	except:
		pass
	return lr


text = u''.join(codecs.open('utf8.pushkin.clean.2.0.txt', 'r', 'utf-8').readlines()).lower()
#text = text[:1000]


chars = sorted(list(set(text)))

print(chars)
print('corpus length:', len(text))
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 120
step = 3

sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

mystem_gr_tokens = \
    'A|ADV|ADVPRO|ANUM|APRO|COM|CONJ|INTJ|NUM|PART|PR|S|SPRO|V' + \
    '|наст|непрош|прош' + \
    '|им|род|дат|вин|твор|пр|парт|местн|зват' + \
    '|ед|мн' + \
    '|деепр|инф|прич|изъяв|пов' + \
    '|кр|полн|притяж' + \
    '|прев|срав' + \
    '|1-л|2-л|3-л' + \
    '|муж|жен|сред' + \
    '|несов|сов' + \
    '|пе|нп' + \
    '|вводн|гео|затр|имя|искаж|мж|обсц|отч|прдк|разг|редк|сокр|устар|фам'
mystem_gr_vocab = mystem_gr_tokens.split('|')
mystemmer = pymystem3.Mystem()
mystemmer_cache = {}

def calc_words_metadata(s):
    tokenizer = RegexpTokenizer(u'[а-яё]+')
    
    mystem_gr_tokenizer = RegexpTokenizer(mystem_gr_tokens)
    mystem_gr_vectorizer = CountVectorizer(tokenizer=mystem_gr_tokenizer.tokenize, vocabulary=mystem_gr_vocab, binary=True)
    
    token_coords = tokenizer.span_tokenize(s)
    
    raw_gr_descs = []
    word_indices = []
    for i in token_coords:
        tn = s[i[0]:i[1]]
        if not tn in mystemmer_cache:
            d = mystemmer.analyze(tn)
            if len(d) < 1 or not 'analysis' in d[0] or len(d[0]['analysis']) < 1 or not 'gr' in d[0]['analysis'][0]:
                ext_gr_string = ''
            else:
                ext_gr_string = d[0]['analysis'][0]['gr']
            mystemmer_cache[tn] = ext_gr_string
        gr_string = mystemmer_cache[tn]
        if 0 == len(gr_string):
            continue
        if i[1] < len(s):
            word_indices.append(i[1])
            raw_gr_descs.append(gr_string)
    rows = mystem_gr_vectorizer.fit_transform(raw_gr_descs)
    rows = rows.toarray()
    result = {i[0]: i[1] for i in zip(word_indices, rows)}
    #del mystemmer
    return result

gr_features_size = len(mystem_gr_vocab)

gr_features = calc_words_metadata(text)

print('Creating the dataset...')
X = np.zeros((len(sentences), maxlen, len(chars) + gr_features_size), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    sentence_start_i = i*step
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
        global_char_i = sentence_start_i + t
        if global_char_i in gr_features:
            for j, fv in enumerate(gr_features[global_char_i]):
                X[i, t, len(chars) + j] = fv
    y[i, char_indices[next_chars[i]]] = 1

print('Build model...')

forward = Sequential()
forward.add(LSTM(128, input_shape=(maxlen, len(chars) + gr_features_size)))

backward = Sequential()
backward.add(LSTM(128, input_shape=(maxlen, len(chars) + gr_features_size), go_backwards=True))

model = Sequential()
model.add(Merge([forward, backward], mode='concat'))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = [p if p > 0.0 else 10**-10 for p in preds]
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def get_next_char(prev_char, preds, diversity):
    if None != re.search('[a-zа-яё]', prev_char):
        return '', np.argmax(preds)
    else:
        return '$', sample(preds, temperature=diversity)

def generate_sample(sentence, diversity):
        generated = u''
        sentence = sentence.rjust(maxlen)
        sentence = sentence[-maxlen:]
        generated += sentence
        sys.stdout.write(generated)
        
        prev_char = sentence[-1]
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars) + gr_features_size))
            gr_sentence_features = calc_words_metadata(sentence)
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
                if t in gr_sentence_features:
                    for j, fv in enumerate(gr_sentence_features[t]):
                        x[0, t, len(chars) + j] = fv

            preds = model.predict([x, x], verbose=0)[0]
            marker_char, next_index = get_next_char(prev_char, preds, diversity)
            next_char = indices_char[next_index]
            prev_char = next_char

            generated += next_char
            sentence = sentence[1:] + next_char
            
            if 0 == i:
                sys.stdout.write(' ====>>>> ')
            sys.stdout.write(next_char)
            sys.stdout.flush()

starter_text = 'поэмы не умел писать я никогда\n' + \
               'о, пушкин, брат, мне помоги поэму написать!\n' + \
               'ты лир и вдохновенье призови!\n' + \
               'и выдай на гора стихи!\n'

history = History()
loss_hist = []

for iteration in range(1, 600):
    #lr = read_lr('lr.val');
    #model.optimizer.lr = lr
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit([X, X], y, batch_size=128, nb_epoch=1, verbose=2, callbacks=[history])

    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    print((u'----- Generating with seed: "' + sentence + u'"'))
    print()

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)
        generate_sample(text[start_index: start_index + maxlen], diversity)
        print()
        
    print()
    print('----- custom starter text:')
    generate_sample(starter_text, 0.5)
    print()

    # save the model
    if 1 == iteration or 0 == iteration%10:
        mfn = 'iter-'+str(iteration)+'.h5'
        model.save(mfn)
    print('')
    print('Mystemmer cache size is: ', len(mystemmer_cache))
    
    loss_hist.append(history.history['loss'][0])
    plt.figure(figsize=(20, 6))
    plt.plot(loss_hist)
    plt.savefig('loss.png', dpi=150)
    plt.close()

