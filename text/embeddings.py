import operator

import numpy as np

from gensim.models import Word2Vec as w2v
from gensim.models import KeyedVectors

def load_w2v(f, binary=True):
    '''
        Loads a file in the format of the Google pretrained binary.
    '''

    return KeyedVectors.load_word2vec_format(f, binary=binary)

def save_w2v(model, f):
    '''
        Saves a trained word2vec model in KeyedVectors format.
    '''

    model.wv.save(f)

def load_glove(f, k=None):
    '''
        Loads a file in the format provided through the GloVe website.

        k: optional cap on vocabulary size.
    '''

    embeddings = {}

    lines = open(f, 'r').readlines()

    if k != None:
        lines = lines[:k]
    
    for line in lines:
        # assume words are sorted by frequency
        splits = line.split()

        word = splits[0]
        vec = np.array([float(x) for x in splits[1:]])

        embeddings[word] = vec

    return embeddings

def train_w2v_on_file(f, tokenizer='nltk',
                      sz=100, win=5, mc=2, wk=4,
                      save_file='trained.model'):
    '''
        Trains a gensim word2vec model on the contents of a file. Assumes one
        sentence per line. Basically a wrapper for train_w2v().
    '''

    sentences = []

    for line in open(f, 'r').readlines():
        # split up lines... somehow
        if tokenizer == 'nltk':
            sentence = nltk.word_tokenize(line)
        elif tokenizer == 'spacy':
            # TODO
            pass
        else:
            sentence = line.split()

        sentences.append(sentence)

    return train_w2v(sentences, tokenizer=tokenizer,
                     sz=sz, win=win, mc=mc, wk=wk,
                     save_file=save_file)

def train_w2v(sentences, tokenizer='nltk',
              sz=100, win=5, mc=2, wk=4,
              save_file='trained.model'):
    '''
        Trains a gensim word2vec model on the given collection of sentences.
    '''

    # train the model
    model = w2v(sentences, size=sz, window=win, min_count=mc, workers=wk)

    if save_file:
        model.save(save_file)

    return model

def find_top_k(v, k):
    '''
        Returns top k words and top k vectors of a word2vec model, the latter
        as rows of a numpy matrix.
    '''

    counts = {}

    for word, vo in v.vocab.items():
        counts[word] = vo.count

    counts = sorted(counts.items(), key=operator.itemgetter(1))
    #print counts[-10:]

    words = [c[0] for c in counts[-k:]]

    vectors = np.matrix([v[w] for w in words])

    return words, vectors, [c[1] for c in counts[-k:]]
    
