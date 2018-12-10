import pickle
import numpy as np

from gensim.models import Word2Vec

from embeddings import load_w2v, find_top_k
from align import align_map, naive_map

print 'loading maps...'

G = pickle.load(open('gamma.pkl', 'r'))
P = pickle.load(open('P.pkl', 'r'))

print np.shape(P)

print 'loading models...'

shakespeare_model = Word2Vec.load('../models/shakespeare.model')
shakespeare_model = shakespeare_model.wv

english_model = load_w2v('../../waypoint/GoogleNews-vectors-negative300.bin', binary=True)
#english_model = load_glove('../embeddings/glove/glove.6b.300d.txt')

print 'sorting...'

swords, svectors, sc = find_top_k(shakespeare_model, 10000)
#ewords, evectors, ec = find_top_k(english_model, 10000)

print swords[:100]
print ewords[:100]

print swords[np.argmax(G[ewords.index('hello'),:])]

print english_model['a']

print align_map(english_model['hello'].reshape(300, 1), shakespeare_model, G, P)
print naive_map(['hello', 'this', 'is', 'a', 'test'], english_model, shakespeare_model, G, P)
