import pickle
import numpy as np

from gensim.models import Word2Vec

from embeddings import load_w2v, find_top_k, load_glove, get_vectors
from align import align_map, naive_map

print 'loading maps...'

k = 10000

G = pickle.load(open('gamma.pkl', 'r'))
P = pickle.load(open('P.pkl', 'r'))

print np.shape(P)

print 'loading models...'

s_model = load_glove('../embeddings/glove/glove.twitter.27B.100d.txt', k=k)
english_model = load_glove('../embeddings/glove/glove.6B.100d.txt', k=k)

print 'sorting...'

swords, svectors = get_vectors(s_model)
ewords, evectors = get_vectors(english_model)

print swords[:100]
print ewords[:100]

print swords[np.argmax(G[ewords.index('dog'),:])]

print english_model['a']
print s_model['understand']

print 'Results: '

print align_map(english_model['dog'], s_model, G, P, mt='glove', dim=100)
print naive_map('the limits of my language mean the limits of my world'.split(), english_model, s_model, G, P, mt='glove', dim=100)
print naive_map('the limits of my language mean the limits of my world'.split(), english_model, s_model, G, P, mt='glove', dim=100, dist='euclidean')
