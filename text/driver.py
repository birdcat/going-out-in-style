import numpy as np
import pickle

from gensim.models import Word2Vec

from embeddings import load_w2v, save_w2v, load_glove, train_w2v, find_top_k, get_vectors
from align import gw, align_map
from preprocess import compile_shakespeare

k = 10000

print 'getting models.'

#sentences = compile_shakespeare()
#shakespeare_model = train_w2v(sentences, sz=300)
#shakespeare_model.save('../models/shakespeare.model')

#shakespeare_model = load_w2v('../models/shakespeare.model')
#shakespeare_model = Word2Vec.load('../models/shakespeare.model')
#shakespeare_model = shakespeare_model.wv

s_model = load_glove('../embeddings/glove/glove.twitter.27B.100d.txt', k=k)

#english_model = load_w2v('../../waypoint/GoogleNews-vectors-negative300.bin', binary=True)
english_model = load_glove('../embeddings/glove/glove.6B.100d.txt', k=k)

print 'models done.'

#print len(shakespeare_model.vocab.keys())
#print shakespeare_model.vocab.keys()

#swords, svectors, sc = find_top_k(shakespeare_model, 10000)
#ewords, evectors, ec = find_top_k(english_model, 10000)

swords, svectors = get_vectors(s_model)
ewords, evectors = get_vectors(english_model)

#sc = np.array(sc)
#ec = np.array(ec)

#print sc[:10]
#print ec[:10]

#sc = sc / np.sum(sc)
#ec = ec / np.sum(ec)

# use uniform distributions for glove (paper recommends zipf-type distribution
# but this should be fine for now)

sc = np.ones(k) / k
ec = np.ones(k) / k

print 'trimming done.'

G, P = gw(evectors, svectors, ec, sc, l=.01)

print 'alignment done.'

pickle.dump(G, open('gamma.pkl', 'w'))
pickle.dump(P, open('P.pkl', 'w'))

print 'saved.'

print align_map(english_model['hello'], shakespeare_model, G, P)
