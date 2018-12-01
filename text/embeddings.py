import operator
import numpy as np

from gensim.models import Word2Vec as w2v

from numpy import matmul as mm
from numpy import equal as eq
from numpy import transpose, division, diag, exp
from numpy.linalg import matrix_power as mp

from sklearn.metrics.pairwise import cosine_distances
from scipy import sparse

def train_embeddings(f, tokenizer='nltk',
                     sz=100, win=5, mc=2, wk=4,
                     save_file='trained.model'):
    '''
        Trains a gensim word2vec model on the contents of a file. Assumes one
        sentence per line.
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

    # train the model
    model = Word2Vec(sentences, size=sz, window=win, min_count=mc, workers=wk)

    if save_file:
        model.save(save_file)

    return model

def find_top_k(v, k):
    '''
        Returns top k words and top k vectors of a vocabulary, the latter as
        rows of a numpy matrix.

        Assumes v is a gensim word2vec model.
    '''

    counts = {}

    for word, vo in v.vocab_items():
        counts[word] = vo.count

    # items will be (word, 
    counts = sorted(counts.items(), key=operator.itemgetter(1))

    print counts[:10]

    words = [c[0] for c in counts[:k]]

    vectors = np.matrix([v[w] for w in words])

    return words, vectors

def gw(X, Y, λ, k, p, q, max_outer_iters=1000, max_inner_iters=1000):
    '''
        Scale-tolerant GW computation, as described in Alvarez-Melis and
        Jaakkola (2018).

        X: word embedding model for source language.
        Y: word embedding model for target language.
        λ: regularization parameter.
        k: vocabulary size cap for initial alignment step.
        p: probability distribution over source language.
        q: probability distribution over target language.
    '''

    # compute cost matrices (cosine distance)
    swords, sv = find_top_k(X)
    twords, tv = find_top_k(Y)

    sv = sparse.csr_matrix(sv)
    tv = sparse.csr_matrix(tv)

    C_s = cosine_distances(sv, dense_output=True)
    C_t = cosine_distances(tv, dense_output=True)

    # clip and renormalize probability vectors
    pc = p[:k]
    pc = pc/numpy.sum(pc)
    
    qc = q[:k]
    qc = qc/numpy.sum(qc)

    C_st = mp(C_s, 2).dot(pc).dot(np.ones((1, k))) + mm(np.ones(k), mm(qc, transpose(mp(C_t, 2))))

    # main loop
    for outer_count in range(max_outer_iters):
        # pseudo-cost matrix
        pC = C_st - 2 * mm(C_s, mm(Γ, transpose(C_t)))
        
        # Sinkhorn-Knopp iterations
        a = np.ones(k)
        b = np.ones(k)
        K = exp(-1 * pC / λ)
        
        for inner_count in range(max_inner_iters):
            a_old = a
            b_old = b
            
            a = division(pc, mm(K, b))
            b = division(qc, mm(transpose(K), a))

            if eq(a, a_old) and eq(b, b_old):
                # converged
                break

        Γ_old = Γ
        Γ = mm(mm(diag(a), K), diag(b))

        if eq(Γ_old, Γ):
            break

    # find projection
    P = []
    # TODO

    return Γ, P

