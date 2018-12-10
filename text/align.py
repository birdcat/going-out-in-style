import numpy as np

from numpy import matmul as mm
from numpy import equal as eq
from numpy import transpose, divide, diag, exp, allclose
from numpy.linalg import matrix_power as mp
from numpy.linalg import svd, norm

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy import sparse

def isclose(m, n, tol=0.001):
    '''
        Checks if two matrices are approximately equal. Helper function.
    '''

    return np.sum(np.sum(np.absolute(m - n))) < tol

    #if np.shape(m) != np.shape(n):
    #    return None
    #
    #for i in range(np.shape(m)[0]):
    #    if not allclose(m[i], n[i]):
    #        return False
    #
    #return True

def gw(X, Y, p, q,
       l=.0001, k=None,
       max_outer_iters=10, max_inner_iters=100,
       verbose='debug'):
    '''
        Scale-tolerant GW computation, as described in Alvarez-Melis and
        Jaakkola (2018).

        X: word embedding model for source language.
        Y: word embedding model for target language.
        p: probability distribution over source language.
        q: probability distribution over target language.
        l: regularization parameter.
        k: vocabulary size cap for initial alignment step.
        max_outer_iters, max_inner_iters: caps on maximum outer- and inner-
            loop iterations.
    '''

    # compute cost matrices (cosine distance)
    if k != None:
        swords, sv = find_top_k(X)
        twords, tv = find_top_k(Y)

        # clip and renormalize probability vectors
        pc = p[:k]
        pc = pc/np.sum(pc)
        
        qc = q[:k]
        qc = qc/np.sum(qc)

    else:
        sv = X
        tv = Y

        pc = transpose(np.matrix(p))
        qc = transpose(np.matrix(q))

        k = np.shape(sv)[0]

    #sv = sparse.csr_matrix(sv)
    #tv = sparse.csr_matrix(tv)

    # apparently sklearn cosine_distance doesn't have a dense output option,
    # but cosine_similarity does? -_-
    C_s = cosine_similarity(sv)
    C_t = cosine_similarity(tv)

    if verbose == 'debug':
        #print sv[:10, :10]
        #print tv[:10, :10]
        print C_s[:10, :10]
        print C_t[:10, :10]
        #print pc[:10]
        #print qc[:10]

    print 'a'

    C_st = C_s * C_s * pc

    if verbose == 'debug':
        print C_st[:10, :10]
    
    C_st = C_st * np.matrix(np.ones(k))

    if verbose == 'debug':
        print C_st[:10, :10]
    
    temp = np.ones((k, 1)) * (transpose(qc) * transpose(C_t * C_t))

    if verbose == 'debug':
        print 'second term'
        print temp[:10, :10]
    
    C_st = C_st + temp

    # just initialize it with ones for comparison purposes
    G = np.ones(np.shape(C_st))

    if verbose == 'debug':
        print 'x'
        print C_st[:10, :10]
        print 'y'

        print pc[:10]
        print qc[:10]
    
        print 'z'

    # main loop
    for outer_count in range(max_outer_iters):
        # pseudo-cost matrix
        C_g = C_st - 2 * (C_s * (G * transpose(C_t)))

        if verbose == 'debug':
            print C_st[:10, :10]
            print G[:10, :10]
            print C_g[:10, :10]
            print (-1 * C_g / l)[:10, :10]
    
            print 'b'
            print np.sum(np.sum(C_g))
            print np.sum(np.sum(G))
        
        # Sinkhorn-Knopp iterations
        a = np.ones((k, 1))
        b = np.ones((k, 1))
        K = exp(-1 * C_g / l)

        if verbose == 'debug':
            print K[:10][:10]

            print 'c'
        
        for inner_count in range(max_inner_iters):
            a_old = a
            b_old = b
            
            a = divide(pc, K * b)
            b = divide(qc, transpose(K) * a)

            converged = True

            if isclose(a, a_old) and isclose(b, b_old):
                break

        if verbose == 'debug':
            print 'd'

            print a[:10]
            print b[:10]

        G_old = G
        G = (diag(np.squeeze(np.asarray(a))) * K) * diag(np.squeeze(np.asarray(b)))

        if isclose(G, G_old):
            break

    if verbose == 'debug':
        print 'e'
        print np.shape(G)
        print np.shape(sv)
        print np.shape(tv)
        
    # find projection
    # SVD of (X G Y^T)
    # P = U V^T
    # note that sv, tv are currently n x d whereas we need them to be d x n,
    # so the transposes in this step are reversed
    u, s, vh = svd(transpose(sv) * (G * tv))

    P = u * vh
    
    return G, P

def find_most_similar(v, model, mt='word2vec', dist='cosine', dim=300):
    '''
        In an ideal world, we would just be able to use model.most_similar();
        alas, 'twas not to be.
    '''

    if mt == 'word2vec':
        vocab = model.vocab.keys()

    elif mt == 'glove':
        vocab = model.keys()

    md = 1000
    word = ''

    lv = norm(v)

    print len(vocab)

    print vocab[0]
    print vocab[-1]
    
    for w in vocab:
        if dist == 'cosine':
            d = 1 - (v.reshape(1, dim) * np.array(model[w]).reshape(dim, 1)) / (lv * norm(model[w]))

        elif dist == 'euclidean':
            d = norm(np.array(v) - np.array(vocab[w]))

        if d < md:
            md = d
            word = w

    return w

def align_map(v, model, G, P, dist='cosine', mt='word2vec', dim=300):
    '''
        After the mapping and projection have been computed, applies them to a
        source-language word embedding and returns the nearest word from the
        target model.
    '''

    y = P * v.reshape(dim, 1)

    #print y

    #return model.most_similar(positive=[y], topn=1)
    return find_most_similar(y, model, dist='cosine', mt=mt, dim=dim)

def naive_map(sentence, source, target, G, P, dist='cosine', mt='word2vec', dim=300):
    '''
        Produces a naive (word-by-word) 'translation' from source to target
        language for a given sentence.
    '''

    translated = []
    
    for word in sentence:
        if word in source:
            translated.append(align_map(source[word], target, G, P, dist=dist, mt=mt, dim=dim))
        else:
            translated.append('<UNK>')

    return translated
