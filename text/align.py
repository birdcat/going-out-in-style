import numpy as np

from numpy import matmul as mm
from numpy import equal as eq
from numpy import transpose, divide, diag, exp, allclose
from numpy.linalg import matrix_power as mp
from numpy.linalg import svd

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy import sparse

def isclose(m, n, tol=0.001):
    '''
        Checks if two matrices are approximately equal. Helper function.
    '''

    return np.sum(np.sum(m - n)) < tol

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
       max_outer_iters=10, max_inner_iters=100):
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
    C_s = cosine_similarity(sparse.csr_matrix(sv), dense_output=True)
    C_t = cosine_similarity(sparse.csr_matrix(tv), dense_output=True)

    print 'a'

    C_st = mp(C_s, 2) * pc
    C_st = C_st * np.matrix(np.ones(k))
    C_st = C_st + np.ones((k, 1)) * (transpose(qc) * transpose(mp(C_t, 2)))

    # just initialize it with ones for comparison purposes
    G = np.ones(np.shape(C_st))

    # main loop
    for outer_count in range(max_outer_iters):
        # pseudo-cost matrix
        C_g = C_st - 2 * (C_s * (G * transpose(C_t)))

        print 'b'
        
        # Sinkhorn-Knopp iterations
        a = np.ones((k, 1))
        b = np.ones((k, 1))
        K = exp(-1 * C_g / l)
        
        for inner_count in range(max_inner_iters):
            a_old = a
            b_old = b
            
            a = divide(pc, K * b)
            b = divide(qc, transpose(K) * a)

            converged = True

            if isclose(a, a_old) and isclose(b, b_old):
                break

        print 'c'

        G_old = G
        G = (diag(np.squeeze(np.asarray(a))) * K) * diag(np.squeeze(np.asarray(b)))

        if isclose(G, G_old):
            break

    print 'd'
    print np.shape(G)
    print np.shape(sv)
    print np.shape(tv)
        
    # find projection
    # SVD of (X G Y^T)
    # P = U V^T
    u, s, vh = svd(sv * (G * transpose(tv)))

    P = u * vh
    
    return G, P

def align_map(v, model, G, P):
    '''
        After the mapping and projection have been computed, applies them to a
        source-language word embedding and returns the nearest word from the
        target model.
    '''

    y = mm(P, v)

    return model.most_similar(y)
