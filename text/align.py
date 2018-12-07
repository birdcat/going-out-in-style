import numpy as np

from numpy import matmul as mm
from numpy import equal as eq
from numpy import transpose, division, diag, exp
from numpy.linalg import matrix_power as mp
from numpy.linalg import svd

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy import sparse

def gw(X, Y, p, q,
       l=.00005, k=None,
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

        pc = p
        qc = q

    #sv = sparse.csr_matrix(sv)
    #tv = sparse.csr_matrix(tv)

    # apparently sklearn cosine_distance doesn't have a dense output option,
    # but cosine_similarity does? -_-
    C_s = cosine_similarity(sparse.csr_matrix(sv), dense_output=True)
    C_t = cosine_similarity(sparse.csr_matrix(tv), dense_output=True)

    C_s = np.subtract(np.ones(np.shape(C_s)), C_s)
    C_t = np.subtract(np.ones(np.shape(C_t)), C_t)

    C_st = mp(C_s, 2).dot(pc).dot(np.ones((1, k))) + mm(np.ones(k), mm(qc, transpose(mp(C_t, 2))))

    # main loop
    for outer_count in range(max_outer_iters):
        # pseudo-cost matrix
        pC = C_st - 2 * mm(C_s, mm(G, transpose(C_t)))
        
        # Sinkhorn-Knopp iterations
        a = np.ones(k)
        b = np.ones(k)
        K = exp(-1 * pC / l)
        
        for inner_count in range(max_inner_iters):
            a_old = a
            b_old = b
            
            a = division(pc, mm(K, b))
            b = division(qc, mm(transpose(K), a))

            if eq(a, a_old) and eq(b, b_old):
                # converged
                break

        G_old = G
        G = mm(mm(diag(a), K), diag(b))

        if eq(G_old, G):
            break

    # find projection
    # SVD of (X G Y^T)
    # P = U V^T
    u, s, vh = svd(mm(sv, mm(G, transpose(tv))))

    P = mm(u, vh)
    
    return G, P

def align_map(v, model, G, P):
    '''
        After the mapping and projection have been computed, applies them to a
        source-language word embedding and returns the nearest word from the
        target model.
    '''

    y = mm(P, v)

    return model.most_similar(y)
