import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix,bsr_matrix,csc_matrix, csgraph
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc,isspmatrix_coo

def sigmoid(x, x0, k, c=1):
    return(c/(1 + np.exp(-1 * k * (x - x0))))

def checkcsr(csr_cm):
    if not issparse(csr_cm):
        csr_cm = csr_matrix(csr_cm)
    if isspmatrix_csc(csr_cm):
        csr_cm = csr_cm.transpose()
    if isspmatrix_coo(csr_cm):
        csr_cm = csr_cm.tocsr()
    if not isspmatrix_csr(csr_cm):
        print(type(csr_cm))
        raise('the diffmap matrix must in csr_matrix type!')
    else:
        return(csr_cm)

def scale_prob(X, axis=1):
    X =X.copy()
    X /= np.sum(X, axis=axis)
    X[np.isnan(X)] = 1/X.shape[axis]
    return(X)

def subcsxm(csxm):
    rowidx, colidx = csxm.nonzero()
    cmdata = csxm.data

def Time():
    import time
    return(time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime()))

def corstate(x,y, methd='pearsonr'):
    from minepy import MINE
    import numpy as np
    import scipy.stats

    if methd =='pearsonr':
        return(scipy.stats.pearsonr(x, y)[0])
    elif methd =='spearmanr':
        return(scipy.stats.spearmanr(x, y)['correlation'])
    elif methd =='kendalltau':
        return(scipy.stats.kendalltau(x, y)['correlation'])
    elif methd=='mic':
        m = MINE()
        m.compute_score(x, y)
        return(m.mic())
