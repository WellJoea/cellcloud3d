import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, issparse, spdiags
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from scipy.stats import gmean
from cellcloud3d.tools._spatial_edges import coord_2_dist

def normalizeScalFactor(Locs, D1 = 1., D2 = 3., qauntilescale=True):
    # scalFactor = gmean(np.max(Locs,axis=0)-np.min(Locs,axis=0))/(Locs.shape[0])**(1/3)
    if qauntilescale:
        scaleFactor = gmean(np.quantile(Locs,0.975,axis=0)-np.quantile(Locs,0.025,axis=0))/(Locs.shape[0])**(1/3)
    else:
        scaleFactor = gmean(np.max(Locs,axis=0)-np.min(Locs,axis=0))/(Locs.shape[0])**(1/3)
    D1 = D1*scaleFactor
    D2 = D2*scaleFactor
    print("Scaled small patch radius:"+str(D1)+"\tScaled big patch radius:"+str(D2))
    return D1,D2

def dKNN(distMax, radius, n_jobs=-1, remove_loop =False, **kargs):
    ([src, dst]), dist, simi = coord_2_dist(distMax, 
                 n_neighbors=None,
                 radius=radius,
                 show_hist=False,
                 remove_loop=remove_loop,
                 verbose = True,
                 n_jobs=n_jobs,
                  **kargs)
    return src, dst

def spvars(spmx, axis=None):
    if not issparse(spmx):
        spmx_ = csr_matrix(spmx).copy()
    else:
        spmx_ = spmx.copy()
    spmx_sq = spmx_.copy()
    spmx_sq.data **= 2
    Va = np.array(spmx_sq.mean(axis)) - np.square(spmx_.mean(axis))
    return np.array(Va)

def scattermean(srt, dist, H, size = None): # H cell*gene
    if not issparse(H):
        H = csr_matrix(H) 
    size = size or H.shape[0]
    data = np.ones(len(dist))
    W = csr_matrix((data, (srt, dist)), shape=(size, size))

    DR = W.sum(0)
    zdr = np.array(DR == 0).flatten()
    if np.sum(zdr)>0:
        ploop = np.zeros(size)
        ploop[zdr] = 1
        ploop = spdiags(ploop, 0., size, size)
        W = W + ploop

    DR = W.sum(0)
    SH = H.transpose().dot(W) # gene * cell
    if (not SH.has_sorted_indices):
        SH.sort_indices()

    SH = csr_matrix(SH/DR).transpose()
    return SH

def adjpvalue(p):
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(float(len(p)), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def statepvalue(T_matrix, mid_threshold=0.9, fitDist='lognormal'):
    T_matrix_c90 = np.quantile(T_matrix, mid_threshold)
    T_matrix_mid = T_matrix[(T_matrix<T_matrix_c90)]

    if fitDist == 'lognormal':
        LogNormPar = [np.mean(np.log(T_matrix_mid)), np.std(np.log(T_matrix_mid))]
        pvalues = [1 - stats.lognorm.cdf(i, scale=np.exp(LogNormPar[0]), s = LogNormPar[1]) for i in T_matrix]
    elif fitDist == 'beta':
        BetaPar = stats.beta.fit(T_matrix_mid, floc=0, fscale=1)
        a0 = BetaPar[0]
        b0 = BetaPar[1]
        pvalues = [1-stats.beta.cdf(i, a0, b0) for i in T_matrix]

    padj = adjpvalue(pvalues)
    return pvalues, padj

def SVGs(Exp, Locs, genename = None, filter_cell = 5, filter_gene = 5, D1=1.0, D2=3.0, 
         remove_loop = True, mid_threshold=0.5, fitDist='beta',
         scaleFactor=1, normalize = 'minmax', n_jobs=-1,):
    assert Exp.shape[0] == Locs.shape[0], 'The number of cells in Exp and Locs should be the same'
    geneIndex  =  np.arange(Exp.shape[1] ) if genename is None else np.array(genename)

    idc = np.array(Exp.sum(1)>filter_cell).flatten()
    idg = np.array(Exp.sum(0)>filter_gene).flatten()

    Exp = Exp[idc,:][:, idg]
    Locs = Locs[idc,:]
    geneIndex = geneIndex[idg]

    print(f'cells number: {Exp.shape[0]}, gene number: {Exp.shape[1]}')

    if  normalize == 'minmax':
        Exp_nom = minmax_scale(Exp.toarray() if issparse(Exp) else Exp,axis=0)
    else:
        Exp_nom = Exp
    D1,D2=normalizeScalFactor(Locs, D1 = D1, D2 = D2)
    Var_G = []

    for iradius in [D1, D2]:
        print(f'*****{iradius}*****')
        src, dst =  dKNN(Locs, iradius, n_jobs=n_jobs, remove_loop=remove_loop)
        aggMean = scattermean(src, dst, Exp_nom)
        geneVar = spvars(aggMean, axis=0).flatten()
        Var_G.append(geneVar)

    Var_Gr = spvars(Exp, axis=0).flatten()
    Var_Gr = (Var_Gr/Var_Gr.max())**scaleFactor
    T_matrix = Var_G[1]/Var_G[0]*Var_Gr

    pvalues, padj = statepvalue(T_matrix, mid_threshold=mid_threshold, fitDist=fitDist)
    outputData = pd.DataFrame({'P_values':pvalues, 'P_adj':padj, 'T_matrix': T_matrix}, index = geneIndex)
    return outputData

def findSVGs(adata, use_raw=False, basis='spatial', add_key = 'SVGs', n_jobs=1, filter_cell=5, 
             filter_gene=5, scaleFactor=1, mid_threshold=0.5, 
             fitDist='beta', D1=1, D2=3, remove_loop=True, 
             normalize='minmax', **kargs):
    
    if use_raw:
        Exp = adata.raw.X
    else:
        Exp = adata.X
    Locs = adata.obsm[basis]
    geneIndex = adata.var_names.values
    Pdata = SVGs(Exp, Locs, genename = geneIndex, 
                 n_jobs=n_jobs, filter_cell=filter_cell, 
                  filter_gene=filter_gene, scaleFactor=scaleFactor, 
                  mid_threshold=mid_threshold, fitDist=fitDist, D1=D1,
                  D2=D2, remove_loop=remove_loop, normalize=normalize,
                  **kargs)
    Pdata.sort_values(by=['P_adj', 'P_values'], inplace=True)
    adata.uns[add_key] = Pdata
    print(f'finished: added to `.uns["{add_key}"]`')
