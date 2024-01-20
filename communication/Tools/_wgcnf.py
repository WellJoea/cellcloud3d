import os
import sys
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import scipy.cluster.hierarchy as sch
from scipy import stats
from  PyWGCNA.wgcna import WGCNA as  PyWN

from Config import Config as cf
import Tools.Cluster as cl
from Plots import _wgcnp, _dotplot
from Plots._Utilis import labels2colors
from Utilis import (checkarray, checkAdjMat, checkcsr, checkdiag, 
                    choose_representation, checksymmetric, normalAdj)

def correctcorr(adata,
                dataX = None,
                neighbor = None, 
                use_rep ='X',
                n_pcs = None,
                copy= False,
                cortype = 'pearson',
                networkType="signed hybrid"):
    cf.ci('compute correlation...', 'blue')
    adata = adata.copy() if copy else adata
    if not neighbor is None:
        CORRX = neighbor.copy()
    else:
        if not dataX is None:
            data = checkarray(dataX).copy()
        else:
            data = choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
            data = checkarray(data).copy()
        if cortype == 'pearson': 
            CORRX = np.corrcoef(data, rowvar=True)
        elif cortype == 'bicor': 
            CORRX = bicor(data, axis=1)
            np.fill_diagonal(CORRX, 1)
        else:
            sys.exit(colored("correlation type is not in pearson or bicor.", "red"))
        
    if CORRX.shape[1] < 3:
        sys.exit(colored("The input data data contain fewer than 3 rows (nodes).\n"
                            "This would result in a trivial correlation network.", "red"))
    CORRX = checkarray(CORRX)
    pcorr = activation(CORRX, acttype=networkType)

    if np.count_nonzero(np.isnan(pcorr)) != 0:
        cf.cw(f"Some correlations are NA in matrix", 'red')

    cf.ci("add 'pcorr' to adata.obsp...", 'green')
    adata.obsp['pcorr'] = pcorr
    return adata if copy else None

def pickpower(adata,
                pcorr=None, 
                use_soft=True,
                use_adjust=False,
                copy= False,
                cortype='pearson',
                networkType="signed hybrid",
                R2Cut=0.8, MeanCut=100, 
                Powers=None, nBreaks=10,
                kernel='power',
                base=1e-09,
                dropna=False,
                **pcorr_args ):
    cf.ci("pick soft threshold...", 'blue')
    adata = adata.copy() if copy else adata

    pcorr_args = dict( neighbor = pcorr,
                        cortype = cortype,
                        networkType=networkType,
                        **pcorr_args)
    correctcorr( adata, **pcorr_args)
    pcorr = adata.obsp['pcorr']
    if use_soft:
        soft_args = dict( pcorr = pcorr,
                          use_adjust=use_adjust,
                            R2Cut=R2Cut,
                            MeanCut=MeanCut,
                            Powers=Powers,
                            nBreaks=nBreaks,
                            kernel=kernel,
                            base=base,
                            dropna =dropna
                        )
        power, sft = soft_power(**soft_args)
        _wgcnp.soft_threshold(sft, use_adjust=use_adjust, R2Cut=R2Cut)
    else:
        power = empirical_power(pcorr.shape[0], acttype=networkType)
    cf.ci(f"set power as {power}", 'green')
    cf.ci(f"add 'power' to adata.uns...", 'green')
    adata.uns['power'] = power
    return adata if copy else None

def adjacency(adata, 
                pcorr = None,
                power=None, 
                kernel='power',
                normalize =True, 
                networkType="unsigned", 
                cortype = 'pearson', pcorr_args = {},
                copy = False,
                key_adj = 'adj',
                weights=None, scaleByMax=True):
    cf.ci("calculating adjacency matrix ...", 'blue')
    adata = adata.copy() if copy else adata
    if power is None:
        power = adata.uns['power'] 
    else:
        adata.uns['power'] = power
        cf.ci(f"power is set to {power} ...", 'blue')

    weights = ScaleWeights(weights, adata.X, scaleByMax=scaleByMax, axis=1 )
    if weights is not None: 
        cf.cw("TODO: weighted correlation has not been writted...", 'red')
        cf.cw("please use unweighted correlation...", 'red')

    if pcorr is None:
        try:
            pcorr = adata.obsp['pcorr']
        except:
            pcorr_args = dict(adata = adata,
                                cortype = cortype,
                                networkType=networkType,
                            **pcorr_args)
            pcorr = correctcorr( **pcorr_args )

    padjacency =  checkdiag(pcorr, diag=0, fill=True, verbose=True)
    padjacency = KERNELs(padjacency, p=power, kernel=kernel)
    if normalize:
        padjacency = normalAdj(padjacency)
    if not checksymmetric(padjacency):
        cf.cw("Error: adjacency matrix is not symmetric...", 'red')
    padjacency = checkdiag(padjacency, diag=0, fill=True, verbose=True)
    add_neighbors(adata, 
                    key_added = key_adj,
                    connectivities=padjacency,
                    n_neighbors =None)
    return adata if copy else None

def TOM(adata, 
            copy = False,
            sadj = None, 
            TOMType="unsigned",
            key_adj = 'adj',
            key_tom = 'tom',
            TOMDenom="min", amin=-1, amax=1):
    # Turn adjacency into topological overlap
    # https://seidr.readthedocs.io/en/devel/source/paradigm/paradigm.html
    cf.ci("calculating TOM similarity matrix ...", 'blue')
    adata = adata.copy() if copy else adata
    pre_adj = 'connectivities' if key_adj is None else f'{key_adj}_connectivities'
    adjMat = adata.obsp[pre_adj] if sadj is None else sadj
    adjMat = checkarray(adjMat).copy()
    checkAdjMat(adjMat, amin=amin, amax=amax)

    np.nan_to_num(adjMat, copy=False, nan=0)
    # Wij = (Lij + Aij)/(min(Ki, Kj)+1-Aij)
    # Prepare adjacency
    adjMat = checkdiag(adjMat, diag=0, fill=True)
    # Compute TOM
    L = np.matmul(adjMat, adjMat)
    ki = adjMat.sum(axis=1) # out-degree
    kj = adjMat.sum(axis=0) # in-degree
    """
    [np.minimum(ki_, kj) for ki_ in ki] ==
    for i in range(adjMat.shape[0]):
        Ki = np.sum(adjMat[i,:])
        for j in range(adjMat.shape[0]):
            Kj = np.sum(adjMat[:,j])
            np.min(Ki, Kj)
    """
    if TOMDenom == 'min':
        MINK = np.array([np.minimum(ki_, kj) for ki_ in ki])
    else:  # mean
        MINK = np.array([((ki_ + kj) / 2) for ki_ in ki])
    if TOMType in ['unsigned']:
        tom = (L + adjMat) / (MINK + 1 - adjMat)
    else:  # signed
        tom = np.fabs((L + adjMat)) / (MINK + 1 - np.fabs(adjMat))
    np.fill_diagonal(tom, 1)
    add_neighbors(adata, 
                    key_added = key_tom,
                    connectivities=tom,
                    n_neighbors =None)
    return adata if copy else None

def dyncluster(adata, 
                dissTOM = None, 
                key_conet = 'adj',
                key_dist = None,
                copy = False,
                deepSplit=2,
                minModuleSize = 10,
                key_add = None,
                pamRespectsDendro=False, **kargs):
    cf.ci("computing dynamic tree based on TOM similarity matrix ...", 'blue')
    from scipy.spatial.distance import pdist, squareform
    adata = adata.copy() if copy else adata

    if not dissTOM is None:
        dist = dissTOM
    elif not key_dist is None:
        dist = adata.obsp[f'{key_dist}_distances']
    elif not key_conet is None:
        dist = adata.obsp[f'{key_conet}_connectivities']
        dist = 1- checkarray(dist)
    else:
        dist = adata.obsp[f'connectivities']
        dist = 1- checkarray(dist)
    dist = checkarray(dist).copy()
    dist = checkdiag(dist, diag=0, fill=True)
    distt = squareform(dist, checks=False)

    # Call the hierarchical clustering function
    geneTree = sch.linkage(distt, method="average")
    # loc1 loc2 dist counts
    # Module identification using dynamic tree cut:
    kwgs = dict(deepSplit=deepSplit, 
                pamRespectsDendro=pamRespectsDendro,
                minClusterSize=minModuleSize, **kargs)
    dynamicMods = cutreeHybrid(dendro=geneTree,
                                distM=pd.DataFrame(dist), **kwgs)
    
    #dynamicMods = PyWN.cutreeHybrid(dendro=geneTree, 
    #                                 distM=pd.DataFrame(dist), 
    #                                 **kwgs)
        
    ngroup = dynamicMods['Value'].unique().shape[0]
    cf.ci( f"Identified {ngroup} modules using dynamic tree cut:", 'blue')
    # Convert numeric labels into colors
    key_add = 'dynamic' if key_add is None else key_add
    adata.obs[key_add] = dynamicMods['Value'].astype(str).values
    adata.obs[f'{key_add}_lc'] = labels2colors(labels=dynamicMods['Value'])
    return adata if copy else None

def merge_modules(adata, clust_lc='dynamic_lc',
                    key_add = None,
                    copy = False, expr=None, use_raw=False,
                    MEDissThres=0.2, 
                    min_size = 3,
                    **kargs):
    cf.ci('Merge close modules...', 'blue')
    if expr is None:
        expr = adata.raw.to_adata().to_df().T if use_raw else adata.to_df().T
    cluster = adata.obs[clust_lc].copy()
    clust_counts =  cluster.value_counts()

    low_clust = clust_counts[clust_counts<min_size]
    if low_clust.shape[0]>0:
        cluster[ cluster.isin(low_clust.index) ] = clust_counts.index[0]
        try:
            cluster.cat.remove_unused_categories(inplace=True)
        except:
            pass
        cf.cw(f'{low_clust.shape[0]} clusters are lower than {min_size} '
              f'and will be replaced as {clust_counts.index[0]}', 'red')

    #import Tools.pyWGCNA as WN
    merge = PyWN.mergeCloseModules(expr.copy(), 
                                    cluster.astype(str),
                                    cutHeight = MEDissThres,
                                    **kargs)
    key_add = clust_lc.replace('_lc','') if key_add is None else key_add
    key_add = f'merge_{key_add}'
    col2num = dict(map(lambda x:  (x[1], str(x[0])),#x[::-1], 
                        enumerate(merge['colors'].value_counts().index, start=1)))
    adata.obs[f'{key_add}'] = merge['colors'].map(col2num)
    adata.uns[f'{key_add}_modules'] = merge
    adata.uns[f'{key_add}_ME'] = merge['newMEs']
    adata.obs[f'{key_add}_lc'] = merge['colors']
    return adata if copy else None

def eigengenes(adata, clust_lc='dynamic_lc',
                key_add = None,
                copy = False, expr=None, use_raw=True,
                MEDissThres=0.2,
                plot = True,
                **kargs):
    cf.ci('Calculate eigengenes...', 'blue')
    adata = adata.copy() if copy else adata
    key_add = clust_lc.replace('_lc','') if key_add is None else key_add

    if expr is None:
        expr = adata.raw.to_adata().to_df().T if use_raw else adata.to_df().T
    MEList = moduleEigengenes(expr=expr.copy(),
                              colors=adata.obs[clust_lc].astype(str),
                               **kargs)
    cf.ci(f"add {key_add}_MEs to adata.uns...", 'green')
    adata.uns[f'{key_add}_MEs'] = MEList
    
    if plot:
        import matplotlib.pyplot as plt
        MEs = MEList['eigengenes']
        MEs.drop(['MEgrey'], axis=1, errors='ignore', inplace=True)
        plt.figure(figsize=(max(20, round(MEs.shape[1] / 20)), 8), facecolor='white')
        _dotplot.DotPlot.dendrogram_plot(MEs.T, color_threshold=0.2, leaf_font_size=8)
        plt.axhline(y=0.2, c='grey', lw=1, linestyle='dashed')
        plt.title('Clustering of module eigengenes')
        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()

    return adata if copy else None

def update_eigengenes(adataI,  **kargs):
    cf.ci('Recalculate MEs with color labels', 'blue')
    self.datME = WGCNA.moduleEigengenes(self.adata.to_df(),
                                        self.adata.var['moduleColors'], **kargs)['eigengenes']
    if 'MEgrey' in self.datME.columns:
        self.datME.drop(['MEgrey'], axis=1, inplace=True)
    self.MEs = WGCNA.orderMEs(self.datME)
    
def add_neighbors(
    adata,
    connectivities = None, 
    distances = None,
    setdiag = None,
    n_neighbors = 15,
    n_pcs = None,
    use_rep = None,
    random_state = 0,
    method = 'umap',
    metric = 'euclidean',
    metric_kwds = {},
    key_added  = None,
    copy = False):

    adata = adata.copy() if copy else adata
    if key_added is None:
        key_added = 'neighbors'
        conns_key = 'connectivities'
        dists_key = 'distances'
    else:
        conns_key = key_added + '_connectivities'
        dists_key = key_added + '_distances'

    adata.uns[key_added] = {}

    neighbors_dict = adata.uns[key_added]

    neighbors_dict['connectivities_key'] = conns_key
    neighbors_dict['distances_key'] = dists_key

    neighbors_dict['params'] = {'n_neighbors': n_neighbors, 'method': method}
    neighbors_dict['params']['random_state'] = random_state
    neighbors_dict['params']['metric'] = metric
    if metric_kwds:
        neighbors_dict['params']['metric_kwds'] = metric_kwds
    if use_rep is not None:
        neighbors_dict['params']['use_rep'] = use_rep
    if n_pcs is not None:
        neighbors_dict['params']['n_pcs'] = n_pcs

    if not distances is None:
        adata.obsp[dists_key] = checkcsr(distances).copy()
        if not setdiag is None:
            adata.obsp[dists_key].setdiag(setdiag)
    if not connectivities is None:
        adata.obsp[conns_key] = checkcsr(connectivities).copy()
        if not setdiag is None:
            adata.obsp[conns_key].setdiag(setdiag)

    cf.ci( f'\nadd "{key_added!r}" to adata.uns...\n'
           f'    `.obsp[{dists_key!r}]`, distances for each pair of neighbors\n'
           f'    `.obsp[{conns_key!r}]`, weighted adjacency matrix',
           'green')
    return adata if copy else None

def soft_power(pcorr,
                    R2Cut=0.9, MeanCut=100, Powers=None,
                    nBreaks=10, base=1, dropna=False,
                    kernel='power',
                    use_adjust =False,
                    use_block=True, maxMemoryAllocation = 50 * 1024**3, 
                    blockSize=None):
    cf.ci('find soft power threshod', 'blue')
    pcorr = checkdiag(pcorr, diag=0, fill=True)
    pcorr = checkarray(pcorr)
    Powers = list(range(1, 11)) + list(range(11, 21, 2)) if Powers is None else Powers
    datk = []
    for j in Powers: # slow
        #corxCur = pcorr ** j
        corxCur = KERNELs(pcorr, p=j, kernel=kernel)
        #corxCur = normalAdj(corxCur)
        datk_j = np.nansum(corxCur, axis=0) #- 1
        datk.append(datk_j)
    datk = np.vstack(datk).T

    datout = [scaleFreePower(datk[:,i], nBreaks= nBreaks, base=base, dropna=dropna) 
                            for i in range(datk.shape[1]) ]
    datout = pd.concat(datout, axis=1).T
    datout['Power'] = Powers
    datout.index = Powers
    print(datout)
    # detect threshold more than 0.9 by default
    X_R2 = 'truncated R.sq' if use_adjust else 'SFT.R.sq'
    candidate = datout[((datout[X_R2] > R2Cut) & (datout['mean(k)'] <= MeanCut))]
    if candidate.shape[0] > 0:
        powerEstimate = candidate['Power'].iloc[0]
        cf.ci(f"Selected power to have scale free network is {powerEstimate}", 'green')
    else:
        powerEstimate = datout.loc[ datout[X_R2].idxmax(), 'Power' ]
        cf.cw(f"No power detected to have scale free network!\nUse the max power given by R^2: {powerEstimate}", 'red')
    return powerEstimate, datout

def empirical_power(nsample, acttype='unsigned'):
    type1 = ['unsigned', 'relu', 'signed hybrid', 'leaky_relu2', 'elu2', 'swish2']
    type2 = ['signed', 'softplus2', 'softmax2']
    if nsample <20:
        return (9 if acttype in type1 else 18)
    elif nsample >=20 & nsample <30:
        return (8 if acttype in type1 else 16)
    if nsample >=30 & nsample <40:
        return (7 if acttype in type1 else 14)
    else:
        return (6 if acttype in type1 else 12)

def KERNELs(X, p=2, kernel='power'):
    X = checkarray(X).astype(np.float64)
    if kernel=='power':
        return np.power(X, p)
    elif kernel=='linear':
        Y =  np.dot(X.T, X)
        return (Y / Y.max())
    elif kernel=='poly':
        Y =  np.dot(X.T, X)
        Y /= Y.max()
        return np.power(Y, p)
    elif kernel=='tanh':
        return np.tanh(np.dot(X.T, X))

def activation(X, acttype='unsigned', alpha=0.02):
    X = checkarray(X).astype(np.float64).copy()
    if acttype=='unsigned':
        return np.abs(X)
    elif acttype=='signed':
        return (1 + X) / 2
    elif acttype in ['relu', 'signed hybrid']:
        return np.where(X>0, X, 0)
    elif acttype=='leaky_relu2':
        return np.where(X>0, X, -alpha * X)
    elif acttype=='elu2':
        return np.where(X>0, X, -alpha * (np.exp(X)-1))
    elif acttype=='swish2':
        return np.where(X>0, X, -X/(1 + np.exp(-X)))
    elif acttype=='softplus2':
        #X = np.log(1+np.exp(X))
        return np.log2(1+np.power(2,X-1))
    elif acttype=='softmax2':
        X = np.exp(X)
        X = X/X.sum(0)
        return(X)
    else:
        'add tanh'
        cf.pw(f"invalid activate type. Will ignore activation.", 'red')
        return X

def bicor_stand(x):
    me = np.median(x)
    md = np.median(np.abs(x-me)) #*1.4826
    if md==0:
        xb = x-np.mean(x)
        st = np.std(x) * np.sqrt(xb.shape[0])
        return(xb/st)
    else:
        ux = (x -me)/(9*md)
        wx = np.where(np.abs(ux)<1, (1-ux**2)**2, 0)
        sx = (x-me)*wx
        Xs = sx/np.sqrt(np.sum(sx**2))
        return(Xs)

def bicor(x, y=None,  axis=0):
    if len(x.shape) ==1:
        X = bicor_stand(x)
    else:
        X = np.apply_along_axis(bicor_stand, axis, x)
    if y is None:
        C = np.dot(X.T, X) if axis==0 else np.dot(X, X.T) 
    else:
        if len(y.shape) ==1:
            Y = bicor_stand(y)
        else:
            Y = np.apply_along_axis(bicor_stand, axis, y)
        C = np.dot(X, Y)
    try:
        np.clip(C.real, -1, 1, out=C.real)
        np.fill_diagonal(C.real, 1)
    except:
        pass
    return(C)

def wcor_stand(x, w=None):
    if w is None:
        w = np.ones(len(x))
    wm = np.sum(x*w)/len(x)
    xb = x- wm
    st = np.sqrt(np.sum(xb**2))
    return(xb/st)

def wcor(x, y, wx, wy = None):
    wy = wx if wy is None else wy
    return(np.dot(self.wcor_stand(x, wx), self.wcor_stand(y, wy)))

def scaleFreePower( k, nBreaks=10, base=1e-9, dropna=True):
    """
    calculates several indices (fitting statistics) for evaluating scale free topology fit.
    """
    k = checkarray(k).copy()
    nodes = k.shape[0]
    df = pd.DataFrame({'data': k, 
                    "discretized_k":pd.cut(k, nBreaks)})
    df = df.groupby('discretized_k').agg({'data':['mean', 'size']}).fillna(0)
    df.columns = ['dk', 'p_dk']
    df['p_dk'] /=df['p_dk'].sum()
    df.reset_index(inplace=True)

    edges = np.linspace(start=min(k), stop=max(k), num=nBreaks + 1)
    dk2 = (edges[1:] + edges[:-1])/2
    df.loc[(df['dk'] == 0), 'dk'] = dk2[df['dk'] == 0]

    if dropna:
        df = df[(df['p_dk']>0)]
        #base = 0
    df['log_dk'] = np.log10(df['dk'])
    df['log_p_dk'] = np.log10(df['p_dk'] + base)
    df['log_p_dk_10'] = np.power(10, df['log_dk'])

    model1 = ols(formula='log_p_dk ~ log_dk', data=df).fit()
    model2 = ols(formula='log_p_dk ~ log_dk + log_p_dk_10', data=df).fit()
    
    return pd.Series({'SFT.R.sq': model1.rsquared,
            'slope': model1.params.values[1],
            'truncated R.sq': model2.rsquared_adj,
            'mean(k)': np.mean(k),
            'median(k)': np.median(k),
            'max(k)': np.max(k),
            'Density': np.sum(k) / (nodes * (nodes - 1)),
            'Centralization': (np.max(k) - np.mean(k)) * nodes / ((nodes - 1) * (nodes - 2)),
            'Heterogeneity': np.sqrt(nodes * np.sum(k ** 2) / np.sum(k) ** 2 - 1) })

def ScaleWeights(weights, expr, scaleByMax=True, axis=1):
    if weights is None:
        return weights

    weights = checkarray(weights)
    if expr.shape != weights.shape:
        sys.exit("When 'weights' are given, they must have the same dimensions as 'expr'.")
    if (weights < 0).any():
        sys.exit("Found negative weights. All weights must be non-negative.")

    nf = np.isinf(weights)
    if any(nf):
        print(f"WARNING:Found non-finite weights. The corresponding data points will be removed.")
        weights[nf] = 0

    if scaleByMax:
        maxw = np.amax(weights, axis=axis)
        maxw[maxw == 0] = 1
        weights = weights / maxw

    return weights

def cutreeHybrid(dendro, distM, cutHeight=None, minClusterSize=20, deepSplit=1,
                    maxCoreScatter=None, minGap=None, maxAbsCoreScatter=None,
                    minAbsGap=None, minSplitHeight=None, minAbsSplitHeight=None,
                    externalBranchSplitFnc=None, nExternalSplits=0, minExternalSplit=None,
                    externalSplitOptions=pd.DataFrame(), externalSplitFncNeedsDistance=None,
                    assumeSimpleExternalSpecification=True, pamStage=True,
                    pamRespectsDendro=True, useMedoids=False, maxPamDist=None,
                    respectSmallClusters=True):
    """
    Detect clusters in a dendorgram produced by the function hclust.

    :param dendro: a hierarchical clustering dendorgram such as one returned by hclust.
    :type dendro: ndarray
    :param distM: Distance matrix that was used as input to hclust.
    :type distM: pandas dataframe
    :param cutHeight: Maximum joining heights that will be considered. It defaults to 99of the range between the 5th percentile and the maximum of the joining heights on the dendrogram.
    :type cutHeight: int
    :param minClusterSize: Minimum cluster size. (default = 20)
    :type minClusterSize: int
    :param deepSplit: Either logical or integer in the range 0 to 4. Provides a rough control over sensitivity to cluster splitting. The higher the value, the more and smaller clusters will be produced. (default = 1)
    :type deepSplit: int or bool
    :param maxCoreScatter: Maximum scatter of the core for a branch to be a cluster, given as the fraction of cutHeight relative to the 5th percentile of joining heights.
    :type maxCoreScatter: int
    :param minGap: Minimum cluster gap given as the fraction of the difference between cutHeight and the 5th percentile of joining heights.
    :type minGap: int
    :param maxAbsCoreScatter: Maximum scatter of the core for a branch to be a cluster given as absolute heights. If given, overrides maxCoreScatter.
    :type maxAbsCoreScatter: int
    :param minAbsGap: Minimum cluster gap given as absolute height difference. If given, overrides minGap.
    :type minAbsGap: int
    :param minSplitHeight: Minimum split height given as the fraction of the difference between cutHeight and the 5th percentile of joining heights. Branches merging below this height will automatically be merged. Defaults to zero but is used only if minAbsSplitH
    :type minSplitHeight: int
    :param minAbsSplitHeight: Minimum split height given as an absolute height. Branches merging below this height will automatically be merged. If not given (default), will be determined from minSplitHeight above.
    :type minAbsSplitHeight: int
    :param externalBranchSplitFnc: Optional function to evaluate split (dissimilarity) between two branches. Either a single function or a list in which each component is a function.
    :param minExternalSplit: Thresholds to decide whether two branches should be merged. It should be a numeric list of the same length as the number of functions in externalBranchSplitFnc above.
    :type minExternalSplit: list
    :param externalSplitOptions: Further arguments to function externalBranchSplitFnc. If only one external function is specified in externalBranchSplitFnc above, externalSplitOptions can be a named list of arguments or a list with one component.
    :type externalSplitOptions: pandas dataframe
    :param externalSplitFncNeedsDistance: Optional specification of whether the external branch split functions need the distance matrix as one of their arguments. Either NULL or a logical list with one element per branch
    :type externalSplitFncNeedsDistance: pandas dataframe
    :param assumeSimpleExternalSpecification: when minExternalSplit above is a scalar (has length 1), should the function assume a simple specification of externalBranchSplitFnc and externalSplitOptions. (default = True)
    :type assumeSimpleExternalSpecification: bool
    :param pamStage: If TRUE, the second (PAM-like) stage will be performed. (default = True)
    :type pamStage: bool
    :param pamRespectsDendro: If TRUE, the PAM stage will respect the dendrogram in the sense an object can be PAM-assigned only to clusters that lie below it on the branch that the object is merged into. (default = True)
    :type pamRespectsDendro: bool
    :param useMedoids: if TRUE, the second stage will be use object to medoid distance; if FALSE, it will use average object to cluster distance. (default = False)
    :param maxPamDist: Maximum object distance to closest cluster that will result in the object assigned to that cluster. Defaults to cutHeight.
    :type maxPamDist: float
    :param respectSmallClusters: If TRUE, branches that failed to be clusters in stage 1 only because of insufficient size will be assigned together in stage 2. If FALSE, all objects will be assigned individually. (default = False)
    :type respectSmallClusters: bool

    :return: list detailing the deteced branch structure.
    :rtype: list
    """
    tmp = dendro[:, 0] > dendro.shape[0]
    dendro[tmp, 0] = dendro[tmp, 0] - dendro.shape[0]
    dendro[np.logical_not(tmp), 0] = -1 * (dendro[np.logical_not(tmp), 0] + 1)
    tmp = dendro[:, 1] > dendro.shape[0]
    dendro[tmp, 1] = dendro[tmp, 1] - dendro.shape[0]
    dendro[np.logical_not(tmp), 1] = -1 * (dendro[np.logical_not(tmp), 1] + 1)

    chunkSize = dendro.shape[0]

    if maxPamDist is None:
        maxPamDist = cutHeight

    nMerge = dendro.shape[0]
    if nMerge < 1:
        sys.exit("The given dendrogram is suspicious: number of merges is zero.")
    if distM is None:
        sys.exit("distM must be non-NULL")
    if distM.shape is None:
        sys.exit("distM must be a matrix.")
    if distM.shape[0] != nMerge + 1 or distM.shape[1] != nMerge + 1:
        sys.exit("distM has incorrect dimensions.")
    if pamRespectsDendro and not respectSmallClusters:
        print("cutreeHybrid Warning: parameters pamRespectsDendro (TRUE) "
                "and respectSmallClusters (FALSE) imply contradictory intent.\n"
                "Although the code will work, please check you really intented "
                "these settings for the two arguments.", flush=True)
    cf.ci('Going through the merge tree...', 'cyan')
    if any(np.diag(distM) != 0):
        np.fill_diagonal(distM, 0)
    refQuantile = 0.05
    refMerge = round(nMerge * refQuantile) - 1
    if refMerge < 0:
        refMerge = 0
    refHeight = dendro[refMerge, 2]
    if cutHeight is None:
        cutHeight = 0.99 * (np.max(dendro[:, 2]) - refHeight) + refHeight
        print("..cutHeight not given, setting it to", round(cutHeight, 3),
                " ===>  99% of the (truncated) height range in dendro.", flush=True)
    else:
        if cutHeight > np.max(dendro[:, 2]):
            cutHeight = np.max(dendro[:, 2])
    if maxPamDist is None:
        maxPamDist = cutHeight
    nMergeBelowCut = np.count_nonzero(dendro[:, 2] <= cutHeight)
    if nMergeBelowCut < minClusterSize:
        print("cutHeight set too low: no merges below the cut.", flush=True)
        return pd.DataFrame({'labels': np.repeat(0, nMerge + 1, axis=0)})

    if externalBranchSplitFnc is not None:
        nExternalSplits = len(externalBranchSplitFnc)
        if len(minExternalSplit) < 1:
            sys.exit("'minExternalBranchSplit' must be given.")
        if assumeSimpleExternalSpecification and nExternalSplits == 1:
            externalSplitOptions = pd.DataFrame(externalSplitOptions)
        # TODO: externalBranchSplitFnc = lapply(externalBranchSplitFnc, match.fun)
        for es in range(nExternalSplits):
            externalSplitOptions['tree'][es] = dendro
            if len(externalSplitFncNeedsDistance) == 0 or externalSplitFncNeedsDistance[es]:
                externalSplitOptions['dissimMat'][es] = distM

    MxBranches = nMergeBelowCut
    branch_isBasic = np.repeat(True, MxBranches, axis=0)
    branch_isTopBasic = np.repeat(True, MxBranches, axis=0)
    branch_failSize = np.repeat(False, MxBranches, axis=0)
    branch_rootHeight = np.repeat(np.nan, MxBranches, axis=0)
    branch_size = np.repeat(2, MxBranches, axis=0)
    branch_nMerge = np.repeat(1, MxBranches, axis=0)
    branch_nSingletons = np.repeat(2, MxBranches, axis=0)
    branch_nBasicClusters = np.repeat(0, MxBranches, axis=0)
    branch_mergedInto = np.repeat(0, MxBranches, axis=0)
    branch_attachHeight = np.repeat(np.nan, MxBranches, axis=0)
    branch_singletons = pd.DataFrame()
    branch_basicClusters = pd.DataFrame()
    branch_mergingHeights = pd.DataFrame()
    branch_singletonHeights = pd.DataFrame()
    nBranches = -1

    defMCS = [0.64, 0.73, 0.82, 0.91, 0.95]
    defMG = [(1.0 - defMC) * 3.0 / 4.0 for defMC in defMCS]
    nSplitDefaults = len(defMCS)
    if isinstance(deepSplit, bool):
        deepSplit = pd.to_numeric(deepSplit) * (nSplitDefaults - 2)
    if deepSplit < 0 or deepSplit > nSplitDefaults:
        msg = "Parameter deepSplit (value" + str(deepSplit) + \
                ") out of range: allowable range is 0 through", str(nSplitDefaults - 1)
        sys.exit(msg)

    if maxCoreScatter is None:
        maxCoreScatter = interpolate(defMCS, deepSplit)
    if minGap is None:
        minGap = interpolate(defMG, deepSplit)
    if maxAbsCoreScatter is None:
        maxAbsCoreScatter = refHeight + maxCoreScatter * (cutHeight - refHeight)
    if minAbsGap is None:
        minAbsGap = minGap * (cutHeight - refHeight)
    if minSplitHeight is None:
        minSplitHeight = 0
    if minAbsSplitHeight is None:
        minAbsSplitHeight = refHeight + minSplitHeight * (cutHeight - refHeight)
    nPoints = nMerge + 1
    IndMergeToBranch = np.repeat(-1, nMerge, axis=0)
    onBranch = np.repeat(0, nPoints, axis=0)
    RootBranch = 0

    mergeDiagnostics = pd.DataFrame({'smI': np.repeat(np.nan, nMerge, axis=0),
                                        'smSize': np.repeat(np.nan, nMerge, axis=0),
                                        'smCrSc': np.repeat(np.nan, nMerge, axis=0),
                                        'smGap': np.repeat(np.nan, nMerge, axis=0),
                                        'lgI': np.repeat(np.nan, nMerge, axis=0),
                                        'lgSize': np.repeat(np.nan, nMerge, axis=0),
                                        'lgCrSc': np.repeat(np.nan, nMerge, axis=0),
                                        'lgGap': np.repeat(np.nan, nMerge, axis=0),
                                        'merged': np.repeat(np.nan, nMerge, axis=0)})
    if externalBranchSplitFnc is not None:
        externalMergeDiags = pd.DataFrame(np.nan, index=list(range(nMerge)), columns=list(range(nExternalSplits)))

    extender = np.repeat(0, chunkSize, axis=0)

    for merge in range(nMerge):
        if dendro[merge, 2] <= cutHeight:
            if dendro[merge, 0] < 0 and dendro[merge, 1] < 0:
                nBranches = nBranches + 1
                branch_isBasic[nBranches] = True
                branch_isTopBasic[nBranches] = True
                branch_singletons.insert(nBranches, nBranches,
                                            np.concatenate((-1 * dendro[merge, 0:2], extender), axis=0))
                branch_basicClusters.insert(nBranches, nBranches, extender)
                branch_mergingHeights.insert(nBranches, nBranches,
                                                np.concatenate((np.repeat(dendro[merge, 2], 2), extender), axis=0))
                branch_singletonHeights.insert(nBranches, nBranches,
                                                np.concatenate((np.repeat(dendro[merge, 2], 2), extender), axis=0))
                IndMergeToBranch[merge] = nBranches
                RootBranch = nBranches
            elif np.sign(dendro[merge, 0]) * np.sign(dendro[merge, 1]) < 0:
                clust = IndMergeToBranch[int(np.max(dendro[merge, 0:2])) - 1]

                if clust == -1:
                    sys.exit("Internal error: a previous merge has no associated cluster. Sorry!")

                gene = -1 * int(np.min(dendro[merge, 0:2]))
                ns = branch_nSingletons[clust]
                nm = branch_nMerge[clust]

                if branch_isBasic[clust]:
                    branch_singletons.loc[ns, clust] = gene
                    branch_singletonHeights.loc[ns, clust] = dendro[merge, 2]
                else:
                    onBranch[int(gene)] = clust

                branch_mergingHeights.loc[nm, clust] = dendro[merge, 2]
                branch_size[clust] = branch_size[clust] + 1
                branch_nMerge[clust] = nm + 1
                branch_nSingletons[clust] = ns + 1
                IndMergeToBranch[merge] = clust
                RootBranch = clust
            else:
                clusts = IndMergeToBranch[dendro[merge, 0:2].astype(int) - 1]
                sizes = branch_size[clusts]
                rnk = np.argsort(sizes)
                small = clusts[rnk[0]]
                large = clusts[rnk[1]]
                sizes = sizes[rnk]

                if branch_isBasic[small]:
                    coresize = coreSizeFunc(branch_nSingletons[small], minClusterSize) - 1
                    Core = branch_singletons.loc[0:coresize, small] - 1
                    Core = Core.astype(int).tolist()
                    SmAveDist = np.mean(distM.iloc[Core, Core].sum() / coresize)
                else:
                    SmAveDist = 0

                if branch_isBasic[large]:
                    coresize = coreSizeFunc(branch_nSingletons[large], minClusterSize) - 1
                    Core = branch_singletons.loc[0:coresize, large] - 1
                    Core = Core.astype(int).tolist()
                    LgAveDist = np.mean(distM.iloc[Core, Core].sum() / coresize)
                else:
                    LgAveDist = 0

                mergeDiagnostics.loc[merge, :] = [small, branch_size[small], SmAveDist,
                                                    dendro[merge, 2] - SmAveDist,
                                                    large, branch_size[large], LgAveDist,
                                                    dendro[merge, 2] - LgAveDist,
                                                    None]
                SmallerScores = [branch_isBasic[small], branch_size[small] < minClusterSize,
                                    SmAveDist > maxAbsCoreScatter, dendro[merge, 2] - SmAveDist < minAbsGap,
                                    dendro[merge, 2] < minAbsSplitHeight]
                if SmallerScores[0] * np.count_nonzero(SmallerScores[1:]) > 0:
                    DoMerge = True
                    SmallerFailSize = not (SmallerScores[2] | SmallerScores[3])
                else:
                    LargerScores = [branch_isBasic[large],
                                    branch_size[large] < minClusterSize, LgAveDist > maxAbsCoreScatter,
                                    dendro[merge, 2] - LgAveDist < minAbsGap,
                                    dendro[merge, 2] < minAbsSplitHeight]
                    if LargerScores[0] * np.count_nonzero(LargerScores[1:]) > 0:
                        DoMerge = True
                        SmallerFailSize = not (LargerScores[2] | LargerScores[3])
                        x = small
                        small = large
                        large = x
                        sizes = np.flip(sizes)
                    else:
                        DoMerge = False

                if DoMerge:
                    mergeDiagnostics['merged'][merge] = 1

                if not DoMerge and nExternalSplits > 0 and branch_isBasic[small] and branch_isBasic[large]:
                    branch1 = branch_singletons[[large]][0:sizes[1]]
                    branch2 = branch_singletons[[small]][0:sizes[0]]
                    es = 0
                    while es < nExternalSplits and not DoMerge:
                        es = es + 1
                        args = pd.DataFrame({'externalSplitOptions': externalSplitOptions[[es]],
                                                'branch1': branch1, 'branch2': branch2})
                        # TODO: extSplit = do.call(externalBranchSplitFnc[[es]], args)
                        extSplit = None
                        DoMerge = extSplit < minExternalSplit[es]
                        externalMergeDiags[merge, es] = extSplit
                        mergeDiagnostics['merged'][merge] = 0
                        if DoMerge:
                            mergeDiagnostics['merged'][merge] = 2

                if DoMerge:
                    branch_failSize[[small]] = SmallerFailSize
                    branch_mergedInto[small] = large + 1
                    branch_attachHeight[small] = dendro[merge, 2]
                    branch_isTopBasic[small] = False
                    nss = branch_nSingletons[small] - 1
                    nsl = branch_nSingletons[large]
                    ns = nss + nsl
                    if branch_isBasic[large]:
                        branch_singletons.loc[nsl:ns, large] = branch_singletons.loc[0:nss, small].values
                        branch_singletonHeights.loc[nsl:ns, large] = branch_singletonHeights.loc[0:nss,
                                                                        small].values
                        branch_nSingletons[large] = ns + 1
                    else:
                        if not branch_isBasic[small]:
                            sys.exit("Internal error: merging two composite clusters. Sorry!")
                        tmp = branch_singletons[[small]].astype(int).values
                        tmp = tmp[tmp != 0]
                        tmp = tmp - 1
                        onBranch[tmp] = large + 1

                    nm = branch_nMerge[large]
                    branch_mergingHeights.loc[nm, large] = dendro[merge, 2]
                    branch_nMerge[large] = nm + 1
                    branch_size[large] = branch_size[small] + branch_size[large]
                    IndMergeToBranch[merge] = large
                    RootBranch = large
                else:
                    if branch_isBasic[large] and not branch_isBasic[small]:
                        x = large
                        large = small
                        small = x
                        sizes = np.flip(sizes)

                    if branch_isBasic[large] or (pamStage and pamRespectsDendro):
                        nBranches = nBranches + 1
                        branch_attachHeight[[large, small]] = dendro[merge, 2]
                        branch_mergedInto[[large, small]] = nBranches
                        if branch_isBasic[small]:
                            addBasicClusters = [small + 1]
                        else:
                            addBasicClusters = branch_basicClusters.loc[
                                (branch_basicClusters[[small]] != 0).all(axis=1), small]
                        if branch_isBasic[large]:
                            addBasicClusters = np.concatenate((addBasicClusters, [large + 1]), axis=0)
                        else:
                            addBasicClusters = np.concatenate((addBasicClusters,
                                                                branch_basicClusters.loc[(
                                                                                                branch_basicClusters[
                                                                                                    [
                                                                                                        large]] != 0).all(
                                                                    axis=1), large]),
                                                                axis=0)
                        branch_isBasic[nBranches] = False
                        branch_isTopBasic[nBranches] = False
                        branch_basicClusters.insert(nBranches, nBranches,
                                                    np.concatenate((addBasicClusters,
                                                                    np.repeat(0,
                                                                                chunkSize - len(addBasicClusters))),
                                                                    axis=0))
                        branch_singletons.insert(nBranches, nBranches, np.repeat(np.nan, chunkSize + 2))
                        branch_singletonHeights.insert(nBranches, nBranches, np.repeat(np.nan, chunkSize + 2))
                        branch_mergingHeights.insert(nBranches, nBranches,
                                                        np.concatenate((np.repeat(dendro[merge, 2], 2), extender),
                                                                    axis=0))
                        branch_nMerge[nBranches] = 2
                        branch_size[nBranches] = sum(sizes) + 2
                        branch_nBasicClusters[nBranches] = len(addBasicClusters)
                        IndMergeToBranch[merge] = nBranches
                        RootBranch = nBranches
                    else:
                        if branch_isBasic[small]:
                            addBasicClusters = [small + 1]
                        else:
                            addBasicClusters = branch_basicClusters.loc[
                                (branch_basicClusters[[small]] != 0).all(axis=1), small]

                        nbl = branch_nBasicClusters[large]
                        nb = branch_nBasicClusters[large] + len(addBasicClusters)
                        branch_basicClusters.iloc[nbl:nb, large] = addBasicClusters
                        branch_nBasicClusters[large] = nb
                        branch_size[large] = branch_size[large] + branch_size[small]
                        nm = branch_nMerge[large] + 1
                        branch_mergingHeights.loc[nm, large] = dendro[merge, 2]
                        branch_nMerge[large] = nm
                        branch_attachHeight[small] = dendro[merge, 2]
                        branch_mergedInto[small] = large + 1
                        IndMergeToBranch[merge] = large
                        RootBranch = large

    nBranches = nBranches + 1
    isCluster = np.repeat(False, nBranches)
    SmallLabels = np.repeat(0, nPoints)

    for clust in range(nBranches):
        if np.isnan(branch_attachHeight[clust]):
            branch_attachHeight[clust] = cutHeight
        if branch_isTopBasic[clust]:
            coresize = coreSizeFunc(branch_nSingletons[clust], minClusterSize)
            Core = branch_singletons.iloc[0:coresize, clust] - 1
            Core = Core.astype(int).tolist()
            CoreScatter = np.mean(distM.iloc[Core, Core].sum() / (coresize - 1))
            isCluster[clust] = (branch_isTopBasic[clust] and branch_size[clust] >= minClusterSize and
                                CoreScatter < maxAbsCoreScatter and branch_attachHeight[
                                    clust] - CoreScatter > minAbsGap)
        else:
            CoreScatter = 0
        if branch_failSize[clust]:
            SmallLabels[branch_singletons[[clust]].astype(int) - 1] = clust + 1

    if not respectSmallClusters:
        SmallLabels = np.repeat(0, nPoints)

    Colors = np.zeros((nPoints,))
    coreLabels = np.zeros((nPoints,))
    clusterBranches = np.where(isCluster)[0].tolist()
    branchLabels = np.zeros((nBranches,))
    color = 0

    for clust in clusterBranches:
        color = color + 1
        tmp = branch_singletons[[clust]].astype(int) - 1
        tmp = tmp[tmp != -1]
        tmp.dropna(inplace=True)
        tmp = tmp.iloc[:, 0].astype(int)
        Colors[tmp] = color
        SmallLabels[tmp] = 0
        coresize = coreSizeFunc(branch_nSingletons[clust], minClusterSize)
        Core = branch_singletons.loc[0:coresize, clust] - 1
        Core = Core.astype(int).tolist()
        coreLabels[Core] = color
        branchLabels[clust] = color

    Labeled = np.where(Colors != 0)[0].tolist()
    Unlabeled = np.where(Colors == 0)[0].tolist()
    nUnlabeled = len(Unlabeled)
    UnlabeledExist = nUnlabeled > 0

    if len(Labeled) > 0:
        LabelFac = pd.Categorical(Colors[Labeled])
        nProperLabels = len(LabelFac.categories)
    else:
        nProperLabels = 0

    if pamStage and UnlabeledExist and nProperLabels > 0:
        nPAMed = 0
        if useMedoids:
            Medoids = np.repeat(0, nProperLabels)
            ClusterRadii = np.repeat(0, nProperLabels)
            for cluster in range(nProperLabels):
                InCluster = np.where(Colors == cluster)[0].tolist()
                DistInCluster = distM.iloc[InCluster, InCluster]
                DistSums = DistInCluster.sum(axis=0)
                Medoids[cluster] = InCluster[DistSums.idxmin()]
                ClusterRadii[cluster] = np.max(DistInCluster[:, DistSums.idxmin()])

            if respectSmallClusters:
                FSmallLabels = pd.Categorical(SmallLabels)
                SmallLabLevs = pd.to_numeric(FSmallLabels.categories)
                nSmallClusters = len(FSmallLabels.categories) - (SmallLabLevs[1] == 0)

                if nSmallClusters > 0:
                    for sclust in SmallLabLevs[SmallLabLevs != 0]:
                        InCluster = np.where(SmallLabels == sclust)[0].tolist()
                        if pamRespectsDendro:
                            onBr = np.unique(onBranch[InCluster])
                            if len(onBr) > 1:
                                msg = "Internal error: objects in a small cluster are marked to belong\n " \
                                        "to several large branches:" + str(onBr)
                                sys.exit(msg)

                            if onBr > 0:
                                basicOnBranch = branch_basicClusters[[onBr]]
                                labelsOnBranch = branchLabels[basicOnBranch]
                            else:
                                labelsOnBranch = None
                        else:
                            labelsOnBranch = list(range(nProperLabels))

                        DistInCluster = distM.iloc[InCluster, InCluster]

                        if len(labelsOnBranch) > 0:
                            if len(InCluster) > 1:
                                DistSums = DistInCluster.sum(axis=1)
                                smed = InCluster[DistSums.idxmin()]
                                DistToMeds = distM.iloc[Medoids[labelsOnBranch], smed]
                                closest = DistToMeds.idxmin()
                                DistToClosest = DistToMeds[closest]
                                closestLabel = labelsOnBranch[closest]
                                if DistToClosest < ClusterRadii[closestLabel] or DistToClosest < maxPamDist:
                                    Colors[InCluster] = closestLabel
                                    nPAMed = nPAMed + len(InCluster)
                            else:
                                Colors[InCluster] = -1
                        else:
                            Colors[InCluster] = -1

            Unlabeled = np.where(Colors == 0)[0].tolist()
            if len(Unlabeled > 0):
                for obj in Unlabeled:
                    if pamRespectsDendro:
                        onBr = onBranch[obj]
                        if onBr > 0:
                            basicOnBranch = branch_basicClusters[[onBr]]
                            labelsOnBranch = branchLabels[basicOnBranch]
                        else:
                            labelsOnBranch = None
                    else:
                        labelsOnBranch = list(range(nProperLabels))

                    if labelsOnBranch is not None:
                        UnassdToMedoidDist = distM.iloc[Medoids[labelsOnBranch], obj]
                        nearest = UnassdToMedoidDist.idxmin()
                        NearestCenterDist = UnassdToMedoidDist[nearest]
                        nearestMed = labelsOnBranch[nearest]
                        if NearestCenterDist < ClusterRadii[nearestMed] or NearestCenterDist < maxPamDist:
                            Colors[obj] = nearestMed
                            nPAMed = nPAMed + 1
                UnlabeledExist = (sum(Colors == 0) > 0)
        else:
            ClusterDiam = np.zeros((nProperLabels,))
            for cluster in range(nProperLabels):
                InCluster = np.where(Colors == (cluster + 1))[0].tolist()
                nInCluster = len(InCluster)
                DistInCluster = distM.iloc[InCluster, InCluster]
                if nInCluster > 1:
                    AveDistInClust = DistInCluster.sum(axis=1) / (nInCluster - 1)
                    AveDistInClust.reset_index(drop=True, inplace=True)
                    ClusterDiam[cluster] = AveDistInClust.max()
                else:
                    ClusterDiam[cluster] = 0

            ColorsX = Colors.copy()
            if respectSmallClusters:
                FSmallLabels = pd.Categorical(SmallLabels)
                SmallLabLevs = pd.to_numeric(FSmallLabels.categories)
                nSmallClusters = len(FSmallLabels.categories) - (SmallLabLevs[0] == 0)
                if nSmallClusters > 0:
                    if pamRespectsDendro:
                        for sclust in SmallLabLevs[SmallLabLevs != 0]:
                            InCluster = list(range(nPoints))[SmallLabels == sclust]
                            onBr = pd.unique(onBranch[InCluster])
                            if len(onBr) > 1:
                                msg = "Internal error: objects in a small cluster are marked to belong\n" \
                                        "to several large branches:" + str(onBr)
                                sys.exit(msg)
                            if onBr > 0:
                                basicOnBranch = branch_basicClusters[[onBr]]
                                labelsOnBranch = branchLabels[basicOnBranch]
                                useObjects = ColorsX in np.unique(labelsOnBranch)
                                DistSClustClust = distM.iloc[InCluster, useObjects]
                                MeanDist = DistSClustClust.mean(axis=0)
                                useColorsFac = pd.Categorical(ColorsX[useObjects])
                                # TODO
                                MeanMeanDist = MeanDist.groupby(
                                    'useColorsFac').mean()  # tapply(MeanDist, useColorsFac, mean)
                                nearest = MeanMeanDist.idxmin()
                                NearestDist = MeanMeanDist[nearest]
                                if np.logical_or(np.all(NearestDist < ClusterDiam[nearest]),
                                                    NearestDist < maxPamDist).tolist()[0]:
                                    Colors[InCluster] = nearest
                                    nPAMed = nPAMed + len(InCluster)
                                else:
                                    Colors[InCluster] = -1
                    else:
                        labelsOnBranch = list(range(nProperLabels))
                        useObjects = np.where(ColorsX != 0)[0].tolist()
                        for sclust in SmallLabLevs[SmallLabLevs != 0]:
                            InCluster = np.where(SmallLabels == sclust)[0].tolist()
                            DistSClustClust = distM.iloc[InCluster, useObjects]
                            MeanDist = DistSClustClust.mean(axis=0)
                            useColorsFac = pd.Categorical(ColorsX[useObjects])
                            MeanDist = pd.DataFrame({'MeanDist': MeanDist, 'useColorsFac': useColorsFac})
                            MeanMeanDist = MeanDist.groupby(
                                'useColorsFac').mean()  # tapply(MeanDist, useColorsFac, mean)
                            nearest = MeanMeanDist[['MeanDist']].idxmin().astype(int) - 1
                            NearestDist = MeanMeanDist[['MeanDist']].min()
                            if np.logical_or(np.all(NearestDist < ClusterDiam[nearest]),
                                                NearestDist < maxPamDist).tolist()[0]:
                                Colors[InCluster] = nearest
                                nPAMed = nPAMed + len(InCluster)
                            else:
                                Colors[InCluster] = -1
            Unlabeled = np.where(Colors == 0)[0].tolist()
            if len(Unlabeled) > 0:
                if pamRespectsDendro:
                    unlabOnBranch = Unlabeled[onBranch[Unlabeled] > 0]
                    for obj in unlabOnBranch:
                        onBr = onBranch[obj]
                        basicOnBranch = branch_basicClusters[[onBr]]
                        labelsOnBranch = branchLabels[basicOnBranch]
                        useObjects = ColorsX in np.unique(labelsOnBranch)
                        useColorsFac = pd.Categorical(ColorsX[useObjects])
                        UnassdToClustDist = distM.iloc[useObjects, obj].groupby(
                            'useColorsFac').mean()  # tapply(distM[useObjects, obj], useColorsFac, mean)
                        nearest = UnassdToClustDist.idxmin()
                        NearestClusterDist = UnassdToClustDist[nearest]
                        nearestLabel = pd.to_numeric(useColorsFac.categories[nearest])
                        if np.logical_or(np.all(NearestClusterDist < ClusterDiam[nearest]),
                                            NearestClusterDist < maxPamDist).tolist()[0]:
                            Colors[obj] = nearest
                            nPAMed = nPAMed + 1
                else:
                    useObjects = np.where(ColorsX != 0)[0].tolist()
                    useColorsFac = pd.Categorical(ColorsX[useObjects])
                    tmp = pd.DataFrame(distM.iloc[useObjects, Unlabeled])
                    tmp['group'] = useColorsFac
                    UnassdToClustDist = tmp.groupby(
                        ['group']).mean()  # apply(distM[useObjects, Unlabeled], 2, tapply, useColorsFac, mean)
                    nearest = np.subtract(UnassdToClustDist.idxmin(axis=0),
                                            np.ones(UnassdToClustDist.shape[1])).astype(
                        int)  # apply(UnassdToClustDist, 2, which.min)
                    nearestDist = UnassdToClustDist.min(axis=0)  # apply(UnassdToClustDist, 2, min)
                    nearestLabel = nearest + 1
                    sumAssign = np.sum(np.logical_or(nearestDist < ClusterDiam[nearest], nearestDist < maxPamDist))
                    assign = np.where(np.logical_or(nearestDist < ClusterDiam[nearest], nearestDist < maxPamDist))[
                        0].tolist()
                    tmp = [Unlabeled[x] for x in assign]
                    Colors[tmp] = [nearestLabel.iloc[x] for x in assign]
                    nPAMed = nPAMed + sumAssign

    Colors[np.where(Colors < 0)[0].tolist()] = 0
    UnlabeledExist = (np.count_nonzero(Colors == 0) > 0)
    NumLabs = list(map(int, Colors.copy()))
    Sizes = pd.DataFrame(NumLabs).value_counts().sort_index()
    OrdNumLabs = pd.DataFrame({"Name": NumLabs, "Value": np.repeat(1, len(NumLabs))})

    if UnlabeledExist:
        if len(Sizes) > 1:
            SizeRank = np.insert(stats.rankdata(-1 * Sizes[1:len(Sizes)], method='ordinal') + 1, 0, 1)
        else:
            SizeRank = 1
        for i in range(len(NumLabs)):
            OrdNumLabs.Value[i] = SizeRank[NumLabs[i]]
    else:
        SizeRank = stats.rankdata(-1 * Sizes[0:len(Sizes)], method='ordinal')
        for i in range(len(NumLabs)):
            OrdNumLabs.Value[i] = SizeRank[NumLabs[i]]

    OrdNumLabs.Value = OrdNumLabs.Value - UnlabeledExist
    return OrdNumLabs

def coreSizeFunc(BranchSize, minClusterSize):
    BaseCoreSize = minClusterSize / 2 + 1
    if BaseCoreSize < BranchSize:
        CoreSize = int(BaseCoreSize + np.sqrt(BranchSize - BaseCoreSize))
    else:
        CoreSize = BranchSize
    return CoreSize

def interpolate(data, index):
    i = round(index)
    n = len(data)
    if i < 1:
        return data[1]
    if i >= n:
        return data[n]
    r = index - i
    return data[i] * (1 - r) + data[i + 1] * r

def moduleEigengenes(expr, colors, impute=True, nPC=1,
                     align="along average", excludeGrey=False, grey="grey",
                     subHubs=True, softPower=6, scaleVar=True, trapErrors=False):
    """
    Calculates module eigengenes (1st principal component) of modules in a given single dataset.

    :param expr: Expression data for a single set in the form of a data frame where rows are samples and columns are genes (probes).
    :type expr: pandas dataframe
    :param colors: A list of the same length as the number of probes in expr, giving module color for all probes (genes). Color "grey" is reserved for unassigned genes.
    :type colors: list
    :param impute: If TRUE, expression data will be checked for the presence of NA entries and if the latter are present, numerical data will be imputed. (defualt = True)
    :type impute: bool
    :param nPC: Number of principal components and variance explained entries to be calculated. Note that only the first principal component is returned; the rest are used only for the calculation of proportion of variance explained. If given nPC is greater than 10, a warning is issued. (default = 1)
    :type nPC: int
    :param align: Controls whether eigengenes, whose orientation is undetermined, should be aligned with average expression (align = "along average") or left as they are (align = ""). Any other value will trigger an error. (default = "along average")
    :type align: str
    :param excludeGrey: Should the improper module consisting of 'grey' genes be excluded from the eigengenes (default = False)
    :type excludeGrey: bool
    :param grey: Value of colors designating the improper module. Note that if colors is a factor of numbers, the default value will be incorrect. (default = grey)
    :type grey: str
    :param subHubs: Controls whether hub genes should be substituted for missing eigengenes. If TRUE, each missing eigengene (i.e., eigengene whose calculation failed and the error was trapped) will be replaced by a weighted average of the most connected hub genes in the corresponding module. If this calculation fails, or if subHubs==FALSE, the value of trapErrors will determine whether the offending module will be removed or whether the function will issue an error and stop. (default = True)
    :type subHubs: bool
    :param softPower: The power used in soft-thresholding the adjacency matrix. Only used when the hubgene approximation is necessary because the principal component calculation failed. It must be non-negative. The default value should only be changed if there is a clear indication that it leads to incorrect results. (default = 6)
    :type softPower: int
    :param trapErrors: Controls handling of errors from that may arise when there are too many NA entries in expression data. If TRUE, errors from calling these functions will be trapped without abnormal exit. If FALSE, errors will cause the function to stop. Note, however, that subHubs takes precedence in the sense that if subHubs==TRUE and trapErrors==FALSE, an error will be issued only if both the principal component and the hubgene calculations have failed. (default = False)
    :type trapErrors: bool
    :param scaleVar: can be used to turn off scaling of the expression data before calculating the singular value decomposition. The scaling should only be turned off if the data has been scaled previously, in which case the function can run a bit faster. Note however that the function first imputes, then scales the expression data in each module. If the expression contain missing data, scaling outside of the function and letting the function impute missing data may lead to slightly different results than if the data is scaled within the function. (default = True)
    :type scaleVar: bool

    :return: A dictionary containing: "eigengenes": Module eigengenes in a dataframe, with each column corresponding to one eigengene. The columns are named by the corresponding color with an "ME" prepended, e.g., MEturquoise etc. If returnValidOnly==FALSE, module eigengenes whose calculation failed have all components set to NA. "averageExpr": If align == "along average", a dataframe containing average normalized expression in each module. The columns are named by the corresponding color with an "AE" prepended, e.g., AEturquoise etc. "varExplained": A dataframe in which each column corresponds to a module, with the component varExplained[PC, module] giving the variance of module module explained by the principal component no. PC. The calculation is exact irrespective of the number of computed principal components. At most 10 variance explained values are recorded in this dataframe. "nPC": A copy of the input nPC. "validMEs": A boolean vector. Each component (corresponding to the columns in data) is TRUE if the corresponding eigengene is valid, and FALSE if it is invalid. Valid eigengenes include both principal components and their hubgene approximations. When returnValidOnly==FALSE, by definition all returned eigengenes are valid and the entries of validMEs are all TRUE. "validColors": A copy of the input colors with entries corresponding to invalid modules set to grey if given, otherwise 0 if colors is numeric and "grey" otherwise. "allOK": Boolean flag signalling whether all eigengenes have been calculated correctly, either as principal components or as the hubgene average approximation. "allPC": Boolean flag signalling whether all returned eigengenes are principal components. "isPC": Boolean vector. Each component (corresponding to the columns in eigengenes) is TRUE if the corresponding eigengene is the first principal component and FALSE if it is the hubgene approximation or is invalid. "isHub": Boolean vector. Each component (corresponding to the columns in eigengenes) is TRUE if the corresponding eigengene is the hubgene approximation and FALSE if it is the first principal component or is invalid. "validAEs": Boolean vector. Each component (corresponding to the columns in eigengenes) is TRUE if the corresponding module average expression is valid. "allAEOK": Boolean flag signalling whether all returned module average expressions contain valid data. Note that returnValidOnly==TRUE does not imply allAEOK==TRUE: some invalid average expressions may be returned if their corresponding eigengenes have been calculated correctly.
    :rtype: dict
    """
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import scale

    cf.ci(f'Calculating {len(pd.Categorical(colors).categories)} module eigengenes in given set...', 'cyan')
    check = True
    pc = None
    returnValidOnly = trapErrors
    if all(isinstance(x, int) for x in colors):
        grey = 0
    if expr is None:
        sys.exit("moduleEigengenes: Error: expr is NULL.")
    if colors is None:
        sys.exit("moduleEigengenes: Error: colors is NULL.")
    if isinstance(expr, dict):
        expr = expr['data']
    if expr.shape is None or len(expr.shape) != 2:
        sys.exit("moduleEigengenes: Error: expr must be two-dimensional.")
    if expr.shape[1] != len(colors):
        sys.exit("moduleEigengenes: Error: ncol(expr) and length(colors) must be equal (one color per gene).")
    # TODO: "Argument 'colors' contains unused levels (empty modules). Use colors[, drop=TRUE] to get rid of them."
    if softPower < 0:
        sys.exit("softPower must be non-negative")
    maxVarExplained = 10
    if nPC > maxVarExplained:
        cf.cw(f"Given nPC is too large. Will use value {str(maxVarExplained)}", 'red')
    nVarExplained = min(nPC, maxVarExplained)
    modlevels = pd.Categorical(colors).categories
    if excludeGrey:
        if len(np.where(modlevels != grey)) > 0:
            modlevels = modlevels[np.where(modlevels != grey)]
        else:
            sys.exit("Color levels are empty. Possible reason: the only color is grey and grey module is excluded "
                        "from the calculation.")
    PrinComps = np.empty((expr.shape[0], len(modlevels)))
    PrinComps[:] = np.nan
    PrinComps = pd.DataFrame(PrinComps)
    averExpr = np.empty((expr.shape[0], len(modlevels)))
    averExpr[:] = np.nan
    averExpr = pd.DataFrame(averExpr)
    varExpl = np.empty((nVarExplained, len(modlevels)))
    varExpl[:] = np.nan
    varExpl = pd.DataFrame(varExpl)
    validMEs = np.repeat(True, len(modlevels))
    validAEs = np.repeat(False, len(modlevels))
    isPC = np.repeat(True, len(modlevels))
    isHub = np.repeat(False, len(modlevels))
    validColors = colors
    PrinComps.columns = ["ME" + str(modlevel) for modlevel in modlevels]
    averExpr.columns = ["AE" + str(modlevel) for modlevel in modlevels]

    if expr.index is not None:
        PrinComps.index = expr.index
        averExpr.index = expr.index
    for i in range(len(modlevels)):
        modulename = modlevels[i]
        restrict1 = (colors == modulename)
        datModule = expr.loc[:, restrict1].T #Gene Cell
        n = datModule.shape[0]
        p = datModule.shape[1]
        try:
            if datModule.shape[0] > 1 and impute:
                seedSaved = True
                if datModule.isnull().values.any():
                    # define imputer
                    imputer = KNNImputer(n_neighbors=np.min(10, datModule.shape[0] - 1))
                    # fit on the dataset
                    imputer.fit(datModule)
                    # transform the dataset
                    datModule = imputer.transform(
                        datModule)  # datModule = impute.knn(datModule, k = min(10, nrow(datModule) - 1))

            if scaleVar:
                datModule = pd.DataFrame(scale(datModule.T).T, index=datModule.index, columns=datModule.columns)

            u, d, v = np.linalg.svd(datModule)
            u = u[:, 0:min(n, p, nPC)]
            v = v[0:min(n, p, nPC), :]
            tmp = datModule.T.copy() # Cell Gene
            tmp[[str(x) for x in range(min(n, p, nVarExplained))]] = v[0:min(n, p, nVarExplained), :].T
            veMat = pd.DataFrame(np.corrcoef(tmp.T)).iloc[-1, :-1].T
            varExpl.iloc[0:min(n, p, nVarExplained), i] = (veMat ** 2).mean(axis=0)
            pc = v[0].tolist()
        except:
            if not subHubs:
                sys.exit("Error!")
            if subHubs:
                print(" ..principal component calculation for module", modulename,
                        "failed with the following error:", flush=True)
                print("     ..hub genes will be used instead of principal components.", flush=True)

                isPC[i] = False
                check = True
                try:
                    scaledExpr = pd.DataFrame(scale(datModule.T).T, index=datModule.index,
                                                columns=datModule.columns)
                    covEx = np.cov(scaledExpr) #Gene * Gene
                    covEx[not np.isfinite(covEx)] = 0
                    modAdj = np.abs(covEx) ** softPower #?
                    kIM = (modAdj.mean(axis=0)) ** 3 #?
                    if np.max(kIM) > 1:
                        kIM = kIM - 1 #?
                    kIM[np.where(kIM is None)] = 0
                    hub = np.argmax(kIM)
                    alignSign = np.sign(covEx[:, hub])
                    alignSign[np.where(alignSign is None)] = 0
                    isHub[i] = True
                    tmp = np.array(kIM * alignSign)
                    tmp.shape = scaledExpr.shape #?
                    pcxMat = scaledExpr * tmp / sum(kIM)
                    pcx = pcxMat.mean(axis=0)
                    varExpl[0, i] = np.mean(np.corrcoef(pcx, datModule.transpose()) ** 2)
                    pc = pcx
                except:
                    check = False
        if not check:
            if not trapErrors:
                sys.exit("Error!")
            print(" ..ME calculation of module", modulename, "failed with the following error:", flush=True)
            print("     ", pc, " ..the offending module has been removed.", flush=True)
            print(
                f"Warnning:Eigengene calculation of module {modulename} failed with the following error \n"
                f"{pc} The offending module has been removed.")
            validMEs[i] = False
            isPC[i] = False
            isHub[i] = False
            validColors[restrict1] = grey
        else:
            PrinComps.iloc[:, i] = pc
            try:
                if isPC[i]:
                    scaledExpr = scale(datModule.T) # Cell Gene
                averExpr.iloc[:, i] = scaledExpr.mean(axis=1) # Gene * modles
                if align == "along average":
                    corAve = np.corrcoef(averExpr.iloc[:, i], PrinComps.iloc[:, i])[0, 1]
                    if not np.isfinite(corAve):
                        corAve = 0
                    if corAve < 0:
                        PrinComps.iloc[:, i] = -1 * PrinComps.iloc[:, i]
                validAEs[i] = True
            except:
                if not trapErrors:
                    sys.exit("Error!")
                print(" ..Average expression calculation of module", modulename,
                        "failed with the following error:", flush=True)
                print(" ..the returned average expression vector will be invalid.", flush=True)

                print(f"WARNING:Average expression calculation of module {modulename} "
                        f"failed with the following error.\nThe returned average expression vector will "
                        f"be invalid.")

    allOK = (sum(np.logical_not(validMEs)) == 0)
    if returnValidOnly and sum(np.logical_not(validMEs)) > 0:
        PrinComps = PrinComps[:, validMEs]
        averExpr = averExpr[:, validMEs]
        varExpl = varExpl[:, validMEs]
        validMEs = np.repeat(True, PrinComps.shape[1])
        isPC = isPC[validMEs]
        isHub = isHub[validMEs]
        validAEs = validAEs[validMEs]

    allPC = (sum(np.logical_not(isPC)) == 0)
    allAEOK = (sum(np.logical_not(validAEs)) == 0)

    return {"eigengenes": PrinComps, "averageExpr": averExpr, "varExplained": varExpl, "nPC": nPC,
            "validMEs": validMEs, "validColors": validColors, "allOK": allOK, "allPC": allPC, "isPC": isPC,
            "isHub": isHub, "validAEs": validAEs, "allAEOK": allAEOK}
