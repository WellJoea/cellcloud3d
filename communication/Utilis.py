import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,bsr_matrix,csc_matrix, csgraph
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc,isspmatrix_coo
from termcolor import colored, cprint

def checkcsr(csr_cm):
    if not issparse(csr_cm):
        return csr_matrix(csr_cm)
    if isspmatrix_csc(csr_cm):
        return csr_cm.transpose()
    elif isspmatrix_coo(csr_cm):
        return csr_cm.tocsr()
    if not isspmatrix_csr(csr_cm):
        print(type(csr_cm))
        raise('the matrix must in csr_matrix type!')

def checkarray(mtx):
    if isinstance(mtx, np.ndarray):
        return mtx.copy()
    elif issparse(mtx):
        return mtx.toarray()
    elif isinstance(mtx, pd.DataFrame):
        return mtx.values
    elif isinstance(mtx, list):
        return np.array(mtx)
    else:
        raise('incorrect input data type.')

def checkdiag(mtx, diag=1, fill=True, verbose=False ):
    if not issparse(mtx):
        mtx = checkarray(mtx).copy()
        if not all( np.isclose(np.diag(mtx), diag)):
            if verbose:
                cprint(f"diagonal values of matrix are not equal to {diag}", 'red')
            if fill:
                if verbose:
                    cprint(f"fill diagonal values to {diag}", 'red')
                np.fill_diagonal(mtx, diag)
    else:
        mtx = mtx.copy()
        if not all( np.isclose( mtx.diagonal(), diag)):
            if verbose:
                cprint(f"diagonal values of sparse matrix are not equal to {diag}", 'red')
            if fill:
                if verbose:
                    cprint(f"fill diagonal values to {diag}", 'red')
                mtx.setdiag(diag)
    return mtx

def checksymmetric(adjMat, rtol=1e-05, atol=1e-08):
    if issparse(adjMat):
        #from scipy.linalg import issymmetric
        #return issymmetric(adjMat.astype(np.float64), atol=atol, rtol=rtol)
        adjMat = adjMat.toarray()
    return np.allclose(adjMat, adjMat.T, rtol=rtol, atol=atol)

def transsymmetric(mtx):
    if not checksymmetric(mtx):
        return (mtx + mtx.T)/2
    else:
        return mtx

def normalAdj(A):
    A = checkarray(A)
    assert A.shape[0] == A.shape[1]
    d = np.sum(A, axis = 1)
    d = 1/np.sqrt(d)
    D = np.diag(d)
    return D @ A @ D 

def scaleAdjMat(adjMat, axis=1):
    adjMat = adjMat.astype(np.float64)
    if axis==1:
        return adjMat/adjMat.sum(axis)[:,None]
    else:
        return adjMat/adjMat.sum(axis)[None,:]

def checkAdjMat(adjMat, amin=0, amax=1):
    shape = adjMat.shape
    assert len(shape) == 2, \
        cprint("adjacency is not two-dimensional", 'red')
    assert issubclass(adjMat.dtype.type, np.floating), \
        cprint("adjacency is not numeric", 'red')
    assert shape[0] == shape[1], \
        cprint("adjacency is not square", 'red')
    assert np.min(adjMat) >= amin and np.max(adjMat) <= amax, \
        cprint(f"some entries are not between {amin} and {amax}",'red')
    if issparse(adjMat):
        adjMat = adjMat.toarray()
    if not np.allclose(np.subtract(adjMat, adjMat.T), 0):
        cprint("Warnning: adjacency is not symmetric", "red")

def choose_representation(adata, use_rep=None, n_pcs=None):
    '''
    like scanpy.tools._utils._choose_representation
    '''
    use_rep = 'X_pca' if use_rep is None else use_rep
    n_pcs = 80 if n_pcs is None else n_pcs

    if use_rep in ['X']:
        X = adata.X
    elif use_rep in ['raw']:
        X = adata.raw.X
    elif use_rep in adata.obsm.keys():
        X = adata.obsm[use_rep]
        if n_pcs > X.shape[1]:
            cprint(f'{use_rep} does not have enough PCs. set n_pca = {X.shape[1]}...', 'red')
        X = X[:, :n_pcs]
    else:
        raise ValueError('Did not find valid `use_rep`.')
    return X

def corstate(x,y, methd='pearsonr'):
    from minepy import MINE
    import numpy as np
    import scipy.stats

    if methd =='pearsonr':
        return(scipy.stats.pearsonr(x, y)[0])
    elif methd =='spearmanr':
        return(scipy.stats.spearmanr(x, y)[0])
    elif methd =='kendalltau':
        return(scipy.stats.kendalltau(x, y)[0])
    elif methd=='mic':
        m = MINE()
        m.compute_score(x, y)
        return(m.mic())

def setBlockSize(matrixSize = None, blockSize=None, rectangularBlocks=True,  maxMemoryAllocation=None, overheadFactor=3):
    """
    find suitable block size for calculating soft power threshold
    """
    import resource
    if maxMemoryAllocation is None:
        soft_memory,hard_memory  = resource.getrlimit(resource.RLIMIT_AS)
        if hard_memory<0:
            hard_memory=50 * 1024**3
    else:
        hard_memory = maxMemoryAllocation / 5
    hard_memory = hard_memory / overheadFactor
    if  blockSize is None:
        if (not matrixSize is None) & rectangularBlocks:
            blockSize = np.floor(hard_memory /matrixSize) 
        else:
            blockSize = np.floor(np.sqrt(hard_memory))
    if (not matrixSize is None):
        blockRg = np.arange(0, matrixSize, int(blockSize))
        if blockRg[-1] < matrixSize:
            blockRg = np.append(blockRg, matrixSize)
        return [list(zip(blockRg[:-1], blockRg[1:])), blockSize]
    else:
        return [list(), blockSize]

def extend_colormap(value, under='#EEEEEE', over=-0.2, bits=256, color_name='my_cmp'):
    import matplotlib
    med_col = value
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    under = 'white' if under is None else under
    rgb = [ int(value[i:i + lv // 3], 16)/bits for i in range(0, lv, lv // 3)]

    if (over is None) or ( type(over) is int and over==0):
        return(matplotlib.colors.LinearSegmentedColormap.from_list(color_name, [under, med_col]))
    elif type(over) is int:
        rgb = [ i + over/bits for i in rgb ]
    elif type(over) is float and (0<= abs(over) <= 1) :
        rgb = [ i + over for i in rgb ]
    rgb = [ np.median([0, i, 1]) for i in rgb ]
    return(matplotlib.colors.LinearSegmentedColormap.from_list(color_name, [under, med_col, rgb]))

def df_sort_max0(df, ascending=False):
    df = df.copy()
    df['groups'] = pd.Categorical(df.idxmax(axis="columns"), categories=df.columns)
    dfs =[ v.sort_values(by=k, ascending=ascending) 
           for k,v in df.groupby(by='groups', sort=True)]
    dfs = pd.concat(dfs, axis=0)
    dfs.drop('groups',axis=1, inplace=True)
    return(dfs)

def df_sort_max(df, ascending=False, columns=None):
    cols = df.columns if columns is None else columns
    df = df[cols].copy()
    rowdf = df.idxmax(1).to_frame('max_col')
    rowdf['max_col'] = pd.Categorical(rowdf['max_col'], categories=cols)
    rowdf['max_val'] = df.max(1)
    rowdf.sort_values(by=['max_col', 'max_val'], 
                      ascending=[True, ascending],
                      inplace=True)

    return(df.loc[rowdf.index,:])

def get_meat_df(mean_df, pvalue_df=None, celltypes = None,  sort_max=True,
                min_v = None, min_p =None, max_p=None, drop_dup=True, drop_zero=True):
    mean_df = mean_df.fillna(0).copy()
    if not celltypes is None:
        import itertools
        columns = list(itertools.permutations(celltypes, 2))
        columns = [ '%s|%s'%(i) for i in columns ]
    elif 'rank' in mean_df.columns:
        columns = mean_df.loc[:,"rank":].columns[1:]
    elif 'is_integrin' in mean_df.columns:
        columns = mean_df.loc[:,"is_integrin":].columns[1:]
    else:
        columns = mean_df.filter(regex=r'\|', axis=1).columns

    sig_mean = mean_df[columns].copy()
    min_p = 0 if min_p is None else min_p
    max_p = 1 if max_p is None else max_p
    if not pvalue_df is None:
        pvalue_df = pvalue_df.fillna(0).copy()
        inter_id = np.intersect1d(mean_df['id_cp_interaction'], pvalue_df['id_cp_interaction'])
        mean_id_uniq = np.setdiff1d(mean_df['id_cp_interaction'], pvalue_df['id_cp_interaction'])
        pvalue_id_uniq = np.setdiff1d(pvalue_df['id_cp_interaction'], mean_df['id_cp_interaction'])
        if len(mean_id_uniq) >0:
            print(f'{len(mean_id_uniq)} unique interactions was found in mean value file and will be remove in download analysis.')
            print(mean_id_uniq)
        if len(pvalue_id_uniq) >0:
            print(f'{len(pvalue_id_uniq)} unique interactions was found in pvalue file and will be remove in download analysis.')
            print(pvalue_id_uniq)
        pvalue_df = pvalue_df[columns]
        sig_mean[(pvalue_df<min_p)] = 0
        sig_mean[(pvalue_df>max_p)] = 0
        
        pvalue_df['interacting_pair'] = mean_df['interacting_pair'].copy()
        if drop_dup:
            pvalue_df = pvalue_df.drop_duplicates()
        pvalue_df.set_index('interacting_pair', inplace=True)

    sig_mean['interacting_pair'] = mean_df['interacting_pair'].copy()
    if drop_dup:
        sig_mean = sig_mean.drop_duplicates()
    sig_mean.set_index('interacting_pair', inplace=True)

    if not min_v is None:
        sig_mean[(sig_mean<min_v)] = 0
    if drop_zero:
        sig_mean = sig_mean[(sig_mean.sum(1)>0)].copy()

    if sort_max:
        sig_mean = df_sort_max(sig_mean, ascending=False)
    if pvalue_df is None:
        return(sig_mean)
    else:
        return((sig_mean, pvalue_df.loc[sig_mean.index, sig_mean.columns]))

def row_max_in_col(df, ascending=False):
    df = df.copy()
    df['groups'] = pd.Categorical(df.idxmax(axis="columns"), categories=df.columns)
    dfs =[ v.sort_values(by=k, ascending=ascending) 
           for k,v in df.groupby(by='groups', sort=True)]
    dfs = pd.concat(dfs, axis=0)
    #dfs.drop('groups',axis=1, inplace=True)
    df_max_group = dfs.pop('groups').to_frame()
    return(dfs, df_max_group)