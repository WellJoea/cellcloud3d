import scanpy as sc
import anndata as ad
import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse import issparse
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

def _default_kargs(_func):
    import inspect
    signature = inspect.signature(_func)
    return { k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty}

def inplaceadata(_func):
    from functools import wraps
    @wraps(_func)
    def wrapper(*args, **kargs):
        rkargs = _default_kargs(_func)
        rkargs.update(kargs)
        adata = args[0]
        import anndata
        if isinstance(adata, anndata._core.anndata.AnnData):
            if ('inplace' in rkargs):
                inplace = rkargs['inplace']
            else:
                inplace = True

            adata = adata if inplace else adata.copy()
            Fun = _func(adata, *args[1:], **rkargs)
            if not inplace:
                return adata
        else:
            Fun = _func(adata, *args[1:], **rkargs)
            return Fun
    return wrapper

def scale_array( X:np.ndarray,
                zero_center: bool = True,
                max_value: Optional[float] = None,
                axis: Optional[int] = 0,
                copy: bool = False,
                verbose: Optional[int] =1,
    ):
    if issparse(X):
        X = X.toarray()
    if copy:
        X = X.copy()
    if (verbose) and (not zero_center) and (max_value is not None):
        print( "... be careful when using `max_value` " "without `zero_center`.")

    if (verbose) and np.issubdtype(X.dtype, np.integer):
        print('... as scaling leads to float results, integer '
             'input is cast to float, returning copy.'
              'or you forget to normalize.')
        X = X.astype(float)

    mean = np.expand_dims(np.mean(X, axis=axis), axis=axis)
    std  = np.expand_dims(np.std(X, axis=axis), axis=axis)
    std[std == 0] = 1

    if zero_center:
        X -= mean
    X /= std

    # do the clipping
    if max_value is not None:
        (verbose>1) and print(f"... clipping at max_value {max_value}")
        X[X > max_value] = max_value

    return X

@inplaceadata
def normalize(adata, inplace=True,  dolog=True, target_sum=1e4, **kargs):
    sc.pp.normalize_total( adata, target_sum=target_sum, **kargs)
    if dolog:
        sc.pp.log1p(adata)

@inplaceadata
def findHGVsc(adata, batch_key=None, minnbat=2, min_mean=0.0125, min_disp=0.5, n_top_genes=None, 
              max_disp=np.inf, subset=False, flavor='seurat', inplace=True,
              max_mean=4, layer=None, n_bins=20, check_values=True, 
              span=0.3):
    sc.pp.highly_variable_genes(adata, batch_key = batch_key, min_mean=min_mean, min_disp=min_disp,
                                inplace=True,
                                flavor=flavor,n_top_genes=n_top_genes, span=span,check_values=check_values,
                                layer=layer, n_bins=n_bins,max_mean=max_mean, subset=subset, max_disp=max_disp)
    if batch_key:
        print(adata.var.highly_variable_nbatches.value_counts())
        if (not minnbat is None):
            adata.var.highly_variable = ((adata.var.highly_variable_nbatches>=minnbat) & (adata.var.highly_variable))
    print(f'Compute HGVs: {adata.var.highly_variable.sum()}')

# @inplaceadata
def dropFuncGene(adata, dropMT=False, inplace=True, dropRibo=False, dropHb=False):
    if  not 'mt' in adata.var.columns:
        adata.var['mt'] = adata.var_names.str.contains('^M[Tt]', regex=True)

    if not 'ribo' in adata.var.columns:
        adata.var['ribo'] = adata.var_names.str.contains('^RP[SL]', regex=True)

    if not 'hb' in adata.var.columns:
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', regex=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)

    MTinHGVs = adata.var['mt'] & (adata.var.highly_variable)
    RiboinHGVs = adata.var['ribo'] & (adata.var.highly_variable)
    HbHGVs = adata.var['hb'] & (adata.var.highly_variable)
    print('MT in HGVs: %s'%MTinHGVs.sum())
    print('Ribo in HGVs: %s'%RiboinHGVs.sum())
    print('Hb in HGVs: %s'%HbHGVs.sum())

    if dropMT:
        adata.var.highly_variable = ((~adata.var['mt']) & (adata.var.highly_variable))
    if dropRibo:
        adata.var.highly_variable = ((~adata.var['ribo']) & (adata.var.highly_variable))
    if dropHb:
        adata.var.highly_variable = ((~adata.var['hb']) & (adata.var.highly_variable))

    print(f'keep HGVs: {adata.var.highly_variable.sum()}')

@inplaceadata
def Scale(adata, inplace=True, n_jobs=50, zero_center=True, max_value=None, vargress=None):
    #adata = adata[:, adata.var.highly_variable] if usehvgs else adata
    print(f'Scale: set the max_value to {max_value}.')
    if (not vargress is None) and (len(vargress)>0):
        #vargress=['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'CC_diff']
        sc.pp.regress_out(adata, vargress, n_jobs =n_jobs) #overcorrected
    sc.pp.scale(adata, zero_center=zero_center, max_value=max_value)
    '''
    sc.pp.recipe_zheng17(adata)
    sc.pp.recipe_weinreb17
    sc.pp.recipe_seurat
    '''

def NormScale(adata, batch_key=None, donormal=True, doscale=True, n_top_genes=None, n_jobs=None,
                local_features = None, dolog=True, target_sum=1e4,
                dropMT=False, dropRibo=False, dropHb=False,
                savecounts=False, saveraw=False, vargress=None,
                max_value = 10,
                usehvgs=True, **kwargs):
    adata = adata.copy()
    if savecounts:
        adata.layers['counts'] = adata.X.copy()
    if saveraw:
        adata.raw = adata

    if donormal:
        normalize(adata, inplace=True, dolog=dolog, target_sum=target_sum)

    if local_features is None:
        findHGVsc(adata, inplace=True, batch_key=batch_key, n_top_genes=n_top_genes, **kwargs)
    else:
        adata.var["highly_variable"] = adata.var_names.isin(local_features)
        print(f'the number of local features is {len(local_features)}.')

    dropFuncGene(adata, inplace=True, dropMT=dropMT, dropRibo=dropRibo, dropHb=dropHb)
    if local_features is None:
        sc.pl.highly_variable_genes(adata)

    adata = adata[:, adata.var.highly_variable] if usehvgs else adata
    if doscale:
        Scale(adata, n_jobs=n_jobs, inplace=True,  max_value=max_value, vargress=vargress)
    return adata

def highly_variable_genes_pr(adata, markers=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    hvgs = adata.var["highly_variable"]

    ax.scatter(
        adata.var["mean_counts"], adata.var["residual_variances"], s=3, edgecolor="none"
    )
    ax.scatter(
        adata.var["mean_counts"][hvgs],
        adata.var["residual_variances"][hvgs],
        c="tab:red",
        label="selected genes",
        s=3,
        edgecolor="none",
    )
    if not markers is None:
        ax.scatter(
            adata.var["mean_counts"][np.isin(adata.var_names, markers)],
            adata.var["residual_variances"][np.isin(adata.var_names, markers)],
            c="k",
            label="known marker genes",
            s=10,
            edgecolor="none",
        )
    ax.set_xscale("log")
    ax.set_xlabel("mean expression")
    ax.set_yscale("log")
    ax.set_ylabel("residual variance")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.legend()
    plt.show()

@inplaceadata
def HVG_PR(adata, batch_key=None, n_top_genes =None,  minnbat=2, subset=False,
            flavor='pearson_residuals',  layer=None, 
            inplace=True,
            dropMT=False, dropRibo=False, dropHb=False, **kargs):

    sc.experimental.pp.highly_variable_genes(
                adata, batch_key = batch_key,
                flavor=flavor, layer=layer, n_top_genes=n_top_genes, subset=subset,**kargs)

    if batch_key:
        print(adata.var.highly_variable_nbatches.value_counts())
        if (not minnbat is None):
            adata.var.highly_variable = ((adata.var.highly_variable_nbatches>=minnbat) & (adata.var.highly_variable))

    MTinHGVs = adata.var['mt'] & (adata.var.highly_variable)
    RiboinHGVs = adata.var['ribo'] & (adata.var.highly_variable)
    HbHGVs = adata.var['hb'] & (adata.var.highly_variable)
    print('MT in HGVs: %s'%MTinHGVs.sum())
    print('Ribo in HGVs: %s'%RiboinHGVs.sum())
    print('Hb in HGVs: %s'%HbHGVs.sum())

    if dropMT:
        adata.var.highly_variable = ((~adata.var['mt']) & (adata.var.highly_variable))
    if dropRibo:
        adata.var.highly_variable = ((~adata.var['ribo']) & (adata.var.highly_variable))
    if dropRibo:
        adata.var.highly_variable = ((~adata.var['hb']) & (adata.var.highly_variable))

    print(adata.var.highly_variable.sum())
    highly_variable_genes_pr(adata)

@inplaceadata
def dropFuncGene1(adata, dropMT=False, inplace=True, dropRibo=False, dropHb=False):
    if  not 'mt' in adata.var.columns:
        adata.var['mt'] = adata.var_names.str.contains('^M[Tt]', regex=True)

    if not 'ribo' in adata.var.columns:
        adata.var['ribo'] = adata.var_names.str.contains('^RP[SL]', regex=True)

    if not 'hb' in adata.var.columns:
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', regex=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)

    MTinHGVs = adata.var['mt']
    RiboinHGVs = adata.var['ribo']
    HbHGVs = adata.var['hb']
    print(f'MT in var genes: {MTinHGVs.sum()}')
    print(f'Ribo in var genes: {RiboinHGVs.sum()}')
    print(f'Hb in var genes: {HbHGVs.sum()}')

    adata.var['dropfuncgene'] = False
    if dropMT:
        adata.var.dropfuncgene = (adata.var['mt']) & (adata.var.dropfuncgene)
    if dropRibo:
        adata.var.dropfuncgene = (adata.var['ribo']) & (adata.var.dropfuncgene)
    if dropHb:
        adata.var.dropfuncgene = (adata.var['hb']) & (adata.var.dropfuncgene)