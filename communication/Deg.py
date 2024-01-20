import numpy as np
import pandas as pd

def _group_wilcoxon(X, Y, axis=0, method='mhu'):
    from scipy.stats import mannwhitneyu, ranksums
    from scipy import sparse
    
    if sparse.issparse(X):
        X = X.toarray()
    if sparse.issparse(Y):
        Y = Y.toarray()

    if method == 'mhu':
        U, p = mannwhitneyu(X, Y, alternative='two-sided', method='auto', axis=axis)
    elif method == 'rs':
        U, p = ranksums(X, Y, alternative='two-sided', axis=axis)
    return p

def _group_logmean(X, base=2, bairs=1):
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()
    igroup_mean = np.expm1(X.copy())
    return np.log2(igroup_mean.mean(0) + bairs)/np.log2(base)

def _group_fraction(X,thresh_min = 0, axis=0):
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()
    return np.sum(X >thresh_min,axis=axis)/X.shape[axis]

def _group_padj(pvals, n_genes=None, corr_method = 'benjamini-hochberg'):
    if corr_method == 'benjamini-hochberg':
        from statsmodels.stats.multitest import multipletests
        pvals[np.isnan(pvals)] = 1
        _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    elif corr_method == 'bonferroni':
        pvals_adj = np.minimum(pvals * n_genes, 1.0)
    return pvals_adj
                
def DEGovr(adataI, 
           groupby, 
           use_raw=True, 
           only_pos = False,
           min_in_group_fraction=0.005,
           min_fold_change=0.1,
           max_out_group_fraction=1,
           max_padj=0.2,
           test_use = 'wilcox',
           use_filter =True,
           base=2,
           bairs=1):
    from termcolor import colored, cprint
    from tqdm import tqdm

    adataI = adataI.copy().raw.to_adata() if use_raw else adataI.copy()
    try:
        groups = adataI.obs[groupby].cat.categories
    except:
        groups = adataI.obs[groupby].unique()
    
    degs = pd.DataFrame()
    for igroup in tqdm(groups, colour='green', bar_format='{l_bar}{bar}{r_bar}\n'):
        print(f"computing cluster", colored(f'{igroup}.', 'red', attrs=['bold']))
        in_adata = adataI[adataI.obs[groupby]==igroup,:].copy()
        in_group_mena = _group_logmean(in_adata.X, base=base, bairs=bairs)
        in_group_frac = _group_fraction(in_adata.X)
        
        out_adata = adataI[adataI.obs[groupby]!=igroup,:].copy()
        out_group_mena = _group_logmean(out_adata.X, base=base, bairs=bairs)
        out_group_frac = _group_fraction(out_adata.X)    
        ilog2fc = in_group_mena - out_group_mena

        if use_filter:
            keep_gene = (in_group_frac>=min_in_group_fraction) & \
                         (out_group_frac<=max_out_group_fraction)
            if only_pos:
                keep_gene = keep_gene & (ilog2fc>=min_fold_change)
            else:
                keep_gene = keep_gene & (np.abs(ilog2fc)>=min_fold_change)
            Keep_idx = np.where(keep_gene)[0]
        else:
            Keep_idx = np.arange(in_adata.shape[1])
        p_value = _group_wilcoxon(in_adata.X[:,Keep_idx], 
                                  out_adata.X[:,Keep_idx], axis=0, method='mhu')
        padj = _group_padj(p_value)
        ideg = pd.DataFrame({'gene': adataI.var_names[Keep_idx],
                             'cluster': igroup,
                             'avg_log2FC' : ilog2fc[Keep_idx],
                             'pct.1': in_group_frac[Keep_idx],
                             'pct.2': out_group_frac[Keep_idx],
                             'p_val': p_value,
                             'p_val_adj': padj})
        ideg.sort_values(by=['avg_log2FC', 'pct.1', 'pct.2'],
                         ascending=[False, False, True],
                         inplace=True)
        degs = degs.append(ideg)
    return degs[(degs['p_val_adj']<=max_padj)]