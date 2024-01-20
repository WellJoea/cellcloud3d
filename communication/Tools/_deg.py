import numpy as np
import pandas as pd
from termcolor import colored, cprint
from tqdm import tqdm

class diffexp():
    def __init__(self, adata, inplace=False):
        self.adata = adata if inplace else adata.copy()
        self.deg = {}

    def avglogfc(self,
                groupby,
                adata=None,
                use_raw=True,
                base=2,
                bairs=1):
        
        adataI = self.adata if adata is None else adata
        adataI = adataI.raw.to_adata() if use_raw else adataI
        try:
            groups = adataI.obs[groupby].cat.categories
        except:
            groups = adataI.obs[groupby].unique()

        fcs = pd.DataFrame()
        for igroup in tqdm(groups, colour='green', bar_format='{l_bar}{bar}{r_bar}\n'):
            print(f"computing cluster avglogfc", colored(f'{igroup}.', 'red', attrs=['bold']))
            in_adata = adataI[adataI.obs[groupby]==igroup,:].copy()
            in_group_mena = self._group_logmean(in_adata.X, base=base, bairs=bairs)
            in_group_frac = self._group_fraction(in_adata.X)
            
            out_adata = adataI[adataI.obs[groupby]!=igroup,:].copy()
            out_group_mena = self._group_logmean(out_adata.X, base=base, bairs=bairs)
            out_group_frac = self._group_fraction(out_adata.X)
            ilog2fc = in_group_mena - out_group_mena
            ifc = pd.DataFrame(data={'gene': adataI.var_names,
                                     'cluster': igroup,
                                     'avg_log2FC': ilog2fc,
                                     'avg_log2': in_group_mena,
                                     'pct.1': in_group_frac,
                                     'pct.2': out_group_frac})
            fcs = fcs.append(ifc)
        return fcs

    def depvalue(self,
                groupby,
                adata=None,
                use_raw=True,
                gene=None,
                test_use = 'wilcox',
                rst_method = 'mhu',
                corr_method= 'benjamini-hochberg',
                ):
        
        adataI = self.adata if adata is None else adata
        adataI = adataI.raw.to_adata() if use_raw else adataI
        try:
            groups = adataI.obs[groupby].cat.categories
        except:
            groups = adataI.obs[groupby].unique()

        pval = pd.DataFrame()
        for igroup in tqdm(groups, colour='green', bar_format='{l_bar}{bar}{r_bar}\n'):
            print(f"computing cluster P values", colored(f'{igroup}.', 'red', attrs=['bold']))
            in_adata = adataI[adataI.obs[groupby]==igroup,:].copy()
            out_adata = adataI[adataI.obs[groupby]!=igroup,:].copy()
            if gene is None:
                genes = in_adata.var_names
            elif isinstance(gene, dict):
                genes = gene[igroup]
            else:
                genes = gene

            if test_use=='wilcox':
                _p = self._group_wilcoxon(in_adata[:, genes].X,
                                            out_adata[:, genes].X,
                                            axis=0,
                                            method=rst_method)
            _padj = self._group_padj(_p, corr_method = corr_method)
            ipp = pd.DataFrame({'gene': genes,
                                'cluster': igroup,
                                'p_val': _p,
                                'p_val_adj': _padj})
            pval = pval.append(ipp)
        return pval

    def degovr(self, 
                groupby, 
                use_raw=True, 
                only_pos = False,
                min_in_group_fraction=0.005,
                min_fold_change=0.05,
                max_out_group_fraction=1,
                #max_padj=0.2,
                test_use = 'wilcox',
                rst_method = 'mhu',
                use_filter =True,
                force=False,
                base=2,
                bairs=1):

        adataI = self.adata.raw.to_adata() if use_raw else self.adata.copy()
        try:
            groups = adataI.obs[groupby].cat.categories
        except:
            groups = adataI.obs[groupby].unique()

        if (not force) and (groupby in self.deg.keys()):
            cprint(f'{groupby} has been computed and exit!', 'red')
            cprint(f'use `force=True` to recomputed!', 'red')
        else:
            self.deg[groupby] = {}
            degs = self.deg[groupby]
            fcs = self.avglogfc(groupby, use_raw=use_raw, base=base, bairs=bairs)
            degs['fcs'] =fcs

            if use_filter:
                keeps = (fcs['pct.1']>=min_in_group_fraction) & \
                            (fcs['pct.2']<=max_out_group_fraction)
                if only_pos:
                    keeps = keeps & (fcs['avg_log2FC']>=min_fold_change)
                else:
                    keeps = keeps & (fcs['avg_log2FC'].abs()>=min_fold_change)
                genes = fcs[keeps].groupby(by='cluster')['gene'].apply(list).to_dict()
            else:
                genes = None
            pval = self.depvalue(groupby,
                                 use_raw=use_raw,
                                 gene=genes,
                                 test_use = test_use,
                                 rst_method = rst_method)
            degs['deg'] = pd.merge(fcs, pval, 
                                    on =['cluster', 'gene'],
                                    how='left',
                                    left_index=False,
                                    right_index=False)
            degs['deg'][['p_val', 'p_val_adj']] = degs['deg'][['p_val', 'p_val_adj']].fillna(1.)
            degs['deg']['cluster'] = pd.Categorical(degs['deg']['cluster'], 
                                                    categories=groups)

            degs['deg'].sort_values(by=['cluster','avg_log2FC', 'pct.1', 'pct.2'],
                                    ascending=[True, False, False, True],
                                    inplace=True)


    @staticmethod
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

    @staticmethod
    def _group_logmean(X, base=2, bairs=1):
        from scipy import sparse
        if sparse.issparse(X):
            X = X.toarray()
        igroup_mean = np.expm1(X.copy())
        return np.log2(igroup_mean.mean(0) + bairs)/np.log2(base)

    @staticmethod
    def _group_fraction(X,thresh_min = 0, axis=0):
        from scipy import sparse
        if sparse.issparse(X):
            X = X.toarray()
        return np.sum(X >thresh_min,axis=axis)/X.shape[axis]

    @staticmethod
    def _group_padj(pvals, n_genes=None, corr_method = 'benjamini-hochberg'):
        if corr_method == 'benjamini-hochberg':
            from statsmodels.stats.multitest import multipletests
            pvals[np.isnan(pvals)] = 1
            _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        elif corr_method == 'bonferroni':
            pvals_adj = np.minimum(pvals * n_genes, 1.0)
        return pvals_adj
