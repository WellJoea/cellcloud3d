import scanpy as sc
import anndata as ad
import scvelo as scv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
from scipy import sparse
from anndata import AnnData
import os

import sys
import importlib
sys.path.append('/home/gpfs/home/wulab17/JupyterCode')
import SCFunc
importlib.reload(SCFunc)
from SCFunc import *

def velos(adataV, replace=False, n_jobs=10, 
          min_r2=0.01, r2_adjusted=None, perc=[5,95],
          use_raw=False, use_highly_variable=True, ):
    adataV = adataV if replace else adataV.copy()
    #deterministic, dynamical,stochastic
    scv.tl.velocity(adataV, vkey='steady_state_velocity',
                    min_r2=min_r2, r2_adjusted=r2_adjusted, perc=perc,
                    use_highly_variable=use_highly_variable,
                    use_raw=use_raw, mode='steady_state')
    scv.tl.velocity_graph(adataV, vkey='steady_state_velocity')

    scv.tl.velocity(adataV, vkey='stochastic_velocity', 
                    min_r2=min_r2, r2_adjusted=r2_adjusted, perc=perc,
                    use_highly_variable=use_highly_variable,
                    use_raw=use_raw, mode='stochastic')
    scv.tl.velocity_graph(adataV, vkey='stochastic_velocity')

    scv.tl.recover_dynamics(adataV, n_jobs=n_jobs)
    scv.tl.velocity(adataV, vkey='dynamical_velocity', 
                    min_r2=min_r2, r2_adjusted=r2_adjusted, perc=perc,
                    use_highly_variable=use_highly_variable,
                    use_raw=use_raw, mode='dynamical')
    scv.tl.velocity_graph(adataV, vkey='dynamical_velocity')
    return adataV

def modelcompare(adata, replace=False):
    adata = adata if replace else adata.copy()
    vkey_ss = 'steady_state_velocity'
    vkey_sm = 'stochastic_velocity'
    vkey_dm = 'dynamical_velocity'
    scv.tl.velocity_confidence(adata, vkey=vkey_ss)
    scv.tl.velocity_confidence(adata, vkey=vkey_sm)
    scv.tl.velocity_confidence(adata, vkey=vkey_dm)

    scv.pl.hist([adata.obs[vkey_dm + '_confidence'].values,
                 adata.obs[vkey_sm + '_confidence'].values,
                 adata.obs[vkey_ss + '_confidence'].values],
                labels=['dynamical model', 'stochastic model', 'steady-state model'], kde=True,
                bins=200, xlim=[0, 1], fontsize=18, legend_fontsize=16)
    return adata

def plvelo(veloadata, colorby = 'CellType', legend_loc='none', dpi=300, fscale=8, basis='umap',
           vkeys = ['steady_state_velocity', 'stochastic_velocity','dynamical_velocity'], 
           save=None, show=True):
    ncols = len(vkeys)
    fig, axes = plt.subplots(2, ncols, figsize=(fscale*ncols,fscale*2))
    for _i, ivkey in enumerate(vkeys):
        if ncols >1:
            axi0 = axes[0, _i]
            axi1 = axes[1, _i]
        else:
            axi0 = axes[0]
            axi1 = axes[1]
            
        scv.pl.velocity_embedding_grid(veloadata, vkey=ivkey, basis=basis,
                                       color=colorby, scale=1,projection='2d',
                                       arrow_length=2.5, arrow_size=1, dpi=dpi,
                                       legend_loc=legend_loc, alpha=0.5,
                                       title=f'{colorby}_{ivkey}',
                                       figsize= (fscale,fscale),
                                       size=None, fontsize=10, 
                                       show=False, ax=axi0)

        scv.pl.velocity_embedding_stream(veloadata, vkey=ivkey, basis=basis,
                                         color=colorby,add_rug=None, projection='2d',
                                         legend_loc=legend_loc,
                                         density=4,
                                         title=f'{colorby}_{ivkey}',
                                         fontsize=10, add_text=colorby,
                                         dpi=dpi, 
                                         figsize= (fscale,fscale),
                                         show=False, ax=axi1)
        axi0.grid(False)
        axi1.grid(False)

    plt.tight_layout()
        #Scvplot().scv3d(veloadata, mode=mode, basis='umap_3d', groupby=colorby, vkey='velocity') 
    if show:
        plt.show()
    if save is None :
        out_pre = f'{colorby}.{basis}.velocity_embedding_grid_stream'
        try:
            plt.savefig(f'{out_pre}.pdf')
        except:
            plt.savefig(f'{out_pre}.svg')
            plt.savefig(f'{out_pre}.png')
    elif save:
        plt.savefig(save)
    else:
        return (fig, axes)

def plveloEach(veloadata, colorby = 'CellType', basis='umap', legend_loc='none', show=False,
               figsize=(10,10)):
    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = False
    for _i, ivkey in enumerate(['steady_state_velocity', 'stochastic_velocity','dynamical_velocity']):
        scv.pl.velocity_embedding_grid(veloadata, vkey=ivkey, basis=basis,
                                       color=colorby, scale=1,projection='2d',
                                       arrow_length=2.5, arrow_size=1, dpi=300,
                                       legend_loc=legend_loc, alpha=0.5,
                                       title=f'{colorby}_{ivkey}',
                                       figsize= figsize,
                                       size=None, 
                                       fontsize=10, 
                                       show=show,
                                       save=f'{colorby}.{ivkey}.{basis}.velocity_embedding_grid.pdf')

        scv.pl.velocity_embedding_stream(veloadata, vkey=ivkey, basis=basis,
                                         color=colorby,add_rug=None, projection='2d',
                                         legend_loc=legend_loc,
                                         density=4,
                                         title=f'{colorby}_{ivkey}',
                                         fontsize=10, add_text=colorby,
                                         dpi=300,
                                         figsize= figsize,
                                         show=show,
                                         save=f'{colorby}.{ivkey}.{basis}.velocity_embedding_stream.svg')


def plveloEach_3d(veloadata, colorby = 'CellType', basis='umap_3d', legend_loc='none', show=False,
                   figsize=(10,10)):
    for _i, ivkey in enumerate(['steady_state_velocity', 'stochastic_velocity','dynamical_velocity']):
        Scvplot().scv3d(veloadata, mode=ivkey, basis=basis, 
                        groupby=colorby,show=show, vkey=ivkey)

def plveloback(veloadata, scdata, mode='dynamical', colorby = 'celltype_4'):
    fig, axes = plt.subplots(1, 1, figsize=(10,10))
    sc.pl.umap(scdata, ax=axes,  show=False)
    scv.pl.velocity_embedding_grid(veloadata, basis='umap',color=colorby, scale=1,projection='2d',
                                   arrow_length=2.5, arrow_size=1, legend_loc='on data',alpha=1,
                                   size=None, ax=axes, show=False)
    plt.savefig(f'{colorby}.{mode}.cones.grid.2d.add.backgroud.pdf')

    fig, axes = plt.subplots(1, 1, figsize=(10,10))
    sc.pl.umap(scdata, ax=axes,  show=False)
    scv.pl.velocity_embedding_stream(veloadata, basis='umap', color=colorby,
                                  add_rug=None, projection='2d',
                                  legend_loc='on data', show=False, ax=axes)
    try:
        plt.savefig(f'{colorby}.{mode}.cones.stream.2d.add.backgroud.pdf')
    except:
        plt.savefig(f'{colorby}.{mode}.cones.stream.2d.add.backgroud.svg')

def Dopseudotime(adataV, root_cells, end_cells=None, replace=False, n_dcs=30,
                vkey='velocity', heatmapkey=None, col_color='CellType'):
    adataV = adataV if replace else adataV.copy()
    adataV.uns['root_cells'] = root_cells
    adataV.uns['iroot'] = root_cells
    scv.tl.velocity_pseudotime(adataV, vkey=vkey,
                               #end_key= end_cells,
                               n_dcs=n_dcs, root_key=root_cells)
    scv.tl.latent_time(adataV, vkey=vkey, root_key=root_cells)
    sc.tl.diffmap(adataV, n_comps=n_dcs)
    sc.tl.dpt(adataV, n_branchings=0, n_dcs=n_dcs)
                 
    sc.pl.umap(adataV, color= ['dpt_pseudotime',f'{vkey}_pseudotime', 'latent_time'], 
               cmap='gnuplot', save=f'{col_color}_{vkey}_dpt_velocity_latent_pseudotime.pdf')
    
    if not heatmapkey is None:
        top_genes = adataV.var['fit_likelihood'].sort_values(ascending=False).index[:100]
        scv.pl.heatmap(adataV, var_names=top_genes, sortby=heatmapkey, 
                       col_color=col_color, n_convolve=100)
    return adataV

def lowessfit(trajdata, xcol, ycol, ax, colorby='CellType', size = 0.05,
                alpha= 1, frac=0.5, it=3):
    timedf = trajdata.obs.sort_values(by=xcol) 
    colorcol = trajdata.uns[f'{colorby}_colors'] 
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from sklearn.metrics import r2_score
    
    for _n, _c in enumerate(timedf[colorby].cat.categories):
        _idf = timedf[(timedf[colorby]==_c)]
        x= _idf[xcol]
        y= _idf[ycol]
        ysi = lowess(y, x, frac=frac, it=it, return_sorted=False)
        r2i = r2_score(x, ysi)
        #xsii = lowess(x, y, frac=frac, it=it, return_sorted=False)
        #r2ii = r2_score(y, xsii)

        ax.scatter(x, y,  c=colorcol[_n], s=0.08, alpha=0.3, label=_c)
        ax.scatter(x, ysi, c=colorcol[_n], s=1, alpha=alpha,  label=_c)
        #ax.scatter(xsii, y, c=colorcol[_n], s=1, alpha=alpha,  label=_c)

def pseudCompare(adataV, PseudoTs, colorby = 'CellType', fscale=8):
    fig, axes = plt.subplots(len(PseudoTs), len(PseudoTs), 
                             figsize=(fscale*(len(PseudoTs)), fscale*(len(PseudoTs))), 
                             constrained_layout=False, 
                             sharex=False,
                             sharey=False)
    for _i, _ict in enumerate(PseudoTs):
        for _j, _jct in enumerate(PseudoTs):
            axes[_i, _j].grid(False)
            if _i ==len(PseudoTs)-1:
                axes[_i, _j].set_xlabel(_jct)
            if _j ==0:
                axes[_i, _j].set_ylabel(_ict)
            if _i != _j:
                lowessfit(adataV, _ict, _jct, axes[_i, _j], colorby=colorby)
            else:
                sc.pl.umap(adataV, color= _jct, ax=axes[_i, _j], show=False, cmap='gnuplot')
                #sns.boxplot(x="monocle3_pseudotime", y="CellType", data=trajdata.obs, ax=ax[0])
    plt.tight_layout()
    plt.savefig('Pseudotimes.compare.lowess.pdf')
    plt.savefig('Pseudotimes.compare.lowess.png')

def pagaplot0(adataV, pseudotimes, groups='CellType', fscale=8,error=1,
             basis='umap',
             vkey='dynamical_velocity', minimum_spanning_tree=False):
    adataV = adataV.copy()
    # this is needed due to a current bug - bugfix is coming soon.
    #connectivities, distances = adataV.obsp['connectivities'].copy(), adataV.obsp['distances'].copy()
    #sc.pp.neighbors(adataV, n_neighbors=30, n_pcs=60, use_rep='X_scanorama')

    adataV.uns['neighbors']['distances'] = adataV.obsp['distances']
    adataV.uns['neighbors']['connectivities'] = adataV.obsp['connectivities']

    nlen = len(pseudotimes)
    fig, axes = plt.subplots(1,nlen, figsize=((fscale+error)*nlen,fscale))
    for _i, _p in enumerate(pseudotimes):
        iax = axes[_i] if nlen>1 else axes
        scv.tl.paga(adataV, 
                    groups=groups, 
                    vkey=vkey, 
                    #root_key=root_cells, 
                    #end_key=end_cells,
                    minimum_spanning_tree=minimum_spanning_tree,
                    threshold_root_end_prior=0.5,
                    use_time_prior= _p)
        #sc.tl.paga(adataV, groups='CellType', use_rna_velocity='dynamical_velocity_graph', )
        df = adataV.uns['paga']['transitions_confidence'].toarray().T
        sc.pl.umap(adataV, color=groups, ax=iax, show=False,)
        scv.pl.paga(adataV, basis=basis,
                    color=groups,
                    vkey=vkey, size=1, alpha=0.8,
                    threshold=0, 
                    edge_width_scale=0.4,
                    title=f'{groups}_{_p}',
                    show=False, 
                    ax= iax,
                    min_edge_width=2, 
                    node_size_scale=1)
    plt.tight_layout()
    if minimum_spanning_tree:
        plt.savefig('Pseudotimes.compare.paga.mst.v1.pdf')
    else:
        plt.savefig('Pseudotimes.compare.paga.v1.pdf')
    return(adataV)

def pagaplot(adataV, pseudotimes, groups='CellType', fscale=8, error=1,
             log_scale=True, threshold_root_end_prior=0.9,
             edge_width_scale=0.1, threshold=None,
             legend_loc = 'right margin',
             directed_scale=100, dashed_scale=100,
             save=None, show=True, ncols=4, nrows=None, soft=True,
             vkey='dynamical_velocity', minimum_spanning_tree=False):
    adataV = adataV.copy()
    # this is needed due to a current bug - bugfix is coming soon.
    #connectivities, distances = adataV.obsp['connectivities'].copy(), adataV.obsp['distances'].copy()
    #sc.pp.neighbors(adataV, n_neighbors=30, n_pcs=60, use_rep='X_scanorama')

    adataV.uns['neighbors']['distances'] = adataV.obsp['distances']
    adataV.uns['neighbors']['connectivities'] = adataV.obsp['connectivities']

    ncell = len(pseudotimes)
    nrowS, ncolS = PlotUti().colrows(ncell, ncols=ncols, nrows=nrows, soft=soft)
    fig, axes = plt.subplots(nrowS, ncolS, figsize=((fscale+error)*ncolS,fscale*nrowS))
    for _i, _p in enumerate(pseudotimes):
        if ncell==1:
            iax = axes
        elif min(nrowS, ncolS) ==1:
            iax = axes[n]
        else:
            iax = axes[_i//ncolS,_i%ncolS]
        scv.tl.paga(adataV, 
                    groups=groups, 
                    vkey=vkey, 
                    #root_key=root_cells, 
                    #end_key=end_cells,
                    minimum_spanning_tree=minimum_spanning_tree,
                    threshold_root_end_prior=threshold_root_end_prior,
                    use_time_prior= _p)
        #sc.tl.paga(adataV, groups='CellType', use_rna_velocity='dynamical_velocity_graph', )
        print('transitions_confidence max:',adataV.uns['paga']['transitions_confidence'].max())
        print('connectivities max:', adataV.uns['paga']['connectivities'].max())
        print('connectivities_tree max:', adataV.uns['paga']['connectivities_tree'].max())
        
        if log_scale:
            import scipy as sp
            #draw directed edges
            adataV.uns['paga']['transitions_confidence'] = \
                sp.sparse.csr_matrix(np.log2(adataV.uns['paga']['transitions_confidence'].toarray()*directed_scale +1))

            # draw dashed edges
            adataV.uns['paga']['connectivities'] = \
                sp.sparse.csr_matrix(np.log2(adataV.uns['paga']['connectivities'].toarray()*dashed_scale +1))

            # draw solid edges
            adataV.uns['paga']['connectivities_tree'] = \
                sp.sparse.csr_matrix(np.log2(adataV.uns['paga']['connectivities_tree'].toarray()*100 +1))

        infer_thre =adataV.uns['paga']['transitions_confidence'][
                          adataV.uns['paga']['transitions_confidence']>0].min()
        if threshold is None or type(threshold) in [int, float]:
            threshold=threshold
        elif type(threshold)== str and threshold=='infer':
            threshold=infer_thre
        print(threshold, infer_thre)
        sc.pl.umap(adataV, color=groups, ax=iax, show=False, legend_loc=legend_loc)
        scv.pl.paga(adataV, basis='umap',
                    color=groups,
                    vkey=vkey, size=1,
                    alpha=0.8,
                    threshold=threshold, 
                    edge_width_scale=edge_width_scale,
                    title=f'{groups}_{_p}',
                    show=False, 
                    ax= iax,
                    min_edge_width= adataV.uns['paga']['transitions_confidence'].min(), 
                    max_edge_width= adataV.uns['paga']['transitions_confidence'].max(),
                    node_size_scale=1)

    plt.tight_layout()
    if save is None :
        out_file = (f'Pseudotimes.compare.paga.mst.{groups}.pdf' 
                        if minimum_spanning_tree else 
                    f'Pseudotimes.compare.paga.{groups}.pdf')
    elif save:
        out_file = save
    else:
        out_file = None
    if not out_file is None:
        plt.savefig(out_file)
    if show:
        plt.show()
    return(adataV)