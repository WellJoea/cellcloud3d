import scanpy as sc
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype']  = 42
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from cellcloud3d.plotting._colors import adata_color

# import seaborn as sns
# sns.set(font_scale=1)
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 22})


def violin(adata, groupby='sampleid', ncols=5,  s=1,
            scalar =6, ewidth=None, ehight=None,
            show=True, save=None, logy=True, ylim=None,
            trans=False,linewidth=None,
            marker_color = None, fontsize=15,
            hlines = {}, hcolor='red',
            COLs = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo','doubletS'], **kargs):

    adata = adata.copy()
    try:
        G = adata.obs[groupby].cat.remove_unused_categories().cat.categories
    except:
        G = adata.obs[groupby].unique()

    adata_color(adata, value = groupby)
    violin_col = list(adata.uns[f'{groupby}_colors'])

    COLs = [ i for i in COLs if i in adata.obs.columns ]
    if len(COLs)==0:
        print('no column is found in obs.columns.')
        print('run sc.pp.calculate_qc_metrics.')
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        COLs = [ i for i in COLs if i in adata.obs.columns ]
        assert len(COLs)==0, 'no column is found in obs.columns.'

    if (marker_color is None) or (type(marker_color) in [str]):
        marker_color = [marker_color] * len(COLs)

    if len(COLs) < ncols: ncols=len(COLs)
    nrows = int(np.ceil(len(COLs)/ncols))

    ewidth = ewidth or scalar/12*len(G)
    ehight = ehight or scalar
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*ewidth, nrows*ehight))

    for n,i in enumerate(COLs):
        if len(COLs)==1:
            AX = axes
        elif (len(COLs) > 1) and (nrows==1):
            AX = axes[n]
        elif (len(COLs) > 1) and (ncols==1):
            AX = axes[n]
        else:
            AX = axes[n//ncols,n%ncols]
        if trans:
            ax = sns.violinplot(y=groupby, x=i, hue=groupby, data=adata.obs, density_norm='width', cut=0, width=0.8, ax=AX, 
                                linewidth=linewidth, palette=violin_col, **kargs)
            ax = sns.stripplot( y=groupby, x=i, hue=groupby, data=adata.obs, size=1, edgecolor=None, 
                                color=marker_color[n],
                                linewidth=0, jitter=0.2, zorder=1, alpha=0.8, ax=AX, **kargs)
            ax.set_title(i)
            if logy: ax.set_xscale("log")
            if not ylim is None:ax.set(xlim=ylim)
        else:
            ax = sns.violinplot(x=groupby, y=i, hue=groupby, data=adata.obs, density_norm='width', cut=0, width=0.8, ax=AX, 
                                linewidth=linewidth, palette=violin_col, **kargs)
            ax = sns.stripplot( x=groupby, y=i, hue=groupby, data=adata.obs, size=s, edgecolor=None, 
                                color=marker_color[n],
                                linewidth=0, jitter=0.2, zorder=1, alpha=0.8, ax=AX, **kargs)
            ax.set_title(i, fontdict={'fontsize':fontsize})
            if logy: ax.set_yscale("log")
            if not ylim is None:ax.set(ylim=ylim)
            ax.set_xticklabels(
                ax.get_xticklabels(), 
                rotation=90, 
                ha='center',
                va='center_baseline',
                fontsize=fontsize,
            )
        ax.tick_params(axis='both', labelsize=fontsize)
        if i in hlines:
            for hl in hlines[i]:
                ax.axhline(y=hl, color=hcolor, linestyle="--")

    if nrows*ncols - len(COLs) >0:
        for j in range(nrows*ncols - len(COLs)):
            fig.delaxes(axes[-1][ -j-1])

    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        fig.show()
    else:
        plt.close()

def QCplot(adata, groupby='sampleid', header='before', ncols=3, lcol=2, pform='png',
            hlines = {}, hcolor='grey',
        clims= {'n_genes_by_counts':4000, 'total_counts':5000, 'pct_counts_mt':15, 'pct_counts_ribo':30, 'pct_counts_hb':30},
        COLs = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb', 'doubletS', 'CC_diff'], 
        show=True, save=None, **kargs):
    adata = adata.copy()
    try:
        G = adata.obs[groupby].cat.remove_unused_categories().cat.categories
    except:
        G = adata.obs[groupby].unique()
    adata.obs[groupby] = pd.Categorical( adata.obs[groupby], categories=G )

    adata_color(adata, value = groupby)
    violin_col = list(adata.uns[f'{groupby}_colors'])

    obsdata = adata.obs.sort_values('pct_counts_mt')
    COLs = [i for i in COLs if i in adata.obs.columns]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8),constrained_layout=False)
    f1 = sns.histplot(data=obsdata, x="n_genes_by_counts", hue=groupby, palette=violin_col,
                        multiple="stack", linewidth=0, ax=axes[0,0], legend=False)
    f2 = sns.histplot(data=obsdata, x="total_counts", hue=groupby, multiple="stack",  palette=violin_col,
                      linewidth=0,  ax=axes[0,1], legend=False)
    f3 = sns.scatterplot(data=obsdata, x="total_counts", y="n_genes_by_counts", hue=groupby,
                        ax=axes[0, 2],  s=1,linewidth=0, legend='full')
    f4 = sns.scatterplot(data=obsdata, x="total_counts", y="pct_counts_mt", hue=groupby,
                        ax=axes[0, 3],  s=1,linewidth=0, legend=False)
    f5 = sns.scatterplot(data=obsdata, x="total_counts", y="pct_counts_ribo", hue=groupby,
                        ax=axes[1, 0], s=1, linewidth=0, legend=False)
    f6 = sns.scatterplot(data=obsdata, x="pct_counts_mt", y="pct_counts_ribo", hue=groupby,
                        ax=axes[1, 1], s=1, linewidth=0,legend=False)

    f7 = axes[1, 2].scatter(obsdata['total_counts'], 
                                obsdata['n_genes_by_counts'],
                                cmap='cool',
                                c=obsdata['pct_counts_mt'],
                                vmin=0, vmax=clims['pct_counts_mt'],
                                s=5,linewidth=0 ) 

    f8 = axes[1, 3].scatter(adata.obs.sort_values('pct_counts_ribo')['total_counts'], 
                                adata.obs.sort_values('pct_counts_ribo')['n_genes_by_counts'],
                                cmap='RdPu',
                                c=adata.obs.sort_values('pct_counts_ribo')['pct_counts_ribo'],
                                vmin=0, vmax=clims['pct_counts_mt'],
                                s=5, linewidth=0)

    axes[1, 2].set_xlabel('total_counts')
    axes[1, 2].set_ylabel('n_genes_by_counts')
    cax7 = fig.add_axes([1.0, 0.33, 0.015, 0.21])
    fig.colorbar(f7, ax=axes[1, 3], label='pct_counts_mt', cax=cax7)
    axes[1, 3].set_xlabel('total_counts')
    axes[1, 3].set_ylabel('n_genes_by_counts')
    cax8 = fig.add_axes([1, 0.1, 0.015, 0.21])
    fig.colorbar(f8, ax=axes[1, 3], label='pct_counts_ribo', cax=cax8)

    for r in range(2):
        for c in range(4):
            axes[r, c].ticklabel_format( useOffset=False, style='sci', axis='both')
    axes[0,0].set_xlim([0, clims['n_genes_by_counts']])
    axes[0,1].set_xlim([0, clims['total_counts']])
    axes[0,2].set_xlim([0, clims['total_counts']])
    axes[0,2].set_ylim([0, clims['n_genes_by_counts']])
    axes[0,3].set_xlim([0, clims['total_counts']])
    axes[0,3].set_ylim([0, clims['pct_counts_mt']])
    axes[1,0].set_xlim([0, clims['total_counts']])
    axes[1,0].set_ylim([0, clims['pct_counts_ribo']])
    axes[1,1].set_xlim([0, clims['pct_counts_mt']])
    axes[1,1].set_ylim([0, clims['pct_counts_ribo']])
    axes[1,2].set_xlim([0, clims['total_counts']])
    axes[1,2].set_ylim([0, clims['n_genes_by_counts']])
    axes[1,3].set_xlim([0, clims['total_counts']])
    axes[1,3].set_ylim([0, clims['n_genes_by_counts']])

    lines, labels= f3.get_legend_handles_labels()
    axes[0,2].get_legend().remove()
    fig.legend(lines,
                labels=labels[::-1],
                bbox_to_anchor=(1.0, 0.56),
                loc="lower left",
                borderaxespad=0.,  
                title=groupby,
                ncol = lcol)

    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        fig.show()
    else:
        plt.close()

def genePlot(adata, use_res=None, groupby = 'SID', header='before', show=True, save=None, 
                title='the statistics of gene counts and cells', ylim_cell=None,
                yim_counts=None, logy=True, **kargs):
    from scipy.sparse import issparse
    gene_counts = []
    use_res = 'X' if use_res is None else use_res
    if use_res =='X':
        adataX = adata.X
    elif use_res=='raw':
        adataX = adata.raw.X
    elif use_res in adata.layers.keys():
        adataX =  adata.layers[use_res]
    elif use_res =='shared':
        Xs, Xu = adata.layers["spliced"], adata.layers["unspliced"]
        nonzeros = ((Xs > 0).multiply(Xu > 0) if issparse(Xs) else (Xs > 0) * (Xu > 0))
        adataX= ( nonzeros.multiply(Xs) + nonzeros.multiply(Xu)
                    if issparse(nonzeros)
                    else nonzeros * (Xs + Xu))
    else:
        raise ValueError('`use_res` needs to be one of in "X, None, raw, shared, spliced, unspliced "')
    for k in adata.obs[groupby].cat.categories:
        iadata = adataX[np.flatnonzero(adata.obs[groupby]==k),]
        icounts= np.vstack([(iadata>0).sum(0), iadata.sum(0),
                    [k] * iadata.shape[1], adata.var.index ])
        gene_counts.append(icounts.T)
    gene_counts = pd.DataFrame(np.vstack(gene_counts), 
                                columns=['n_cells_by_gene', 'n_counts_by_gene', groupby, 'gene'])
    gene_counts = gene_counts.infer_objects()
    gene_counts_sum=gene_counts.drop(groupby, axis=1).groupby('gene').sum(1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10),constrained_layout=False)
    fig.suptitle(f'{title} ({header} {use_res})', fontsize=12)
    ax0 = sns.histplot(data=gene_counts_sum, x="n_cells_by_gene", binwidth=5, binrange=[1,500],
                            multiple="stack", linewidth=0.1, ax=axes[0,0], legend=True,  **kargs)
    ax0.set_title('n_cells_by_gene_all')
    ax1 = sns.histplot(data=gene_counts, x="n_cells_by_gene", hue=groupby, binwidth=5, binrange=[1,500],
                            multiple="stack", linewidth=0.1, ax=axes[0,1], legend=True,  **kargs)
    ax1.set_title('n_cells_by_gene_each')
    ax2 = sns.violinplot(x=groupby, y='n_cells_by_gene', data=gene_counts, scale='width', cut=0, 
                            width=0.8, ax=axes[0,2])
    ax2 = sns.stripplot( x=groupby, y='n_cells_by_gene', data=gene_counts, size=1, edgecolor=None, 
                        linewidth=0, jitter=0.2, zorder=1, alpha=0.8, ax=axes[0,2])
    ax2.set_title('n_cells_by_gene')
    if logy: ax2.set_yscale("log")
    if not ylim_cell is None:ax2.set(ylim=ylim_cell)
    ax2.set_xticklabels(
        ax2.get_xticklabels(), 
        rotation=90, 
        ha='center',
        va='center_baseline',
        fontsize=10,
    )
    ax3 = sns.histplot(data=gene_counts_sum, x="n_counts_by_gene", bins=100, binrange=[1,1000],
                            multiple="stack", linewidth=0.1, ax=axes[1,0], legend=True,  **kargs)
    ax3.set_title('n_counts_by_gene_all')
    ax4 = sns.histplot(data=gene_counts, x="n_counts_by_gene", hue=groupby, bins=100, binrange=[1,1000],
                            multiple="stack", linewidth=0.1, ax=axes[1,1], legend=True,  **kargs)
    ax4.set_title('n_counts_by_gene_each')
    ax5 = sns.violinplot(x=groupby, y='n_counts_by_gene', data=gene_counts, scale='width',
                            cut=0, width=0.8, ax=axes[1,2])
    ax5 = sns.stripplot( x=groupby, y='n_counts_by_gene', data=gene_counts, size=1, edgecolor=None, 
                        linewidth=0, jitter=0.2, zorder=1, alpha=0.8, ax=axes[1,2])
    ax5.set_title('n_counts_by_gene')
    if logy: ax5.set_yscale("log")
    if not yim_counts is None:ax5.set(ylim=yim_counts)
    ax5.set_xticklabels(
        ax5.get_xticklabels(), 
        rotation=90, 
        ha='center',
        va='center_baseline',
        fontsize=10,
    )
    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        fig.show()
    else:
        plt.close()