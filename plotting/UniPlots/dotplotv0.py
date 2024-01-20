import matplotlib.pyplot as plt
import matplotlib as mpl

def dendrogram_plot(df_data,  cor_method='pearson', 
                   method='complete', fastclust=False,
                   color_threshold=None, leaf_rotation=90,
                   link_colors = list(map(mpl.colors.rgb2hex, plt.get_cmap('tab20').colors)),
                   optimal_ordering=False,**kargs):
    from scipy.spatial import distance
    import scipy.cluster.hierarchy as sch
    categories = df_data.index.tolist()

    if cor_method in ['pearson', 'kendall', 'spearman']:
        corr_matrix = df_data.T.corr(method=cor_method)
        corr_condensed = distance.squareform(1 - corr_matrix)
    elif cor_method in ['sknormal']:
        from sklearn.preprocessing import normalize     
        corr_condensed = normalize(df_data.copy()) 
    else:
        corr_condensed = df_data.copy()
    if fastclust:
        import fastcluster
        z_var = fastcluster.linkage(corr_condensed, method=method,
                                      metric='euclidean')
    else:
        z_var = sch.linkage(corr_condensed,
                        method=method, 
                        metric='euclidean',
                        optimal_ordering=optimal_ordering)

    sch.set_link_color_palette(link_colors)
    dendro_info = sch.dendrogram(z_var,
                                 labels=categories,
                                 leaf_rotation=leaf_rotation,
                                 color_threshold=color_threshold, 
                                 **kargs)
    return(dendro_info)

def dot_plot(size_df, color_df=None, ax=None,
             max_size=150, nomal_color=True, grid=True,
             color_on = 'dot', show_yticks=False, show_xticks=True,
             size_scale='max', color_scale=None, cmap='viridis_r',
             cax = None,
             pad=0.05, pwidth = 3, cwidth=0.03, cheight=0.3,
             swidth=0.06, sheight=0.4):
    import numpy as np
    import matplotlib.pyplot as plt
    ax = plt if ax is None else ax
    
    col_df = size_df.copy() if color_df is None else color_df.copy()
    col1_df = col_df.copy()
    size_df= size_df.copy()
    
    smin = np.floor(size_df.min().min())
    smax = np.ceil(size_df.max().max())
    cmin = np.floor(col_df.min().min())
    cmax = np.ceil(col_df.max().max())
    
    if size_scale=='max':
        size_df = size_df/size_df.max().max()
    elif size_scale=='log1p':
        size_df = np.log1p(size_df)
    elif size_scale=='row':
        size_df = size_df.divide(size_df.max(1), axis=0)
    if color_scale=='row':
        col_df = col_df.divide(col_df.max(1), axis=0)
    
    xorder=range(size_df.shape[1])
    yorder=range(size_df.shape[0])
    size_df = size_df.iloc[yorder[::-1], xorder].copy()
    col_df= col_df.iloc[yorder[::-1], xorder].copy()

    col_df= col_df/col_df.max().max()
    col_df= plt.get_cmap(cmap)(col_df)
    col_df = np.apply_along_axis(mpl.colors.rgb2hex, -1, col_df).flatten()
    
    #xx,yy=np.meshgrid(xorder, yorder)
    yy, xx = np.indices(size_df.shape)
    yy = yy.flatten() + 0.5
    xx = xx.flatten() + 0.5
    #scat = ax.pcolor(size_df, cmap=None, shading='auto')
    scat = ax.scatter(xx, yy, s=size_df*max_size, c=col_df, cmap=cmap, edgecolor=None)

    y_ticks = np.arange(size_df.shape[0]) + 0.5
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [size_df.index[idx] for idx, _ in enumerate(y_ticks)], minor=False
    )

    x_ticks = np.arange(size_df.shape[1]) + 0.5
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [size_df.columns[idx] for idx, _ in enumerate(x_ticks)],
        rotation=90,
        ha='center',
        minor=False,
    )
    ax.tick_params(axis='both', labelsize='small')
    #ax.set_ylabel(y_label)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    #ax.xaxis.set_label_position("top")
    #ax.xaxis.tick_top()
    
    ax.set_ylim(size_df.shape[0], 0)
    ax.set_xlim(0, size_df.shape[1])
    if color_on == 'dot':
        x_padding = 0.5
        y_padding = 0.5
        x_padding = x_padding - 0.5
        y_padding = y_padding - 0.5
        ax.set_ylim(size_df.shape[0] + y_padding, -y_padding)
        ax.set_xlim(-x_padding, size_df.shape[1] + x_padding)
            
    if grid:
        ax.set_axisbelow(True)
        ax.grid(color = 'lightgrey', linestyle = '--', linewidth = 0.4)
    else:
        #ax.margins(x=2/size_df.shape[1], y=2/size_df.shape[0])
        ax.grid(False)
    if not show_yticks:
        ax.set_yticks([])
    if not show_xticks:
        ax.set_xticks([])

    #def legend_plot(): 
    if not show_yticks:
        rigth_dist = 1.01
    else:
        rigth_dist = 1.21
    if cax == 'full':
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=f"{pwidth}%", pad=pad)
    elif (not ax in [plt]) and (cax in [None, 'part']):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax,
                   width=f"{cwidth*100}%",
                   height=f"{cheight*100}%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(rigth_dist, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0)
    elif cax is None:
        cax = [1.01, 0.5, 0.05, 0.3]
        cax = plt.axes(cax)
    else:
        cax = plt.axes(cax)
    
    #cbar = plt.colorbar(scat, cax=cax, ticks=np.linspace(0,1,5))
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap),
                        cax=cax, ticks=np.linspace(0,1,5))
    cbar.set_ticklabels(np.linspace(cmin, cmax,5))
    cbar.set_label('color_scale')

    pos = ax.get_position()
    bbpos = [rigth_dist, pos.y0 + 0.5, swidth, sheight]
    kw = dict(color = scat.cmap(0.7)) if color_df is None else dict(alpha=0.8)
    handles, labels = scat.legend_elements(prop="sizes",fmt="{x:.1f}", **kw)
    legend2 = ax.legend(handles, labels, loc='center right',
                        bbox_to_anchor=bbpos, title="Sizes")

def annotplot(color_df, ax, cmap=None, swap_axes=False, **kargs):
    mtx_df = []
    cmaps = []
    _init = 0
    color_df = color_df.T.iloc[::-1,:].copy() if swap_axes else color_df.copy()
    for icl in color_df.columns:
        i_col = color_df[icl].copy()
        uni_col = np.unique(i_col)
        uni_len = uni_col.shape[0]
        uni_num = dict(zip(uni_col, range(_init, uni_len+_init)))
        i_col = i_col.map(uni_num)
        mtx_df.append(i_col)
        cmaps.extend(uni_col)
        _init+=uni_len
    mtx_df = pd.concat(mtx_df, axis=1)
    #cmap=(cmaps if cmap is None else cmap),
    #mesh = ax.pcolormesh(mtx_df, cmap=color_df, **kargs)
    cmap = mpl.colors.ListedColormap(cmaps) if cmap is None else cmap
    mesh = ax.pcolormesh(mtx_df, cmap=cmap, **kargs)
    ax.set(xlim=(0, mtx_df.shape[1]), ylim=(0, mtx_df.shape[0]))
    x_ticks = np.arange(mtx_df.shape[1]) + 0.5
    y_ticks = np.arange(mtx_df.shape[0]) + 0.5
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)
    ax.set_frame_on(False)
    
    if swap_axes:
        ax.set_yticklabels(mtx_df.index, 
                           #rotation=90,
                            minor=False)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xticks([])
    else:
        ax.set_xticklabels(mtx_df.columns, 
                           rotation=90,
                            ha='center',
                            minor=False)
        ax.set_yticks([])

def clustdot(size_df, color_df=None,
             figsize=(20,30),
             colors_ratio=(0.05, 0.05), 
             row_threshold=None, 
             col_threshold=None,
             dendrogram_ratio=(.1, .1),
             add_row_annot= None,
             add_col_annot= None,
             width_ratios=None,
             height_ratios=None,
             save=None,
             show=True):
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    from scipy.spatial import distance
    import scipy.cluster.hierarchy as sch
    nrows = 3
    ncols = 3    
    
    if height_ratios is None:
        height_ratios= (dendrogram_ratio[1], colors_ratio[1], 1-dendrogram_ratio[1]-colors_ratio[1])
    if width_ratios is None:
        width_ratios = (dendrogram_ratio[0], colors_ratio[0], 1-dendrogram_ratio[0]-colors_ratio[0])

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, 
                            wspace=0.005,
                            hspace=0.005,
                            width_ratios=width_ratios,
                            height_ratios=height_ratios)
    ax_heatmap = fig.add_subplot(gs[-1, -1])
    ax_row_dendrogram = fig.add_subplot(gs[-1, 0])
    ax_col_dendrogram = fig.add_subplot(gs[0, -1])
    ax_row_dendrogram.set_axis_off()
    ax_col_dendrogram.set_axis_off()

    ax_row_colors = fig.add_subplot(gs[-1, 1])
    ax_col_colors = fig.add_subplot(gs[1, -1])
        
    #ax_cbar = fig.add_subplot(gs[0, 0])
    #ax_cbar.set_axis_off()
    #ax_col_colors.set_axis_off()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax_heatmap.spines[axis].set_linewidth(1.5)

    row_dendro = dendrogram_plot(size_df, cor_method='pearson', fastclust=True,
                                 leaf_rotation=0,
                                 color_threshold=row_threshold,
                                 ax=ax_row_dendrogram, orientation='left')

    row_cor_df = pd.DataFrame({'cluster': row_dendro['leaves_color_list']},
                               index=row_dendro['ivl'])
    if not add_row_annot is None:
        row_cor_df = pd.concat([row_cor_df, add_row_annot], axis=1)
    annotplot(row_cor_df, ax_row_colors, linewidths=0.2, edgecolor='white')
    
    
    col_dendro = dendrogram_plot(size_df.T, cor_method='pearson', fastclust=True, 
                                 color_threshold=col_threshold,
                                 ax=ax_col_dendrogram, orientation='top')
    col_cor_df = pd.DataFrame({'cluster' : col_dendro['leaves_color_list']},
                               index=col_dendro['ivl'])
    if not add_col_annot is None:
        col_cor_df = pd.concat([col_cor_df, add_col_annot], axis=1)
    annotplot(col_cor_df, ax_col_colors, linewidths=0.2, edgecolor='white', swap_axes=True)

    if not color_df is None:
        color_df = color_df.iloc[row_dendro['leaves'][::-1], col_dendro['leaves']]
    dot_plot(size_df.iloc[row_dendro['leaves'][::-1], col_dendro['leaves']],
             color_df=color_df,
             show_yticks=True,
             ax=ax_heatmap)
    
    plt.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight')
    if show:
        fig.show()
    else:
        return(fig, gs)

sig_mean, sig_p = get_meat_df(mean_df, pvalue_df=pvalue_df, max_p = 0.05)
sig_padj = sig_p.copy()
sig_padj[sig_padj==0] = 0.001
sig_padj = -np.log10(sig_padj)

COLOR = dict(zip(adataI.obs['CellTypeN'].cat.categories, adataI.uns['CellTypeN_colors']))
add_col_annot = pd.DataFrame(sig_mean.columns.str.split('|', expand=True)\
                                     .to_frame().reset_index(drop=True).values,
                             index=sig_mean.columns,
                             columns=['source','target'])
                                                       

add_col_annot = add_col_annot.replace(COLOR)

clustdot(sig_mean, color_df=sig_padj, 
                       row_threshold=1,
                       col_threshold=0.5,
                       add_col_annot=add_col_annot,
                       figsize=(20,45),
                       save='significant.mean.dotplot.size.mean.color.pvalue.cluster.pdf',
                       width_ratios=[3.5,0.5,16], height_ratios=[3.5,1.5,40])