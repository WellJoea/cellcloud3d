import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
plt.rcdefaults()
    
class DotPlot():
    def __init__(self, size_df, color_df=None, dotcolor='viridis_r', rowcolor=None, colcolor=None, tree_kargs=None,
                  dot_kargs=None, **kargs):
        self.size_df = size_df.copy()
        self.color_df = size_df.copy() if color_df is None else color_df.copy()
        assert self.size_df.shape == self.color_df.shape
        #self.dotcolor =  'viridis_r' if dotcolor is None else dotcolor
        self.dotcolor = dotcolor
        self.rowcolor =  'tab20' if rowcolor is None else rowcolor
        self.colcolor =  'tab20b' if colcolor is None else colcolor
        self.tree_kargs =  {} if tree_kargs is None else tree_kargs
        self.dot_kargs = {} if dot_kargs is None else dot_kargs
        self.kargs = kargs

    def check_cmap(self, imaps):
        if isinstance(imaps, str):
            icmap = plt.get_cmap(imaps)
        elif isinstance(imaps, list):
            icmap = matplotlib.colors.ListedColormap(imaps)
        else:
            icmap = imaps
        return(icmap)

    def trans_cmap_tolist(self, imaps):
        icmap = self.check_cmap(imaps)
        icmap = list(map(mpl.colors.rgb2hex, icmap.colors))
        return(icmap)

    def dendrogram_plot(self, df_data,
                        dist_mtx = None,
                        labels = None,
                        cor_method='pearson', 
                        method='complete', 
                        metric='euclidean',
                        fastclust=True,
                        color_threshold=None, 
                        leaf_rotation=None,
                        link_colors = list(map(mpl.colors.rgb2hex, plt.get_cmap('tab20').colors)),
                        optimal_ordering=False,
                        **kargs):

        if labels is None:
            try:
                labels = df_data.index.tolist()
            except:
                labels = None
        if not dist_mtx is None:
            corr_condensed = dist_mtx
        elif cor_method in ['pearson', 'kendall', 'spearman']:
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
                                        metric=metric)
        else:
            z_var = sch.linkage(corr_condensed,
                            method=method, 
                            metric=metric,
                            optimal_ordering=optimal_ordering)

        sch.set_link_color_palette(link_colors)
        dendro_info = sch.dendrogram(z_var,
                                    labels=labels,
                                    leaf_rotation=leaf_rotation,
                                    color_threshold=color_threshold, 
                                    **kargs)
        return(dendro_info)

    def dot_plot(self, size_df, 
                color_df = None,
                ax=None,
                grid=True,
                color_on = 'dot', 
                show_yticks=False, 
                yticklabels_fs =None,
                xticklabels_fs =None,
                yticks_pos='right',
                show_xticks=True,
                size_scale=None, 
                color_scale=None,
                cmap=None,
                cax = None,
                scax = None,
                facecolor='none',
                grid_color = 'lightgrey', 
                grid_linestyle = '--', 
                grid_linewidth = 0.25,
                size_title='scale_size',
                color_title='color_scale',
                size_min = None, size_max= None, col_max = None, col_min = None,
                col_ticks= 5,
                max_size=150, 
                rigth_dist = None,
                pad=0.05, pwidth = 3, cwidth=0.03, cheight=0.1,
                swidth=0.06, sheight=0.4):
        cmap = self.dotcolor if cmap is None else cmap
        ax = plt if ax is None else ax
        size_df = size_df.copy()
        col_df = self.color_df.copy() if color_df is None else color_df.copy()

        smin = np.floor(size_df.min().min()) if size_min is None else size_min
        smax = np.ceil(size_df.max().max())  if size_max is None else size_max
        cmin = np.floor(col_df.min().min()) if col_min is None else col_min
        cmax = np.ceil(col_df.max().max()) if col_max is None else col_max
        
        if size_scale=='max':
            size_df = size_df/size_df.max().max()
        elif size_scale=='log1p':
            size_df = np.log1p(size_df)
        elif size_scale=='row':
            size_df = size_df.divide(size_df.max(1), axis=0)

        slmin = size_df.min().min() if size_min is None else size_min
        slmax = size_df.max().max()  if size_max is None else size_max
        size_df[size_df<slmin] = slmin
        size_df[size_df>slmax] = slmax
        
        if color_scale=='row':
            col_df = col_df.divide(col_df.max(1), axis=0)
        
        xorder=range(size_df.shape[1])
        yorder=range(size_df.shape[0])
    
        #size_df = size_df.iloc[yorder[::-1], xorder].copy()
        #col_df= col_df.iloc[yorder[::-1], xorder].copy()

        col_df= col_df/col_df.max().max()
        col_df= plt.get_cmap(cmap)(col_df)
        col_df = np.apply_along_axis(mpl.colors.rgb2hex, -1, col_df).flatten()
        
        #xx,yy=np.meshgrid(xorder, yorder)
        yy, xx = np.indices(size_df.shape)
        yy = yy.flatten() + 0.5
        xx = xx.flatten() + 0.5
        #scat = ax.pcolor(size_df, cmap=None, shading='auto')
        msize_df = (size_df.copy()/size_df.max().max())*max_size
        scat = ax.scatter(xx, yy, s=msize_df, c=col_df, cmap=cmap, edgecolor=None)

        y_ticks = np.arange(size_df.shape[0]) + 0.5
        yticklabels_fs = plt.rcParams['axes.titlesize'] if yticklabels_fs is None else yticklabels_fs
        xticklabels_fs = plt.rcParams['axes.titlesize'] if xticklabels_fs is None else xticklabels_fs
        ax.set_yticks(y_ticks)
        ax.set_yticklabels( [size_df.index[idx] for idx, _ in enumerate(y_ticks)], 
                           minor=False,
                           fontdict={'fontsize': yticklabels_fs}
        )

        x_ticks = np.arange(size_df.shape[1]) + 0.5
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [size_df.columns[idx] for idx, _ in enumerate(x_ticks)],
            rotation=90,
            ha='center',
            minor=False,
            fontdict={'fontsize': xticklabels_fs}
        )
        #ax.tick_params(axis='both', labelsize='small')
        #ax.set_ylabel(y_label)
        
        if yticks_pos=='right':
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        # Invert the y axis to show the plot in matrix form
        # ax.invert_yaxis()
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
            ax.grid(color = grid_color, linestyle = grid_linestyle,  linewidth = grid_linewidth)
        else:
            #ax.margins(x=2/size_df.shape[1], y=2/size_df.shape[0])
            ax.grid(False)
        if not show_yticks:
            #ax.set_yticks([])
            ax.yaxis.set_ticklabels([])
        if not show_xticks:
            #ax.set_xticks([])
            ax.xaxis.set_ticklabels([])
        
        ax.set_facecolor(facecolor)
        #def legend_plot(): 
        if rigth_dist is None:
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
                            cax=cax, ticks=np.linspace(0,1,col_ticks))
        cbar.set_ticklabels(np.linspace(cmin, cmax, col_ticks))
        cbar.set_label(color_title)

        pos = ax.get_position()
        bbpos = [rigth_dist, pos.y0 + 0.5, swidth, sheight] if scax is None else scax
        kw = dict(color = scat.cmap(0.7)) if color_df is None else dict(alpha=0.8)
        handles, labels = scat.legend_elements(prop="sizes",num=5, fmt="{x:.1f}", **kw)
        #labels = ['20', '40', '60', '80' ,'100']
        labels = [ "%.1f"%(i) for i in np.linspace(slmin,slmax, 6)[1:] ]
        legend2 = ax.legend(handles, labels, loc='center left',
                            #alignment='right',
                            title=f'{size_title}',
                            bbox_to_anchor=bbpos,)
        
    def annot_plot(self, color_df, ax, cmap=None, swap_axes=False, set_frame_on=True, **kargs):
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
        ax.set_frame_on(set_frame_on)
        
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

    def clustdot(self,
                figsize = None,
                annot_ratio=(0.05, 0.05), 
                row_cluster=True,
                row_annot=True,
                add_row_annot= None,
                col_cluster=True,
                col_annot=True,
                add_col_annot= None,
                dendrogram_ratio=(.1, .1),
                width_ratios=None,
                height_ratios=None,
                wspace=0,
                hspace=0,
                show_yticks=True,
                show_xticks=True,
                max_size=150,
                row_threshold=None, 
                col_threshold=None,
                annot_linewidths=0.2, 
                annot_edgecolor='white',
                gskws = None,
                dotkws= None,
                rowdendkws = None,
                coldendkws = None,
                annotkws = None,
                save=None,
                show=True):
        from matplotlib import gridspec
        size_df = self.size_df.copy()
        color_df = self.color_df.copy()
        self.row_order = size_df.index
        self.col_order = size_df.columns

        figsize = (10,10) if figsize is None else figsize
        is_add_row_annot = (not add_row_annot is None)
        is_add_col_annot = (not add_col_annot is None)

        ncols = sum([row_cluster, (row_annot | is_add_row_annot)]) +1
        nrows = sum([col_cluster, (col_annot | is_add_col_annot)]) +1

        if height_ratios is None:
            height_ratios = []
            if col_cluster:
                height_ratios.append(dendrogram_ratio[1])
            if col_annot or is_add_col_annot:
                height_ratios.append(annot_ratio[1])
            dot_hg = sum(height_ratios[:-1])
            dot_hg = (1- dot_hg) if dot_hg<1 else np.abs(figsize-dot_hg)
            height_ratios.append(dot_hg)

        if width_ratios is None:
            width_ratios = []
            if row_cluster:
                width_ratios.append(dendrogram_ratio[0])
            if row_annot or is_add_row_annot:
                width_ratios.append(annot_ratio[0])
            dot_wd = sum(width_ratios[:-1])
            dot_wd = (1- dot_wd) if dot_wd<1 else np.abs(figsize-dot_wd)
            width_ratios.append(dot_wd)

        gskws = {} if gskws is None else gskws
        gskws.update(dict( wspace=wspace,
                            hspace=hspace,
                            width_ratios=width_ratios,
                            height_ratios=height_ratios,
        ))

        dotkws = {} if dotkws is None else dotkws
        dotkws.update(dict(show_yticks=show_yticks,
                           show_xticks=show_xticks,
                           max_size=max_size,
                           cmap = self.dotcolor,
        ))

        rowdendkws = {} if rowdendkws is None else rowdendkws
        rowdendkws.update(dict(color_threshold=row_threshold, 
                                no_plot=True,
                                link_colors=self.trans_cmap_tolist(self.rowcolor)))

        coldendkws = {} if coldendkws is None else coldendkws
        coldendkws.update(dict(color_threshold=col_threshold,
                                no_plot=True, 
                                link_colors=self.trans_cmap_tolist(self.colcolor)))

        annotkws = {} if annotkws is None else annotkws
        annotkws.update(dict(linewidths=annot_linewidths,
                              edgecolor=annot_edgecolor))


        ##plot figure
        self.fig = plt.figure(figsize=figsize)
        self.gs = gridspec.GridSpec(nrows, ncols, **gskws)
        self.ax_dotmap = self.fig.add_subplot(self.gs[-1, -1])
        for axis in ['top', 'bottom', 'left', 'right']:
            self.ax_dotmap.spines[axis].set_linewidth(1.5)

        if row_cluster:
            self.ax_row_dendrogram = self.fig.add_subplot(self.gs[-1, 0])
            self.ax_row_dendrogram.set_axis_off()
            rowdendkws.update(dict(leaf_rotation=0,
                                    no_plot=False,
                                    ax=self.ax_row_dendrogram, 
                                    orientation='left'))
        if (row_cluster or row_annot):
            self.row_dendrog = self.dendrogram_plot(size_df, **rowdendkws)
            self.row_order = self.row_dendrog['ivl']
            if row_cluster:
                self.ax_row_dendrogram.invert_yaxis() 

        if (row_annot or is_add_row_annot):
            self.ax_row_colors = self.fig.add_subplot(self.gs[-1, (1 if row_cluster else 0)])
            self.row_cor_df = pd.DataFrame(index=self.row_order)
            if row_annot:
                self.row_cor_df['cluster'] =  self.row_dendrog['leaves_color_list']
            if is_add_row_annot:
                self.row_cor_df = pd.merge(self.row_cor_df, add_row_annot, left_index=True, how='left',right_index=True)
            self.annot_plot(self.row_cor_df, self.ax_row_colors, **annotkws)
            self.ax_row_colors.invert_yaxis()

        if col_cluster:
            self.ax_col_dendrogram = self.fig.add_subplot(self.gs[0, -1])
            self.ax_col_dendrogram.set_axis_off()
            coldendkws.update(dict(leaf_rotation=90,
                                    ax=self.ax_col_dendrogram, 
                                    no_plot=False,
                                    orientation='top'))
        if (col_cluster or col_annot):
            self.col_dendrog = self.dendrogram_plot(size_df.T, **coldendkws)
            self.col_order = self.col_dendrog['ivl']
        if (col_annot or is_add_col_annot):
            self.ax_col_colors = self.fig.add_subplot(self.gs[(1 if col_cluster else 0),-1])
            self.col_cor_df = pd.DataFrame(index=self.col_order)
            if col_annot:
                self.col_cor_df['cluster'] =  self.col_dendrog['leaves_color_list']
            if is_add_col_annot:
                self.col_cor_df = pd.merge(self.col_cor_df, add_col_annot, left_index=True, how='left',right_index=True)
            self.annot_plot(self.col_cor_df, self.ax_col_colors, swap_axes=True, **annotkws)
        #ax_cbar = self.fig.add_subplot(gs[0, 0])
        #ax_cbar.set_axis_off()

        self.size_df = self.size_df.loc[self.row_order, self.col_order]
        self.color_df= self.color_df.loc[self.row_order, self.col_order]
        self.dot_plot(self.size_df, color_df=self.color_df, ax=self.ax_dotmap, **dotkws)

        plt.tight_layout()
        if save:
            self.fig.savefig(save, bbox_inches='tight')
        if show:
            self.fig.show()
        else:
            return selfs

def test():
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

    AA = DotPlot(sig_mean, color_df=sig_padj)
    AA.clustdot(row_threshold=1,
                           col_threshold=0.5,
                           add_col_annot=add_col_annot,
                           figsize=(20,45),
                           save='significant.mean.dotplot.size.mean.color.pvalue.cluster.pdf',
                           dotkws={"size_title":'scale_mean_values',
                                    "color_title":'-log10 p values',},
                           width_ratios=[3.5,0.5,16],
                           height_ratios=[3.5,1.5,40])

