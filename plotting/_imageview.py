import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import matplotlib
import matplotlib.pyplot as plt

from skimage import transform as skitf
# from itkwidgets import view
# from itkwidgets import view, compare, checkerboard
# from itkwidgets.widget_viewer import Viewer
# import ipywidgets
# from ipywidgets import Button, Label, HBox, VBox
import scanpy as sc

from cellcloud3d.plotting._utilis import colrows

def qview(*arrays, layout=None, 
                dtypes = None,
                fsize=5, 
                werror=0,
                titles=None,
                nrows=None,
                ncols=None, 
                show=True, 
                save=None,
                invert_xaxis=False,
                invert_yaxis=False,
                rescale = None,
                anti_aliasing=None,
                size = 1,
                color = 'black',
                aspect='auto',
                equal_aspect=False,
                grid=False,
                axis_off=False,
                cmaps=None, alpha=None,
                sharex=False, sharey=False,
                figkargs={},
                title_fontsize = None,
                **kargs
                ):
    ncells= len(arrays)
    nrows, ncols = colrows(ncells, nrows=nrows, ncols=ncols, soft=False)

    if (cmaps is None) or isinstance(cmaps, str):
        cmaps = [cmaps]*ncells

    if (dtypes is None):
        dtypes = []
        for ii in arrays:
            assert ii.ndim >=2
            if min(ii.shape) ==2 and (ii.ndim ==2):
                dtypes.append('loc')
            elif (ii.ndim >=2) and ii.shape[1] >2:
                dtypes.append('image')
    elif (isinstance(dtypes), str):
        dtypes = [dtypes] * ncells
    elif (isinstance(dtypes), list):
        assert len(dtypes) >= ncells
        dtypes = dtypes[:ncells]

    if not rescale is None:
        arrs = []
        for n in range(ncells):
            ida = arrays[n]
            itp = dtypes[n]
            if itp in ['image']:
                hw = ida.shape[:2]
                resize = [int(round(hw[0] * rescale ,0)), int(round( hw[1] *rescale ,0))]
                arrs.append(skitf.resize(ida, resize))
            elif itp in ['loc']:
                arrs.append(ida * rescale)
    else:
        arrs = arrays

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=((fsize+werror)*ncols,fsize*nrows),
                              sharex=sharex, sharey=sharey,**figkargs)
    #fig.patch.set_facecolor('xkcd:mint green')
    for i in range(ncells):
        if ncells ==1:
            ax = axes
        elif min(nrows, ncols)==1:
            ax = axes[i]
        else:
            ax = axes[i//ncols,i%ncols]
        ida = arrs[i]
        itp = dtypes[i]

        if itp in ['image']:
            if layout=='bgr':
                ax.imshow(ida[:,:,::-1], aspect=aspect, cmap=cmaps[i], alpha=alpha,**kargs)
            elif layout=='rbg':
                ax.imshow(ida[:,:,:3], aspect=aspect, cmap=cmaps[i], alpha=alpha, **kargs)
            else:
                ax.imshow(ida, aspect=aspect, cmap=cmaps[i], alpha=alpha, **kargs)
        elif itp in ['loc']:
            ax.scatter(ida[:,0], ida[:,1], s=size, c=color,**kargs)

        if (not titles is None):
            if i < len(titles):
                ax.set_title(titles[i], fontsize=title_fontsize)
        if axis_off:
            ax.set_axis_off()

        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')
        ax.grid(grid)
        
        # if not show_ticks:
        #     ax.tick_params(
        #     axis='x',         
        #     which='both',      
        #     bottom=False,      
        #     top=False,        
        #     labelbottom=False) 
        
        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols - ncells):
            fig.delaxes(axes.flatten()[-j-1])
    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        fig.show()
    else:
        plt.close()

def drawMatches(pairlist, bgs=None,
                pairidx=None,
                matches=None,
                line_color='r', size=5, line_width=1, color='b',
                line_alpha=None, hide_axis=False, grid=False, show_line=True,
                bg_color='black', bg_size=3,titles=None, aspect='auto',
                cmap=None, alpha=None, arrowstyle ='-',
                nrows = None, ncols=None, fsize = 7, werror =0,
                sharex=True, sharey=True,
                figkargs={},
                linekargs={},
                line_limit = None,
                line_sample = None,
                equal_aspect = False, 
                save=None, show=True, 
                invert_xaxis=False,
                invert_yaxis=False,
                seed = None,
                **kargs):
    
    np.random.seed(seed)
    ncells= len(pairlist)
    assert ncells >1, 'pairlist length muse be >1'
    if (not bgs is None) and (len(bgs)!=ncells ) :
        raise('the length of bgs and pairlist must be equal.')
    if isinstance(color, str):
        color = [color] * ncells

    line_sample = line_sample or 1
    assert 0 < line_sample <=1

    nrows, ncols = colrows(ncells, nrows=nrows, ncols=ncols, soft=False)
    fig, axis = plt.subplots(nrows,ncols, 
                             figsize=((fsize+werror)*ncols,fsize*nrows),
                             sharex=sharex, sharey=sharey,**figkargs)
    axes = []
    for i in range(ncells):
        if ncells ==1:
            ax = axis
        elif min(nrows, ncols)==1:
            ax = axis[i]
        else:

            ax = axis[i//ncols,i%ncols]
        axes.append(ax)

    for i in range(ncells):
        axa = axes[i]
        posa= pairlist[i]
        if (not bgs is None):
            bga = bgs[i]
            if (bga.ndim ==2) and (bga.shape[1]==2):
                axa.scatter(bga[:,0], bga[:,1], s=bg_size, c=bg_color)
            elif (bga.shape[1]>2):
                axa.imshow(bga, aspect=aspect, cmap=cmap, alpha=alpha )
        axa.scatter(posa[:,0], posa[:,1], s=size, c=color[i])

        if not titles is None:
            axa.set_title(titles[i])
        if hide_axis:
            axa.set_axis_off()

        if invert_xaxis:
            axa.invert_xaxis()
        if invert_yaxis:
            axa.invert_yaxis()

        if equal_aspect:
            axa.set_aspect('equal', adjustable='box')
        axa.grid(grid)

    if show_line or (line_width> 0):
        if pairidx is None:
            pairidx = [ [i, i +1] for i in range(ncells-1) ]
        if isinstance(line_color, str):
            line_color = [line_color] * len(pairidx)

        for i,(r,q) in enumerate(pairidx):
            rpair = pairlist[r]
            qpair = pairlist[q]
            rax = axes[r]
            qax = axes[q]
            if matches is None:
                ridx = qidx = range(min(len(rpair),len(qpair)))
            else:
                assert len(matches) == len(pairidx), 'the length of pairidx and matches must be equal.'
                ridx = matches[i][0]
                qidx = matches[i][1]
            if line_sample <1:
                smpidx =  np.random.choice(len(ridx), size=int(line_sample*len(ridx)), replace=False)
            else:
                smpidx = range(len(ridx))
            for k in smpidx:
                xy1 = rpair[ridx[k]]
                xy2 = qpair[qidx[k]]

                con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", 
                                      coordsB="data", axesA=rax, axesB=qax, 
                                      alpha=line_alpha,
                                      color=line_color[i], 
                                      arrowstyle=arrowstyle, **linekargs)
                con.set_linewidth(line_width)
                fig.add_artist(con)
                if (not line_limit is None) and (k >= line_limit-1):
                    break

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols - ncells):
            fig.delaxes(axis.flatten()[-j-1])
    fig.tight_layout()
    if save:
        fig.savefig(save)
    if show is True:
        fig.show()
    elif show is False:
        plt.close()
    else:
        return fig, axis

def plt_fit(img, pos, figsize = (10,10), show=True):
    fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.imshow(img,cmap = 'gray')
    ax.scatter(pos[:,0],pos[:,1], color='red', s=30)
    ax.scatter(pos[:,4],pos[:,5], color='blue',  s=30)
    ax.set_axis_off()

    if show:
        fig.show()
    else:
        return (fig, ax)

def parahist(model,  
             fsize=5, 
                werror=0,
                nrows=None,
                ncols=2, 
                show=True, 
                save=None,
                bins=50,
                grid=False,
                axis_off=False,
                sharex=False, sharey=False,
                **kargs):

    ncells = len(list(model.named_parameters())) 
    nrows, ncols = colrows(ncells, ncols=ncols)
    fig, axes = plt.subplots(nrows, ncols*2,
                                figsize=((fsize+werror)*ncols*2, fsize*nrows),
                                sharex=sharex, sharey=sharey,**kargs)
    i = 0
    for name, para in model.named_parameters():
        irow = i //ncols
        icol0 = (i % ncols) *2
        icol1 = icol0 + 1

        if min(nrows, ncols)==1:
            aw, ag = axes[icol0], axes[icol1] 
        else:
            aw, ag = axes[irow, icol0], axes[irow, icol1]
        i +=1

        try:
            W = para.data.detach().cpu().numpy().flatten()
            aw.hist(W, bins=bins, color='red', **kargs)
        except:
            pass
        try:
            G = para.grad.data.detach().cpu().numpy().flatten()
            ag.hist(G, bins=bins, color='blue', **kargs)
        except:
            pass

        aw.set_title(f'{name}_weigth')
        ag.set_title(f'{name}_grad')


        aw.grid(grid)
        ag.grid(grid)

        aw.set_xlabel(f'{name}_weigth')
        ag.set_xlabel(f'{name}_grad')

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols*2 - ncells*2):
            fig.delaxes(axes.flatten()[-j-1])
    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        fig.show()
    else:
        plt.close()

def imagemappoints(images, points, titles=None,
                   ncols=None,
                   equal_aspect = False,
                   fsize = 5, werror=0, sharex=True, sharey=True, grid=True,
                   show=True, save=None, fkarg={}, **kargs):
    ncells = len(images) 
    nrows, ncols = colrows(ncells, ncols=ncols)
    fig, axes = plt.subplots(nrows, ncols,
                                figsize=((fsize+werror)*ncols, fsize*nrows),
                                sharex=sharex, sharey=sharey,**fkarg)
    for i in range(ncells):
        if ncells ==1:
            ax = axes
        elif min(nrows, ncols)==1:
            ax = axes[i]
        else:
            ax = axes[i//ncols,i%ncols]
        imagemappoint(images[i], points[i], ax=ax, **kargs)

        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')
        ax.grid(grid)
        if not titles is None:
            ax.set_title(titles[i])

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols*2 - ncells*2):
            fig.delaxes(axes.flatten()[-j-1])
    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        fig.show()
    else:
        plt.close()

def imagemappoint(image, points, figsize=(7,7), s=1, color='red',
                  rescale=None, edgecolor=None, marker='.',
                   alpha=1, ax=None,  **kargs):
    #import matplotlib
    #Redb = matplotlib.colors.LinearSegmentedColormap.from_list("Redb", ['white', 'red'])
    if not rescale is None:
        hw =image.shape[:2]
        resize = [int(round(hw[0] * rescale ,0)), int(round( hw[1] *rescale ,0))]
        image = skitf.resize(image, resize)
        points = points*rescale

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.imshow(image)
    ax.scatter(points[:,0], points[:,1], s=s, color=color, 
                edgecolor=edgecolor, alpha=alpha, marker=marker, **kargs)
    if ax is None:
        fig.show()


def imagehist(img, layout='rgb', figsize=(20,5), 
                   logyscale=True,
                   bin = None, iterval=None, show=True,):

    iterval=(0, 1)
    bins=100
    xtick = np.round(np.linspace(0,1,bins+1, endpoint=True), 2)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    for i in range(img.shape[0]):
        x = img[i].flatten()
        counts, values=np.histogram(x, bins=bins, range=iterval)
        max_value = int(values[np.argmax(counts)])

        xrange = np.array([values[:-1], values[1:]]).mean(0)
        ax.plot(xrange, counts) #, label=f"{i} {layout[i]} {max_value}", color=layout[i])

    ax.legend(loc="best")
    ax.set_xticks(xtick)
    ax.set_xticklabels(
        xtick,
        rotation=90, 
        ha='center',
        va='center_baseline',
        fontsize=10,
    )
    if logyscale:
        ax.set_yscale('log')
    #ax.set_axis_on()
    if show:
        fig.show()
    else:
        plt.close()

def spatialscattersplit( adatalist, group,
                        group_order=None, 
                        cmap = None,
                        save=None, 
                        nrows = 4,
                        ncols = 4,
                        fsize = 7,

                        lloc='center left',
                        markerscale=4, 
                        lncol=1,
                        mode='expand',
                        frameon=False,
                        bbox_to_anchor=(1, 0, 0.5, 1), #(1, 0.5),
                        borderaxespad=0.5,
                        largs={},
                        titledict={},
                        legend_num = 0,
                         werror =0, show=True, **kargs):
    ncells = len(adatalist)
    fig, axes = plt.subplots(nrows,ncols, figsize=((fsize+werror)*ncols,fsize*nrows))

    for i in range(ncells):
        if ncells ==1:
            ax = axes
        elif min(nrows, ncols)==1:
            ax = axes[i]
        else:
            ax = axes[i//ncols,i%ncols]
        adata = adatalist[i].copy()        
        if not group_order is None:
            adata.obs[group] = pd.Categorical(adata.obs[group], categories=group_order)
        #sc.pl._tools.scatterplots._get_palette(adata, group)
            
        sc.pl.spatial(adata, 
                      color=group,
                      img_key='hires',
                      show=False,
                      ax=ax,
                      
                      **kargs)
        if i ==legend_num:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.set_title(str(i), **titledict)
    fig.legend(handles, labels,
                ncol=lncol,
                loc=lloc, 
                frameon=frameon,
                mode=mode,
                markerscale=markerscale,
                bbox_to_anchor=bbox_to_anchor,
                borderaxespad =borderaxespad,
                **largs)

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols - ncells):
            fig.delaxes(axes.flatten()[-j-1])
    fig.tight_layout()
    if save:
        fig.savefig(save)
    if show is True:
        fig.show()
    elif show is False:
        plt.close()
    else:
        return fig, axes

def splitplot(adata, groupby='sampleid', splitby=None, basis='X_umap', method=None,
                legend_loc='on data',
                legend_ax = None,
                lloc="best", # lloc="center left",
                show_background=True,
                ncols=5, scale=5, werror=0, size=None, markerscale=4,
                bbox_to_anchor=None,# bbox_to_anchor=(1, 0, 0.5, 1),
                lncol=1, mode=None,
                set_aspect=1,
                invert_xaxis=False,
                invert_yaxis=False,
                axis_label=None,
                sharex=True, sharey=True,
                bg_size=4, fontsize=10, bg_color='lightgray', 
                shows=True, save=None,  
                left=None, bottom=None,
                right=None, top=None, 
                wspace=None, hspace=None,
                **kargs):
    adata.obsm[f'X_{basis}'] = adata.obsm[basis]
    if method is None:
        if basis in ['X_umap','umap']:
            method = 'umap'
        elif basis in ['X_tsne','tsne']:
            method = 'tsne'
        else:
            method = 'scatter'

    import math
    try:
        G = adata.obs[groupby].cat.remove_unused_categories().cat.categories
    except:
        G = adata.obs[groupby].unique()
    if len(G) < ncols: ncols=len(G)
    nrows = math.ceil(len(G)/ncols)

    if splitby:
        _data  = adata.obsm[basis]
        _sccor = sc.pl._tools.scatterplots._get_palette(adata, splitby)
        _datacor = adata.obs[splitby].map(_sccor)
        try:
            S = adata.obs[splitby].cat.remove_unused_categories().cat.categories
        except:
            S = adata.obs[splitby].unique()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*(scale+werror), nrows*scale), sharex=sharex, sharey=sharey)
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    for n,i in enumerate(G):
        
        if nrows==1:
            AX = axes[n]
        else:
            AX = axes[n//ncols,n%ncols]
        if invert_xaxis:
            AX.invert_xaxis()
        if invert_yaxis:
            AX.invert_yaxis()

        if splitby is None:
            if method in ['scatter', 'embedding']:
                eval('sc.pl.%s'%method)(adata, basis=basis, color=groupby, groups =i, show=False,
                        size=size, title=i, legend_loc =legend_loc, ax=AX, **kargs)
            else:
                eval('sc.pl.%s'%method)(adata, color=groupby, groups =i, na_in_legend =False, show=False,
                                        size=size, title=i, legend_loc =legend_loc, ax=AX, **kargs)
        else:
            _idx = adata.obs[groupby]==i
            size = size or 5
            if show_background:
                AX.scatter( _data[:, 0][~_idx], _data[:, 1][~_idx], s=bg_size, marker=".", c=bg_color)
            _sid = [k for k in S if k in adata.obs.loc[_idx, splitby].unique()]
            if len(_sid)>0:
                for _s in _sid:
                    _iidx = ((adata.obs[groupby]==i) & (adata.obs[splitby]==_s))
                    AX.scatter(_data[:, 0][_iidx], _data[:, 1][_iidx], s=size,  marker=".", c=_datacor[_iidx], label=_s)
            AX.set_title(i,size=fontsize)
            if not axis_label is False:
                basis = basis if axis_label is None else axis_label
                AX.set_ylabel((basis+'1').upper(), fontsize = fontsize)
                AX.set_xlabel((basis+'2').upper(), fontsize = fontsize)

            AX.set_yticks([])
            AX.set_xticks([])
            AX.set_aspect(set_aspect)
            AX.grid(False)
            if (legend_ax is None) or (n in legend_ax):
                AX.legend(loc=lloc, bbox_to_anchor=bbox_to_anchor, 
                            mode=mode, ncol = lncol, markerscale=markerscale)

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')

    if nrows*ncols - len(G) >0:
        for j in range(nrows*ncols - len(G)):
            fig.delaxes(axes[-1][ -j-1])
            #fig.delaxes(axes.flatten()[-j-1])

    fig.tight_layout()
    if save:
        plt.savefig(save)
    if shows:
        plt.show()
    else:
        return fig, axes

