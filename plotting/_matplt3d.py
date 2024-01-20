import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cellcloud3d.plotting._colors import vartype, color_palette, cmap1, cmaptolist

class matplt3d:
    def __init__(self, figsize=(10,10), dpi = 200,
                labelsize = 18, font_family = 'serif',
                font_serif = ['Computer Modern Roman'],
                axes_grid = False, usetex = True,
                **kargs):
        from matplotlib import rcParams
        rcParams['axes.labelsize'] = labelsize
        rcParams['font.family'] = font_family
        rcParams['font.serif'] = font_serif
        rcParams['text.usetex'] = usetex
        rcParams['axes.grid'] = axes_grid
        plt.style.use('default')

        self.fig = plt.figure(figsize=figsize, dpi=dpi, **kargs)
        self.ax = self.fig.add_subplot(projection='3d')

        self.vartype = vartype
        self.color_palette = color_palette
        self.cmaptolist = cmaptolist

    def scatter3D_cat(self, x, y ,z, groupby=None, cmap=None, alpha=1, size=0.8, lengend=True, 
                      loc="center left", 
                      anchor=(0.93, 0, 0.5, 1),
                      frameon=False,
                      markerscale=5, scatterpoints=1,
                      edgecolors=None,linewidths=0,
                      legend_fsize = 10,
                      lncol=None, mode='expand',largs={},
                      lengend_title=None, **kargs ):

        if groupby is None:
            self.ax.scatter(x, y, z, c=cmap, s=size, alpha=alpha)
        else:
            try:
                labels = groupby.cat.remove_unused_categories().cat.categories
            except:
                labels = groupby.unique()
        
            if type(cmap) is str and cmap in plt.colormaps():
                colorlist = plt.get_cmap(cmap)(range(len(labels)))
            elif type(cmap) is list:
                colorlist = cmap
            else:
                colorlist = color_palette(len(labels))

            for i, c, label in zip(range(len(labels)), colorlist, labels):
                widx = (groupby == label)                
                self.ax.scatter(x[widx], y[widx], z[widx], s=size, 
                                 c=c, 
                                 label=label, 
                                 alpha=alpha,
                                 linewidths=linewidths,
                                 edgecolors=edgecolors, 
                                 **kargs)
        if lengend:
            if lncol is None:
                icol = max(1, int(np.ceil(len(labels)/15)))
            elif isinstance(lncol, int): 
                icol = lncol

            self.ax.legend(title=lengend_title, loc=loc, ncol=icol,
                           prop={'size':legend_fsize},
                           scatterpoints=scatterpoints,
                           bbox_to_anchor=anchor,
                            frameon=frameon,
                            mode=mode,
                            markerscale=markerscale,
                            **largs)

    def scatter3D_con(self, x, y ,z, groupby=None, cmap='viridis', alpha=1, size=0.8, label=None, lengend=True, 
                        loc="center left", markerscale=5, lengend_title=None, edgecolors=None,linewidths=0,
                        norm_color=None, shrink=1, vmin=None, vmax=None, bbox_to_anchor=(0.93, 0, 0.5, 1),
                        caxpos = [0.95, 0.4, 0.012, 0.3], **kargs):
        if cmap is None:
            cmap = cmap1
        sct3 = self.ax.scatter(x, y, z, c=groupby, cmap= cmap, s=size, alpha=alpha, linewidths=linewidths,
                                vmin=vmin, vmax=vmax,
                               edgecolors=edgecolors, **kargs)
        if lengend:
            cax =  None if caxpos is None  else self.fig.add_axes(caxpos)
            if norm_color is None:
                self.fig.colorbar(sct3, ax=self.ax, label=lengend_title, cax=cax, shrink=shrink)
            else:
                norm=mpl.colors.Normalize(vmin=norm_color[0], vmax=norm_color[1])
                self.fig.colorbar(sct3, ax=self.ax, label=lengend_title, cax=cax, norm =norm)

    def quivar3D(self, x, y, z, u, v, w, length=0.1, **kargs):
        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        q = ax.quiver(x, y, z, u, v, w, cmap = cmap, length=0.1)
        q.set_array(np.linspace(0,max_norm,10))
        fig.colorbar(q)
        q.set_edgecolor(c)
        q.set_facecolor(c)
        '''
        self.ax.quiver(x, y, z, u, v, w, length=length, **kargs)

    def line3D(self, X, Y, Z, color='black', **kargs):
        self.ax.plot(X, Y, Z, c=color, **kargs)

    def scatter3D(self, X, Y, Z, color='black', **kargs):
        self.ax.scatter(X, Y, Z, c=color, **kargs)
        
    def line3Dsig(self, SEG, color='black', **kargs):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        lc = Line3DCollection(SEG, color='black', **kargs)
        self.ax.add_collection3d(lc)

    def setbkg(self, axislabel='Dim', title=None,
               invert_xaxis=False, invert_yaxis=False, invert_zaxis = False,
               aspect_equal= False, box_aspect = None,  #box_aspect=[5,5,5],
               hide_axis=True, hide_grid=True,):
        # Get rid of the panes
        try:
            self.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        except:
            self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the spines
        #self.ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        #self.ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        #self.ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        #self.ax.set_xticks([]) 
        #self.ax.set_yticks([]) 
        #self.ax.set_zticks([])
        if hide_grid:
            self.ax.xaxis._axinfo["grid"]['linewidth'] = 0
            self.ax.yaxis._axinfo["grid"]['linewidth'] = 0
            self.ax.zaxis._axinfo["grid"]['linewidth'] = 0

        if invert_xaxis:
            self.ax.invert_xaxis()
        if invert_yaxis:
            self.ax.invert_yaxis()
        if invert_zaxis:
            self.ax.invert_zaxis()

        self.ax.set_xlabel(f'{axislabel}_1' )
        self.ax.set_ylabel(f'{axislabel}_2' )
        self.ax.set_zlabel(f'{axislabel}_3' )
        self.ax.set_title(title)

        if hide_axis:
            self.ax.set_axis_off()
        if aspect_equal:
            self.ax.set_aspect('equal', 'box')

        if not box_aspect is None:
            self.ax.set_box_aspect(box_aspect)

    def adddynamic(self, angle=5, elev = 0, interval=50, fps=10, dpi=100, 
                   vertical_axis='z',
                    bitrate=1800, save=None, show=True):
        from matplotlib import animation
        def rotate(angle):
            self.ax.view_init(elev =elev, azim=angle, vertical_axis=vertical_axis)

        if show:
            self.fig.show()
        if save:
            ani = animation.FuncAnimation(self.fig, rotate, frames=np.arange(0, 360, angle), interval=interval)
            if save.endswith('gif'):
                ani.save(save, writer=animation.PillowWriter(fps=fps), dpi=dpi)
            elif save.endswith('mp4'):
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
                ani.save(save, writer=writer, dpi=dpi)
            self.ani = ani

def lineMatches3D(matches, bgs=None, groupby=None, line_cmap = 'viridis',
                  line_color='gray',  line_width=1, line_alpha = 1, color='b',
                   line_weight = None, figsize = (10,10),
                   axislabel='Dim', title=None,
                   invert_xaxis=False, 
                   invert_yaxis=False, invert_zaxis = False,
                   aspect_equal= False, box_aspect=None,
                   hide_axis=True, hide_grid=True,
                  cmap = None,
                  line_limit = None,
                  line_sample = None,
                   seed=None, 
                  angle=5, interval=50,  elev=10,
                  fps=10, dpi=100, vertical_axis='z',
                  save=None, show=True,
                  labelsize = 18, font_family = 'serif',
                  font_serif = ['Computer Modern Roman'],
                  axes_grid = False, usetex = True,
                  **kargs):
    '''
    matches: np.narray([['x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2']])
    '''
    mpl = matplt3d(figsize=figsize, dpi = dpi,
                        labelsize = labelsize, 
                         font_family = font_family,
                        font_serif = font_serif,
                        axes_grid = axes_grid, usetex = usetex)
    kidx = np.arange(matches.shape[0])
    if not line_limit is None:
        kidx = kidx[:line_limit]
        matches = matches[kidx,:]

    line_sample = line_sample or 1
    assert 0 < line_sample <=1
    if line_sample <1:
        np.random.seed(seed)
        kidx =  np.random.choice(kidx, size=int(line_sample*kidx.shape[0]), replace=False)
        matches = matches[kidx,:]

    if not line_weight is None:
        line_weight = np.array(line_weight)[kidx]
        line_weight = line_weight/line_weight.max()
        line_widths = line_weight * line_width
    else:
        line_widths = np.ones(kidx.shape[0]) * line_width

    if not line_weight is None:
        line_colors = mpl.cmaptolist(line_cmap, spaces=line_weight)
    else:
        line_colors = [line_color] * matches.shape[0]

    for i in range(matches.shape[0]):
        mpl.line3D(matches[i,[0,3]], matches[i,[1,4]], matches[i,[2,5]], 
                   color=line_colors[i], alpha=line_alpha, 
                   linewidth=line_widths[i])

    if not bgs is None:
        if mpl.vartype(groupby) == 'discrete':
            mpl.scatter3D_cat(bgs[:,0], bgs[:,1], bgs[:,2], groupby=groupby, **kargs)
        elif mpl.vartype(groupby) == 'continuous':
            mpl.scatter3D_con(bgs[:,0], bgs[:,1], bgs[:,2], groupby=groupby, **kargs)

    mpl.setbkg( axislabel=axislabel,
                invert_xaxis=invert_xaxis,
               invert_yaxis=invert_yaxis, 
               invert_zaxis = invert_zaxis,
               aspect_equal= aspect_equal,
               box_aspect = box_aspect,
               hide_axis=hide_axis,
               hide_grid=hide_grid)

    mpl.adddynamic(angle=angle, 
                   interval=interval, 
                   fps=fps, 
                   dpi=dpi,
                   title=title,
                   elev=elev,
                   vertical_axis=vertical_axis,
                   save=save,
                   show=show)
    return mpl

def scatter3d_dy_sc(adata, groupby, basis='X_umap_3d', xyz=[0,1,2], axislabel='UMAP', size=1, width=14, heigth=8, 
                    dpi = 200, angle=5, elev=10, interval=50, fps=10, vertical_axis='z', 
                    title=None,alpha=1,
                    box_aspect=None, #box_aspect=[5,5,5],
                     loc="center left", 
                     invert_xaxis=False,
                    invert_yaxis=False,
                     invert_zaxis=True,
                     bbox_to_anchor=(0.93, 0, 0.5, 1),
                      frameon=False,
                      hide_axis=False,
                      aspect_equal = True,
                      markerscale=5, scatterpoints=1,
                     lncol=None, mode='expand',
                    largs={"alignment":'left',},
                    show=True, save=None):
    #from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation
    from matplotlib import rcParams
    import numpy as np

    rcParams['axes.labelsize'] = 18
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['axes.grid'] = False
    plt.style.use('default')

    fig = plt.figure(figsize=(width,heigth), dpi=dpi)
    #ax = Axes3D(fig)
    ax = fig.add_subplot(projection='3d')

    colors = adata.uns[f'{groupby}_colors']
    labels = adata.obs[groupby].cat.remove_unused_categories().cat.categories.tolist()
    mapdata= adata.obsm[basis][:,xyz]

    for i, c, label in zip(range(len(labels)), colors, labels):
        widx = (adata.obs[groupby] == label)
        imap = mapdata[widx,:]
        
        ax.scatter(imap[:, 0], imap[:, 1], imap[:, 2], 
                s=size, 
                c=c, 
                label=label, 
                alpha=alpha)

    # Get rid of the panes
    try:
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    except:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    #ax.set_xticks([]) 
    #ax.set_yticks([]) 
    #ax.set_zticks([])

    ax.xaxis._axinfo["grid"]['linewidth'] = 0
    ax.yaxis._axinfo["grid"]['linewidth'] = 0
    ax.zaxis._axinfo["grid"]['linewidth'] = 0

    ax.set_xlabel(f'{axislabel}_1' )
    ax.set_ylabel(f'{axislabel}_2' )
    ax.set_zlabel(f'{axislabel}_3' )
    ax.set_title(title if title else groupby)

    #ax.legend(title=groupby, loc="center left", prop={'size':10},
    #           markerscale=5, frameon=False, scatterpoints=1, bbox_to_anchor=bbox_to_anchor)
    if lncol is None:
        icol = max(1, int(np.ceil(len(labels)/15)))
    elif isinstance(lncol, int): 
        icol = lncol
    else:
        icol = lncol[n]      
    ax.legend( #title=groupby, 
                loc=loc, ncol=icol,
                prop={'size':10},
                #alignment='left',
                scatterpoints=scatterpoints,
                bbox_to_anchor=bbox_to_anchor,
                frameon=frameon,
                mode=mode,
                markerscale=markerscale,
                **largs)

    if hide_axis:
        ax.set_axis_off()

    if aspect_equal:
        try:
            ax.set_aspect('equal', 'box')
        except:
            ax.set_box_aspect(np.ptp(mapdata, axis=0))
    if not box_aspect is None:
        ax.set_box_aspect(box_aspect)

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()
    if invert_zaxis:
        ax.invert_zaxis()

    def rotate(angle):
        ax.view_init(elev=elev, azim=angle, vertical_axis=vertical_axis)

    if show:
        fig.show()
    if save:
        angle = angle
        ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=interval)
        ani.save(save, writer=animation.PillowWriter(fps=fps), dpi=dpi)

def scatter3d_dy_con(adata, groupby, basis='X_umap_3d', xyz=[0,1,2],
                     axislabel='UMAP', size=1, width=14, heigth=8, 
                     cmap='viridis',
                     vmax=None, vmin=None,
                     use_raw =False,
                     dpi = 200, angle=5,elev=10, interval=50, fps=10,
                     vertical_axis='z',
                     alpha=1,
                     box_aspect=None, #box_aspect=[5,5,5],
                     loc="center left", 
                    invert_xaxis=False,
                    invert_yaxis=False,
                     invert_zaxis=True,
                      hide_axis=False,
                      aspect_equal = True,
                      markerscale=5, 
                    show=True, save=None, **kargs):
    #from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation
    from matplotlib import rcParams
    import numpy as np

    rcParams['axes.labelsize'] = 18
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['axes.grid'] = False
    plt.style.use('default')

    #colors = adata.uns[f'{groupby}_colors']
    mapdata= adata.obsm[basis][:,xyz]
    adata = adata.raw.to_adata() if use_raw else adata
    if groupby in adata.obs_names:
        colorcol = adata.obs[groupby]
    elif groupby in adata.var_names:
        colorcol = adata[:,groupby].to_df()[groupby]
    else:
        print('cannot find {groupby} in obs and vars.')

    X,Y,Z = mapdata[:,0], mapdata[:,1], mapdata[:,2]
    p3d = matplt3d(dpi = dpi,  heigth=heigth, width=width, box_aspect=box_aspect)
    p3d.scatter3D_con(X, Y, Z, size=size, colorcol=colorcol, vmax=vmax, vmin=vmin, 
                        cmap=cmap, alpha=alpha,
                        loc=loc,  **kargs)
    #p3d.setbkg()

    if aspect_equal:
        try:
            p3d.ax.set_aspect('equal', 'box')
        except:
            p3d.ax.set_box_aspect(np.ptp(mapdata, axis=0))
    if not box_aspect is None:
        p3d.ax.set_box_aspect(box_aspect)

    if invert_xaxis:
        p3d.ax.invert_xaxis()
    if invert_yaxis:
        p3d.ax.invert_yaxis()
    if invert_zaxis:
        p3d.ax.invert_zaxis()

    if hide_axis:
        p3d.ax.set_axis_off()
    p3d.ax.set_title(groupby)


    p3d.ax.set_xlabel(f'{axislabel}_1' )
    p3d.ax.set_ylabel(f'{axislabel}_2' )
    p3d.ax.set_zlabel(f'{axislabel}_3' )

    if save:
        p3d.adddynamic(title=groupby, save=save, angle=angle, elev=elev,
                       vertical_axis=vertical_axis, dpi=dpi, interval=interval, fps=fps)
    if show:
        p3d.fig.show()
