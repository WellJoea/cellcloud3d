import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px #5.3.1
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scanpy as sc
from cellcloud3d.plotting._matplt3d import matplt3d

#%matplotlib inline

def get_slice_score(adatas, groups, basis=None):
    basis = 'spatial' if basis is None else basis
    obs_df = []
    for i, adata in enumerate(adatas):
        locs = adata.obsm[basis]
        obs = adata.obs[groups].copy()
        obs[['X', 'Y']] = locs
        obs['Z'] = i
        obs_df.append(obs)
    obs_df = pd.concat(obs_df, axis=0)
    return obs_df

def vartype(pdseries):
    if pdseries.dtype in ['float32', 'float64', 'float', 'int32', 'int64', 'int']:
        return 'continuous'
    elif pdseries.dtype in ['category', 'object', 'bool']:
        return 'discrete'

def dynam3d(Data, X, Y, Z, group, save=None, show=False, size=2, invert_zaxis=True):    
    X,Y,Z = Data[Z], Data[X], Data[Y]
    p3d = matplt3d(dpi = 300)
    p3d.scatter3D_con(X, Y, Z, size=size, colorcol=Data[group], vmax=1, vmin=0)
    #p3d.setbkg()
    try:
        p3d.ax.set_aspect('equal', 'box')
    except:
        asx, asy, asz = np.ptp(X), np.ptp(Y), np.ptp(Z)
        p3d.ax.set_box_aspect([asx, asy, asz])
    p3d.ax.set_axis_off()
    p3d.ax.set_title(group)
    if invert_zaxis:
        p3d.ax.invert_zaxis()
    p3d.adddynamic(title=group, save=save)

def read_h5_st(path, sid, use_diopy=False,
               assay_name='Spatial',
               slice_name=None):
    slice_name = sid if slice_name is None else slice_name 
    if use_diopy:
        import diopy 
        adata =  diopy.input.read_h5(f'{path}/{sid}.h5', assay_name=assay_name)
    else:
        adata = sc.read(f'{path}/{sid}.h5ad')

    with open(f'{path}/{sid}.scale.factors.json', 'r') as f:
        sf_info = json.load(f)
        
    coor = pd.read_csv( f'{path}/{sid}.coordinates.csv',index_col=0)
    image = np.load( f'{path}/{sid}.image.npy')
    print(sf_info, image.shape)
    assert (coor.index != adata.obs_names).sum() == 0
    adata.obs[coor.columns] = coor
    adata.obsm['spatial'] = coor[['imagecol', 'imagerow']].values
    adata.uns['spatial'] = {}
    adata.uns['spatial'][slice_name] ={
        'images':{'hires':image, 'lowres':image},
        #unnormalized.radius <- scale.factors$fiducial_diameter_fullres * scale.factors$tissue_lowres_scalef
        #spot.radius <-  unnormalized.radius / max(dim(x = image))
        'scalefactors': {'spot_diameter_fullres': sf_info['fiducial'], ##??
                         'fiducial_diameter_fullres': sf_info['fiducial'],
                         'tissue_hires_scalef': sf_info['hires'], # ~0.17.
                         'tissue_lowres_scalef': sf_info['lowres'],
                         'spot.radius': sf_info['spot.radius'], 
                        },
        'metadata': {'chemistry_description': 'custom',
                       'spot.radius':  sf_info['spot.radius'], 
                       'assay': sf_info['assay'], 
                       'key': sf_info['key'], 
                      }
    }
    return(adata)

def str2list(lists, values=None):
    if lists is None:
        lists = values
    if type(lists) in [str, int, float, bool]:
        return [lists]
    else:
        return lists

def get_color(adata, values=None, value=None,  palette=None):
    if not values is None:
        value = "pie" if value is None else value
        celllen = adata.shape[0]
        add_col = int(np.ceil(celllen/len(values)))
        add_col = values * add_col 
        adata.obs[value] = pd.Categorical(add_col[:celllen], categories=values) 
    sc.pl._tools.scatterplots._get_palette(adata, value, palette=palette)
    
def get_spatial_info(adata, img_key="hires", basis = None, 
                    library_id=None, get_pix_loc=False, rescale=None):

    basis = 'spatial' if basis is None else basis
    rescale = 1 if rescale is None else rescale
    library_id = list(adata.uns[basis].keys())[0] if (library_id is None) else library_id
    
    img_dict = adata.uns[basis][library_id]
    iimg = img_dict['images'][img_key]
    scale_factor = img_dict['scalefactors'].get(f'tissue_{img_key}_scalef', 1)
    spot_diameter_fullres = img_dict['scalefactors'].get('spot_diameter_fullres',1)
    
    scales = scale_factor*rescale
    if rescale != 1:
        import cv2
        rsize = np.ceil(iimg.shape[:2]/rescale)[::-1].astype(np.int32)
        iimg = cv2.resize(iimg[:,:,::-1].copy(), rsize, interpolation= cv2.INTER_LINEAR)        
    locs = pd.DataFrame(adata.obsm[basis] * scales, 
                        index=adata.obs_names)
    
    st_loc = np.round(locs, decimals=0).astype(np.int32)
    #iimg = np.round(iimg*255)
    #iimg = np.clip(iimg, 0, 255).astype(np.uint32)
    if get_pix_loc:
        st_img = np.zeros(iimg.shape[:2], dtype=bool)
        st_img[st_loc[:,1], st_loc[:,0]] = True
        from scipy.ndimage.morphology import binary_dilation
        pix = np.round(spot_diameter_fullres*scale_factor/2).astype(np.int32)
        strel = np.ones((pix, pix))
        st_img = binary_dilation(st_img, structure=strel).astype(np.int32)
    else:
        st_img = st_loc
    return {"img":iimg,
            "locs":locs, 
            'loc_img':st_img,
            'scale_factor':scale_factor, 
            'rescale': scales,
            'spot_size':spot_diameter_fullres }

def loc_pie(ax, ratios, colors,  X=0, Y=0, size = 1500, curnum=30, **kargs): 
    xy = []; 
    s=[]
    start = 0
    for ratio in ratios:
        y = [0] + np.cos(np.linspace(2*np.pi*start,2*np.pi*(start+ratio),curnum)).tolist() #30
        x = [0] + np.sin(np.linspace(2*np.pi*start,2*np.pi*(start+ratio),curnum)).tolist() #30
        xy1 = np.column_stack([x, y])
        s1 = np.abs(xy1).max()

        xy.append(xy1)
        s.append(s1)
        start += ratio

    for i, (xyi, si) in enumerate(zip(xy,s)):
        #ax.scatter(xyi[:,0], xyi[:,1], c=colors[i])
        ax.scatter([X],[Y] ,
                   marker=xyi, 
                   s=size, #*si**2,
                   facecolor=colors[i],
                   **kargs)

def scatter_pie0(ax, XYs, ratios, colors, scale_ratio=True, size = 1500, **kargs):
    if scale_ratio:
        ratios = np.asarray(ratios)
        ratios /= ratios.sum(1)[:,None]
    assert ratios.shape[1] <= len(colors), print('length colors < ratios!')
    for i in range(len(XYs)):
        iX, iY = XYs[i]
        iratios = ratios[i]
        loc_pie(ax, iratios, colors, 
                X=iX,
                Y=iY, 
                size = size, 
                **kargs)

def scatter_pie(ax, XYs, ratios, colors, scale_ratio=True, curnum = 100, size = 1500, **kargs):
    keep =  ratios.sum(1) >0
    if sum(keep) <- len(ratios):
        print('exsit zeores lines, will drop it.')
        ratios = ratios[keep]
        XYs = XYs[keep]

    if scale_ratio:
        ratios = np.asarray(ratios)
        ratios /= ratios.sum(1)[:,None]
    assert ratios.shape[1] <= len(colors), print('length colors < ratios!')

    colors = np.asarray(colors)
    circus = np.linspace(0, 1, curnum)
    circus = np.c_[np.sin(2*np.pi*circus), np.cos(2*np.pi*circus)]
    cratio = np.round(ratios* curnum, decimals=0).astype(np.int32)
    for i, icratio in enumerate(cratio):
        idx = icratio >0
        icol = colors[idx]
        icrat = icratio[idx]
        XY = XYs[i]
        start = 0
        for j in range(len(icrat)):
            end = start + icrat[j]
            marker =  np.r_[[[0,0]], circus[start:end,]]
            s = np.abs(marker).max() ** 2 * size
            ax.scatter(*XY,
                        marker=marker, 
                        s=s,
                        facecolor=icol[j],
                        **kargs)
            start = end

def spatialpie(adata, 
               img_key='lowres', 
               library_id=None, 
               colors=None,
               obsm_df=None,
               color_df =None,
               show=True,
               basis='spatial',
               get_pix_loc=False,
               scale_ratio=False,
               figsize=(10,10),
               curnum=1000,
               min_thred =1./1000,
               max_thred =None,
               image_alpha=1,
               show_img=True,
               show_pie=True,
               cmap=None,
               name=None,
               ax=None,
               rescale=1,
               size=1,
               legend=True,
               exist_show=True,
               lncols=1,
               title=None,
               ltitle=None,
               dpi=300,
               save=None,
               lloc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1),
               xlim=None,
               ylim=None,
               imgargs = {'origin':'upper'},
               **kargs):
    #https://altair-viz.github.io/gallery/isotype.html
    sinfo = get_spatial_info(adata, 
                             img_key=img_key, 
                             basis = basis, 
                             library_id=library_id, 
                             get_pix_loc=get_pix_loc, 
                             rescale=rescale)
    img = sinfo.get('img')
    st_loc = sinfo.get('locs').values
    scale_factor = sinfo.get('scale_factor',1)
    spot_size = sinfo.get('spot_size',1)

    circle_radius = size * scale_factor * spot_size * rescale * 2
    if not color_df is None:
        ratios = color_df
        colors = color_df.columns.tolist()
    else:
        colors = str2list(colors)
        ratios = adata.obs[colors]
    
    if not min_thred is None:
        ratios[ratios<min_thred]= 0
    ratios[ratios<1./curnum] = 0
    ratios = ratios[(ratios.sum(1)>0)]
    ratios = ratios[(np.isin(ratios.index, adata.obs_names))]

    if not obsm_df is None:
        st_loc = obsm_df
    st_loc = st_loc[np.isin(adata.obs_names, ratios.index), ]

    get_color(adata, values=colors, value=None, palette=cmap)
    piecolor = adata.uns['pie_colors']

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    if show_img:
        ax.imshow(img, alpha=image_alpha, **imgargs)
    if show_pie and ratios.shape[0]>0:
        scatter_pie(ax, st_loc, ratios.values, piecolor,
                    scale_ratio=scale_ratio,
                    curnum=max(curnum,scale_ratio),
                    size=circle_radius, 
                    **kargs )
    if legend:
        import matplotlib.patches as mpatches
        keep_col = ratios.columns[(ratios.sum(0)>0)]

        if exist_show:
            patches = [ mpatches.Patch(color=c, label=l) 
                           for l,c in zip(colors, piecolor) 
                       if l in keep_col]
        else:
            patches = [ mpatches.Patch(color=c, label=l) 
                           for l,c in zip(colors, piecolor)]
        ax.legend(handles=patches,
                  title=ltitle,
                  loc=lloc, 
                  ncols=lncols,
                  bbox_to_anchor=bbox_to_anchor)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(title)
    if not xlim is None:
        ax.set(xlim=xlim)
    if not ylim is None:
        ax.set(ylim=ylim)
    #ax.set_xlabel()
    #ax.set_ylabel()
        
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi)
    if show is True:
        plt.show()
    elif show is False:
        plt.close()
    else:
        return ax
    
def spatialpiesplit( adata, predict_df, save=None,  nrows = 6, ncols = 6, fsize = 7,
                    size=1, werror =0, show=True, **kargs):
    ncells = predict_df.shape[1]
    colors = adata.uns['pie_colors']

    fig, axes = plt.subplots(nrows,ncols, figsize=((fsize+werror)*ncols,fsize*nrows))
    #fig.patch.set_facecolor('xkcd:mint green')
    for i in range(ncells):
        if ncells ==1:
            ax = axes
        elif min(nrows, ncols)==1:
            ax = axes[i]
        else:
            ax = axes[i//ncols,i%ncols]

        i_df = predict_df.iloc[:,[i]].copy()
        i_title =  i_df.columns[0]
        spatialpie(adata.copy(),
                  img_key='hires', 
                  #colors=colors,
                  color_df = i_df,
                  cmap=[colors[i]],
                  size=size,
                  show=False,
                  legend=False,
                  curnum=1000,
                  min_thred =1./1000,
                  title = i_title,
                  ax=ax,
                  **kargs)

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