import numpy as np
import pandas as pd
import scanpy as sc
import json

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

def load_sample(sid):
    # import sys
    # import importlib
    # sys.path.append('/home/gpfs/home/wulab17/JupyterCode')
    # import SCFunc
    # importlib.reload(SCFunc)
    # from SCFunc import Preprocession
    
    data_path = '/share/home/zhonw/WorkSpace/11Project/04GNNST/01DataBase/04ZSJ'
    adata = read_h5_st(data_path, sid, use_diopy=False)
    adata.obs_names = adata.obs_names+':'+sid
    adata.var_names_make_unique()
    #adata = Preprocession(adata).Normal() 
    #adata.layers['count'] = adata.raw.X.copy()
    #adata = sc.pp.subsample(adata, n_obs=8000, random_state=0, copy=True)
    print(adata.shape)
    
    adata.var.index.name='Gene'
    adata.obs['SID'] = sid 
    adata.raw= adata    
    return adata