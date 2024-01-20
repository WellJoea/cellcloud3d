import pandas as pd
import numpy as np
import scanpy as sc

def imread_h5(file, key=None, get='h5_data', **kargs):
    import h5py
    with h5py.File(file, 'r',  **kargs) as h5f:
        if get == 'h5':
            return h5f
        key = list(h5f.keys())[0] if key is None else key
        img  = h5f.get(key)
        if get in ['h5_data']:
            return (img, h5f)
        elif get in ['data','image','img']:
            return img
        elif get=='array':
            img = np.array(img)
            h5f.close()
            return img

def read_image_info(adata, img_key="hires", basis = None,  order=None,
                    library_id=None, get_pix_loc=False, rescale=None):

    basis = basis or 'spatial'
    rescale = rescale or 1
    library_id = list(adata.uns[basis].keys())[0] if (library_id is None) else library_id
    
    img_dict = adata.uns[basis][library_id]
    iimg = img_dict['images'][img_key]
    scale_factor = img_dict['scalefactors'].get(f'tissue_{img_key}_scalef', 1)
    spot_diameter_fullres = img_dict['scalefactors'].get('spot_diameter_fullres',1)
    
    scales = scale_factor*rescale
    if rescale != 1:
        # import cv2
        # rsize = np.round(np.array(iimg.shape[:2])*rescale, 0)[::-1].astype(np.int32)
        # iimg = cv2.resize(iimg[:,:,::-1].copy(), rsize, interpolation= cv2.INTER_LINEAR)        
        import skimage as ski
        rsize = np.round(np.array(iimg.shape[:2])*rescale, 0).astype(np.int32)
        iimg = ski.transform.resize(iimg.copy(), rsize, order=order)
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
    #coor.index = coor.index + f':{sid}'
    image = np.transpose(np.load( f'{path}/{sid}.image.npy'), axes=(1,0,2))
    image = np.clip(np.round(image*255), 0, 255).astype(np.uint8)

    print(sf_info, coor.shape, adata.shape, image.shape)
    assert (coor.index != adata.obs_names).sum() == 0
    adata.obs[coor.columns] = coor
    adata.obsm['spatial'] = coor[['imagerow', 'imagecol']].values
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
