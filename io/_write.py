
import numpy as np
import scanpy as sc
import json
import pandas as pd
import skimage as ski

def imwrite_h5(image, file, key=None, **kargs):
    import h5py
    key = 'img' if key is None else key
    with h5py.File(file, 'w') as h5f:
        h5f.create_dataset(key, data=image, **kargs)
        h5f.close()

def write_image_info(adata, image = None, locs = None, 
                     img_key="hires",
                     basis = None, 
                     library_id=None,  
                     metadata=None,
                     spot_radius = None,
                     scalefactors = None,
                     fiducial_diameter_fullres=None,
                     spot_diameter_fullres = None,
                     tissue_hires_scalef = None,
                     tissue_lowres_scalef = None,
                     keepraw=False,
                     inplace = True,
                     rescale = None,
                     order = None,
                     **kargs):
    adata = adata if inplace else adata.copy()
    basis = basis or 'spatial'
    library_id = library_id or list(adata.uns[basis].keys())[0]
    if not basis in adata.uns.keys():
        adata.uns[f'{basis}'] = {}
    if not bool(library_id)  or (not library_id in adata.uns[f'{basis}'].keys()):
        library_id = library_id or 'cc3d'
        adata.uns[f'{basis}'][library_id] = {'images':{},
                                              'metadata':{'assay': 'Spatial',
                                                         'chemistry_description': 'custom',
                                                         'spot.radius':(spot_radius or 1),
                                                         'key': 'slice1_',
                                                          }, 
                                              'scalefactors':{'fiducial_diameter_fullres':(fiducial_diameter_fullres or 1),
                                                              'spot_diameter_fullres':(spot_diameter_fullres or 1),
                                                              'spot.radius':(spot_radius or 1),
                                                              'tissue_hires_scalef':(tissue_hires_scalef or 1),
                                                              'tissue_lowres_scalef':(tissue_lowres_scalef or 1),
                                                              } 
                                        }

    if keepraw:
        if basis in adata.obsm.keys():
            adata.obsm[f'{basis}_r'] = adata.obsm[f'{basis}']
        if basis in adata.uns.keys():
            adata.uns[f'{basis}_r'] = {library_id : adata.uns[f'{basis}'][library_id]}

    if not locs is None:
        locs = np.array(locs)
        if locs.shape[0] != adata.shape[0]:
            print("Warning: the lenght of locs does not coincide with adata's")
        adata.obsm[f'{basis}'] = locs

    metadata = metadata or {}
    spot_radius and metadata.update({'spot.radius': spot_radius})

    if not 'metadata' in adata.uns[f'{basis}'][library_id].keys():
        adata.uns[f'{basis}'][library_id]['metadata'] = {}
    adata.uns[f'{basis}'][library_id]['metadata'].update(metadata)

    scalefactors = scalefactors or {}
    fiducial_diameter_fullres and scalefactors.update({'fiducial_diameter_fullres':fiducial_diameter_fullres})
    spot_radius and scalefactors.update({'spot.radius': spot_radius})
    tissue_hires_scalef and scalefactors.update({'tissue_hires_scalef': tissue_hires_scalef})
    tissue_lowres_scalef and scalefactors.update({'tissue_lowres_scalef': tissue_lowres_scalef})
    spot_diameter_fullres and scalefactors.update({'spot_diameter_fullres': spot_diameter_fullres})
    if not 'scalefactors' in adata.uns[f'{basis}'][library_id].keys():
        adata.uns[f'{basis}'][library_id]['scalefactors'] = {}
    adata.uns[f'{basis}'][library_id]['scalefactors'].update(scalefactors)

    if (not rescale is None):
        if rescale != 1:
            # rsize = np.round(np.array(image.shape[:2])*rescale, 0).astype(np.int32)
            # image = ski.transform.resize(image.copy(), rsize, order=order)
            import cv2
            rsize = np.round(np.array(image.shape[:2])*rescale, 0)[::-1].astype(np.int32)
            image = cv2.resize(image, rsize, interpolation= cv2.INTER_LINEAR)

        sizeFactor = adata.uns[f'{basis}'][library_id]['scalefactors'].get(f'tissue_{img_key}_scalef', 1)
        sizeFactor *= rescale
        adata.uns[f'{basis}'][library_id]['scalefactors'][f'tissue_{img_key}_scalef'] = sizeFactor

    if not image is None:
        adata.uns[f'{basis}'][library_id]['images'][img_key] = image

    adata.uns[f'{basis}'][library_id].update(kargs)

    if not inplace:
        return adata

def write_image_infos(adata, images = None, 
                      locs = None, 
                      library_ids=None, 
                      inplace=True, 
                     **kargs):
    adata = adata if inplace else adata.copy()

    if library_ids is None:
        library_ids = list(range(len(images)))
    assert len(images) == len(library_ids), 'images and library_ids must be the same length.'
    for idx in range(len(images)):
        image = images[idx]
        library_id = library_ids[idx]
        write_image_info(adata, image=image, library_id=library_id,
                               inplace=inplace, locs=locs, **kargs)
    if not inplace:
        return adata

