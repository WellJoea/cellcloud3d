import numpy as np
import imageio.v3 as iio
import pandas as pd
from scipy.sparse import issparse

def list_get(l, idx, default):
    try:
        return l[idx]
    except:
        return default

def checksymmetric(adjMat, rtol=1e-05, atol=1e-08):
    if issparse(adjMat):
        #from scipy.linalg import issymmetric
        #return issymmetric(adjMat.astype(np.float64), atol=atol, rtol=rtol)
        adjMat = adjMat.toarray()
    return np.allclose(adjMat, adjMat.T, rtol=rtol, atol=atol)

def isidentity(M, equal_nan=True):
    if (M.shape[0] == M.shape[1]) and \
        np.allclose(M, np.eye(M.shape[0]), equal_nan=equal_nan):
        return True
    else:
        return False

def transsymmetric(mtx):
    if not checksymmetric(mtx):
        return (mtx + mtx.T)/2
    else:
        return mtx

def take_data(array, index, axis):
    sl = [slice(None)] * array.ndim
    sl[axis] = index
    return array[tuple(sl)]

def img2pos(img, thred=0):
    pos = np.where(img>thred)
    value = img[pos]
    return np.c_[(*pos, value)]

def loc2mask(locus, size, axes=[2,0,1]):
    img = np.zeros(size, dtype=np.int64)
    if locus.shape[1] - len(size) >= 1:
        values = locus[:,len(size)]
    elif locus.shape[1] - len(size) == 0:
        values = np.ones(locus.shape[0])

    pos = np.round(locus[:,:len(size)]).astype(np.int64)
    print(pos.shape, img.shape, pos.max(0))
    img[tuple(pos.T)] = values
    if not axes is None:
        img = np.transpose(img, axes=axes)
    return img

def sort_array(array, ascending=False):
    try:
        from scipy.sparse import csr_matrix as csr_array
    except:
        from scipy.sparse import csr_array
    arr_sp = csr_array(array)
    arr_dt = arr_sp.data
    arr_rc = arr_sp.nonzero()
    arr_st = np.vstack([arr_rc, arr_dt]).T
    if ascending:
        arr_st = arr_st[arr_st[:,2].argsort()]
    else:
        arr_st = arr_st[arr_st[:,2].argsort()[::-1]]
    return arr_st

def Info(sitkimg):
    print('***************INFO***************')
    print(f"origin: {sitkimg.GetOrigin()}")
    try:
        print(f"size: {sitkimg.GetSize()}")
    except:
        pass
    print(f"spacing: {sitkimg.GetSpacing()}")
    print(f"direction: {sitkimg.GetDirection()}")
    try:
        print( f"dimension: {sitkimg.GetDimension()}" )
    except:
        print( f"dimension: {sitkimg.GetImageDimension()}" )
    try:
        print( f"width: {sitkimg.GetWidth()}" )
        print( f"height: {sitkimg.GetHeight()}" )
        print( f"depth: {sitkimg.GetDepth()}" )
        print( f"pixelid value: {sitkimg.GetPixelIDValue()}" )
        print( f"pixelid type: {sitkimg.GetPixelIDTypeAsString()}" )
    except:
        pass
    print( f"number of components per pixel: {sitkimg.GetNumberOfComponentsPerPixel()}" )
    print('*'*34)

def spimage(adata, file=None, image=None, img_key="hires", basis = None, rescale=None,
           library_id=None, **kargs):
    if not file is None:
        image = iio.imread(file, **kargs)
    
    basis = 'spatial' if basis is None else basis
    library_id = 'slice0' if (library_id is None) else library_id
    
    rescale = 1 if rescale is None else rescale
    if rescale != 1:
        import cv2
        rsize = np.ceil(np.array(image.shape[:2])*rescale)[::-1].astype(np.int32)
        if image.ndim==3:
            image = cv2.resize(image[:,:,::-1], rsize, interpolation= cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, rsize, interpolation= cv2.INTER_LINEAR)

    if (basis in adata.uns.keys()) and (library_id in adata.uns[basis].keys()):
        adata.uns[basis][library_id]['images'][img_key] = image
        adata.uns[basis][library_id]['scalefactors'][f'tissue_{img_key}_scalef'] = rescale
    else:
        img_dict ={
            'images':{img_key: image},
            #unnormalized.radius <- scale.factors$fiducial_diameter_fullres * scale.factors$tissue_lowres_scalef
            #spot.radius <-  unnormalized.radius / max(dim(x = image))
            'scalefactors': {'spot_diameter_fullres': 1, ##??
                             'fiducial_diameter_fullres': 1 ,
                             f'tissue_{img_key}_scalef':rescale,
                             'spot.radius':1, 
                            },
            'metadata': {'chemistry_description': 'custom',
                           'spot.radius':  1, 
                          }
        }
        adata.uns[basis] = {library_id: img_dict}
