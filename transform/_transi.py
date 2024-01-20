import numpy as np
import skimage as ski

from ..utilis._arrays import isidentity
from joblib import Parallel, delayed

uint_scale = {
    np.float64:1.,
    np.float16:1.,
    np.float32:1.,
    np.uint8: 255,
    np.uint16: 65535,
    np.uint32: 4294967295,
    np.uint64: 18446744073709551615,
}

uint_mrange = {
    np.uint8: (0, 255),
    np.uint16: (255, 65535),
    np.uint32: (65535, 4294967295),
    np.uint64: (4294967295, 18446744073709551615),
}

def uint_value(dtype, maxv=None, verbose=True):
    if  np.issubdtype(dtype, np.floating):
        ttype = dtype
        dvalue = maxv or 1
    elif maxv <=1 :
        ttype = dtype
        dvalue = 1
    elif 1 < maxv <= 255 :
        ttype = np.uint8
        dvalue = 255
        if verbose and (not dtype in [np.uint8]):
            print("Warning: image dtype rescaled to 8-bit")
    elif 255 < maxv <= 65535:
        ttype = np.uint16
        dvalue = 65535
        if verbose and (not dtype in [np.uint16]):
            print("Warning: image dtype rescaled to 16-bit")
    elif 65535 < maxv <= 4294967295:
        ttype = np.uint32
        dvalue = 4294967295
        if verbose and (not dtype in [np.uint32]):
            print("Warning: image dtype rescaled to 32-bit")
    elif 4294967295 < maxv <= 18446744073709551615:
        ttype = np.uint64
        dvalue = 18446744073709551615
        if verbose and (not dtype in [np.uint64]):
            print("Warning: image dtype rescaled to 64-bit")
    else:
        ttype = dtype
        dvalue = maxv

    dvalue = maxv or dvalue
    return [ttype, dvalue]

def imageres(image, dtype, scale_max=None):
    scale_max = scale_max or uint_scale[dtype]
    return np.clip(np.round(image * scale_max), 0, scale_max).astype(dtype)

def imagerestrans(image, sdtype, tdtype):
    return imageres(image/uint_scale[sdtype], tdtype)

def rotateion(image, degree=np.pi, keeptype=True, **kargs):
    hw = image.shape[:2]
    shift = np.eye(3)
    shift[:2,2] = [hw[0]/2, hw[1]/2]
    tform = ski.transform.EuclideanTransform(rotation=degree).params.astype(np.float64)
    tform = shift @ tform @ np.linalg.inv(shift)
    tform = tform.astype(np.float64) 
    imagen = ski.transform.warp(image,  tform, **kargs)
    if keeptype and (not np.issubdtype(imagen.dtype, np.integer)) and np.issubdtype(image.dtype, np.integer) and (image.max()>1) :
        imagen = imageres(imagen, image.dtype.type)
    return [imagen, tform ]

def rescale_tmat(tmat, sf, trans_scale=True, dimension=2):
    scale_l  = np.eye(dimension+1)
    scale_l[range(dimension), range(dimension)] = sf
    scale_l[:dimension, dimension] = sf
    scale_r = np.eye(dimension+1)
    scale_r[range(dimension), range(dimension)] = 1/sf

    if trans_scale:
        return scale_l @ tmat @ scale_r
    else:
        return scale_l @ tmat
    
def resize(image, reshape, 
           order=None, 
           mode='reflect',
           cval=0, 
           clip=True, 
           method = 'skimage',
           keeptype=True,
           cv2interp = 3, **kargs):
    if method == 'skimage':
        imagen = ski.transform.resize(image,  
                                        reshape,
                                        order=order,
                                        cval=cval,
                                        mode=mode,
                                        clip=clip, **kargs)
    elif method == 'cv2':
        try:
            import cv2
        except:
            print('you can use cv2 by "pip install opencv-python", or switch method to "ski".')
        imagen = cv2.resize(image, reshape, interpolation= cv2interp )

    if keeptype and (not np.issubdtype(imagen.dtype, np.integer)) and np.issubdtype(image.dtype, np.integer) and (image.max()>1) :
        imagen = imageres(imagen, image.dtype.type)

    return imagen

def resizes(images, reshapes, 
           order=None, 
           mode='reflect',
           cval=0, 
           clip=True, 
           keeptype=True,
           method = 'skimage',
           cv2interp = 3, **kargs):
    if type(reshapes[0]) in [int]:
        reshapes = [reshapes] * len(images)
    assert len(images) == len(reshapes), 'the length between images and reshapes must be same.'
    imagen = [resize(images[i], reshapes[i],
                    order=order, 
                    mode=mode,
                    cval=cval, 
                    clip=clip, 
                    method = method,
                    keeptype=keeptype,
                    cv2interp = cv2interp, **kargs) for i in range(len(images))]
    if isinstance(images, np.ndarray):
        return np.array(imagen)
    else:
        return imagen

def homotransform(
        image,
        tmat,
        order=None,
        keeptype=True,
        trans_name = 'skimage',
        inverse=False,
        **kargs):

    if isidentity(tmat):
        return image

    if inverse:
        tmat = np.linalg.inv(tmat)

    if trans_name=='skimage':
        imagen = ski.transform.warp(image,  tmat, order=order, **kargs)

        if keeptype and (not np.issubdtype(imagen.dtype, np.integer)) and np.issubdtype(image.dtype, np.integer) and (image.max()>1) :
            imagen = imageres(imagen, image.dtype.type)

    return imagen 

def homotransforms( images,
                    tmats,
                    order=None,
                    keeptype=True,
                    trans_name = 'skimage',
                    n_jobs = 10,
                    backend="multiprocessing",
                    verbose=0,
                    **kargs):
    assert len(images) == len(tmats), 'the length between images and tmats must be same.'
    if n_jobs >1:
        imagen = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)\
                            (delayed(homotransform)(images[i], tmats[i],
                                    order=order,
                                    keeptype=keeptype,
                                    trans_name = trans_name,**kargs)
                            for i in range(len(images)))
    else:
        imagen = [homotransform(images[i], tmats[i],
                                order=order,
                                keeptype=keeptype,
                                trans_name = trans_name,
                                **kargs) for i  in range(len(images))]
    if isinstance(images, np.ndarray):
        return np.array(imagen)
    else:
        return imagen

def rescale(image, scale=None, method='skimage',
            order=None, 
            mode='reflect',
            cval=0, 
            clip=True, 
            keeptype=True,
            cv2interp = 3,
            **kargs):
    if (scale is None) or (scale==1):
        return image

    if type(scale) in [int, float]:
        scale = [scale, scale]
    reshape = np.round(np.array(image.shape[:2]) * np.array(scale), 0).astype(np.int64)
    retmat = np.eye(3)
    retmat[[0,1], [0,1]] = scale

    if method in ['homotrans']: # error
        imagen = homotransform(image,
                                retmat,
                                order=order,
                                keeptype=keeptype,
                                trans_name = 'skimage',
                                **kargs)

    else:
        #ski.tf.rescale
        imagen = resize(image, reshape,
                        order=order, 
                        mode=mode,
                        cval=cval, 
                        clip=clip, 
                        keeptype=keeptype,
                        method = method,
                        cv2interp = cv2interp,
                        **kargs)
    return imagen

def rescales(images, scales, 
           order=None, 
           mode='reflect',
           cval=0, 
           clip=True, 
           keeptype=True,
           method = 'skimage',
           cv2interp = 3, **kargs):

    if type(scales)  in [float, int]:
        scales = [scales] * len(images)
    assert len(images) == len(scales), 'the length between images and scales must be same.'
    imagen = [rescale(images[i], scales[i],
                    order=order, 
                    mode=mode,
                    cval=cval, 
                    clip=clip, 
                    method = method,
                    keeptype=keeptype,
                    cv2interp = cv2interp, **kargs) for i in range(len(images))]
    if isinstance(images, np.ndarray):
        return np.array(imagen)
    else:
        return imagen

def mirroraxis(array, points=None, x=False, y=False, z=False, axes=None):
    sl = [slice(None)] * array.ndim
    ts = []
    for i,k in enumerate([x,y,z]):
        if k:
            ts.append(i)
    if not axes is None:
        ts = [*ts, *axes]
    for its in ts:
        sl[its]=slice(None, None, -1)

    arrayn = array[tuple(sl)] if ts else array
    if points is None:
        return arrayn

    if isinstance(points, list) and (points[0].shape[1] == 2):
        pointns = []
        for ipos in points:
            ipos = ipos.copy()
            for its in ts:
                if its != 0:
                    ipos[:, 2-its] = array.shape[its] - ipos[:, 2-its]
            pointns.append(ipos)
        if 0 in ts:
            pointns = pointns[::-1]
    else:
        pointns = points.copy()
        for its in ts:
            pointns[:, its] = array.shape[its] - pointns[:, its]
    return arrayn, pointns

def padsize(img, pad_width=([30,30],[30,30]), constant_values= 0, mode ='constant', **kargs):
    return np.pad(img, pad_width , mode ='constant', constant_values=constant_values)
    # iimg = img.copy()
    # tp,bl = pad_width[0]
    # lf,rg = pad_width[1]
    
    # top   = np.zeros([tp] + list(iimg.shape[1:])) + constant_values
    # below = np.zeros([bl] + list(iimg.shape[1:])) + constant_values

    # iimg = np.concatenate([top, iimg, below], axis=0)
    # left  = np.zeros([iimg.shape[0], lf] + list(iimg.shape[2:])) + constant_values
    # right = np.zeros([iimg.shape[0], rg] + list(iimg.shape[2:])) + constant_values
    # iimg = np.concatenate([left, iimg, right], axis=1)
    # return iimg.astype(img.dtype)

def padcenter(imglist, hw=None, **kargs):
    if hw is None:
        max_hw = max([ max(i.shape[:2]) for i in imglist ])
        H, W = [max_hw,max_hw]
    else:
        H, W =hw[:2]
    imagen  = []
    tblr = []
    for img in imglist:
        h, w = img.shape[:2]
        tp = (H - h)//2
        bl = H - h - tp
        lf = (W - w)//2
        rg = W - w -lf
        pad_width = [(0,0)] * img.ndim
        pad_width[0] = (tp, bl)
        pad_width[1] = (lf, rg)
        iimg = padsize(img, pad_width=pad_width, **kargs)
        imagen.append(iimg)
        tblr.append([tp, bl, lf, rg])
    return [imagen,pad_width]