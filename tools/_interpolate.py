from scipy.ndimage import distance_transform_edt,binary_dilation,center_of_mass, label, center_of_mass
from scipy.interpolate import interpn

import matplotlib.pyplot as plt
import skimage as ski
import imageio.v3 as iio

import numpy as np
import h5py
import os

import io
import glob
import re
import pandas as pd

import multiprocessing
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()


def inter(images,t):
    #input: 
    # images: list of arrays/frames ordered according to motion
    # t: parameter ranging from 0 to 1 corresponding to first and last frame 
    #returns: interpolated image

    #direction of movement, assumed to be approx. linear 
    a=np.array(center_of_mass(images[0]))
    b=np.array(center_of_mass(images[-1]))

    #find index of two nearest frames 
    arr=np.array([center_of_mass(images[i]) for i in range(len(images))])
    v=a+t*(b-a) #convert t into vector 
    idx1 = (np.linalg.norm((arr - v),axis=1)).argmin()
    arr[idx1]=np.array([0,0]) #this is sloppy, should be changed if relevant values are near [0,0]
    idx2 = (np.linalg.norm((arr - v),axis=1)).argmin()

    if idx1>idx2:
        b=np.array(center_of_mass(images[idx1])) #center of mass of nearest contour
        a=np.array(center_of_mass(images[idx2])) #center of mass of second nearest contour
        tstar=np.linalg.norm(v-a)/np.linalg.norm(b-a) #define parameter ranging from 0 to 1 for interpolation between two nearest frames
        im1_shift=shift(images[idx2],(b-a)*tstar) #shift frame 1
        im2_shift=shift(images[idx1],-(b-a)*(1-tstar)) #shift frame 2
        return im1_shift+im2_shift #return average

    if idx1<idx2:
        b=np.array(center_of_mass(images[idx2]))
        a=np.array(center_of_mass(images[idx1]))
        tstar=np.linalg.norm(v-a)/np.linalg.norm(b-a)
        im1_shift=shift(images[idx2],-(b-a)*(1-tstar))
        im2_shift=shift(images[idx1],(b-a)*(tstar))
        return im1_shift+im2_shift

def binary_perim(bimg):
    return binary_dilation(bimg) - bimg
    # rows,cols = bimg.shape
    #
    # # Translate image by one pixel in all direct
    # # ions
    # n = 4 # or 8
    # north = np.zeros((rows,cols))
    # south = np.zeros((rows,cols))
    # west = np.zeros((rows,cols))
    # east = np.zeros((rows,cols))
    #
    # north[:-1,:] = bimg[1:,:]
    # south[1:,:]  = bimg[:-1,:]
    # west[:,:-1]  = bimg[:,1:]
    # east[:,1:]   = bimg[:,:-1]
    # idx = (north == bimg) & \
    #       (south == bimg) & \
    #       (west  == bimg) & \
    #       (east  == bimg)

    # if n == 8:
    #     north_east = np.zeros((rows, cols))
    #     north_west = np.zeros((rows, cols))
    #     south_east = np.zeros((rows, cols))
    #     south_west = np.zeros((rows, cols))
    #     north_east[:-1, 1:]   = bimg[1:, :-1]
    #     north_west[:-1, :-1]  = bimg[1:, 1:]
    #     south_east[1:, 1:]    = bimg[:-1, :-1]
    #     south_west[1:, :-1]   = bimg[:-1, 1:]
    #     idx &= (north_east == bimg) & \
    #            (south_east == bimg) & \
    #            (south_west == bimg) & \
    #            (north_west == bimg)
    # return ~idx * bimg

def binary_mask(bimg):
    img_bp = binary_perim(bimg)
    img_edist = distance_transform_edt(1-img_bp)
    imgm = img_edist*bimg -img_edist * np.logical_not(bimg)
    return imgm

def hybpara(alist, funct,
            
            args=(),kwds={},
            backend='loky',
             verbose=10,
            n_jobs=None):
    n_jobs = 1 if n_jobs is None else n_jobs 
    ielm =''
    if backend=='Threading':
        with ThreadPool(processes=n_jobs) as pool:
            result = [ pool.apply_async(funct, args=(*args,), kwds=kwds) for elm in tqdm(alist) ]
            pool.close()
            pool.join()
            result = [ar.get() for ar in result]

    elif backend=='Multiprocessing':
        with Pool(processes=n_jobs) as pool:
            result = [ pool.apply_async(funct, args=(*args,), kwds=kwds) for elm in tqdm(alist) ]
            pool.close()
            pool.join()
        result = [ar.get() for ar in result]

    else:
        result = Parallel(n_jobs= n_jobs, backend=backend, verbose=verbose)\
                     (delayed(funct)(*args, **kwds) for ielm in tqdm(alist))
    return result

def interp_image(images, points, position, method="linear", bounds_error=False, fill_value=0):
    intimg = interpn(points, images, position, 
                     method=method, 
                     bounds_error=bounds_error, 
                     fill_value=fill_value)
    intimg = intimg.reshape((-1, len(points[1]), len(points[2])))
    return intimg

def interpote_zstack(images, zdist=1, 
                        interpv=0.5, 
                        block=None, 
                        n_jobs=3,
                        method="linear",
                        bounds_error=False,
                        backend='loky',
                        verbose=2,
                        fill_value=0):
    """
    Interpolate zstack
    images : [z, x, y] or [z, x, y, c] 

    """
    images = np.array(images).copy()

    if images.ndim == 3:
        images = images[:,:,:,np.newaxis]
    if images.ndim == 4:
        thick, high, width, chanel = images.shape
    else:
        raise ValueError('images must be 3 or 4 dims.')

    if isinstance(zdist, (float, int)):
         zticks = np.arange(thick) * zdist
    elif isinstance(zdist, (list, np.ndarray)):
        if len(zdist) != thick: 
            raise ValueError('zdist must have same length as images.shape[0]')
        zticks = np.arange(zdist)
    else:
        zticks = np.arange(thick) * 1

    if isinstance(interpv, (float)) and (interpv<=1):
        z_interp = zticks * interpv
    elif isinstance(interpv, (int)) and (interpv>=1):
        zalls = (thick-1)* int(interpv+1) +1
        z_interp = np.linspace(zticks.min(), zticks.max(), zalls)
    elif type(interpv, (list, np.ndarray)):
        z_interp = np.array(interpv)
    else:
        z_interp = zticks * 0.5

    zalls = len(z_interp)
    block = zalls if block is None else block

    points = (zticks, np.arange(high), np.arange(width))
    posit = np.rollaxis(np.mgrid[:high, :width], 0, 3).reshape((high*width, 2))
    positions = np.c_[np.repeat(z_interp, high*width), np.tile(posit, (zalls,1))]
    positions = np.array_split(positions, block)

    volumns = []
    for ic in tqdm(range(chanel)):
        if n_jobs ==1:
            volumn = images[..., ic]
            for ipos in tqdm(range(block)):
                volumn= interp_image(volumn, 
                                      points, positions[ipos],
                                        method=method,
                                        bounds_error=bounds_error,
                                        fill_value=fill_value)
            #volumn.append(ivolumn)
        else:
            volumn = Parallel(n_jobs= n_jobs, 
                                backend=backend, 
                                verbose=verbose)\
                        (delayed(interp_image)
                            (images[..., ic], points, positions[ipos],
                                method=method,
                                bounds_error=bounds_error,
                                fill_value=fill_value)
                        for ipos in tqdm(range(block)))
            volumn = np.vstack(volumn)
        volumns.append(volumn)
    if chanel==1:
        volumns = volumns[0]
    else:
        volumns = np.moveaxis(volumns, 0, -1)

    if (volumns.max()>1) & np.issubdtype(images.dtype, np.integer):
        volumns = np.clip(np.round(volumns, 0), 0,255).astype(images.dtype)
    return volumns

def loc2mask(locus, size, axes=[2,0,1], dtype=np.uint8):
    img = np.zeros(size, dtype=dtype)
    if locus.shape[1] - len(size) >= 1:
        values = locus[:,len(size)]
    elif locus.shape[1] - len(size) == 0:
        values = np.ones(locus.shape[0])

    pos = np.round(locus[:,:len(size)]).astype(dtype)
    print(pos.shape, img.shape, pos.max(0))
    img[tuple(pos.T)] = values
    if not axes is None:
        img = np.transpose(img, axes=axes)
    return img

def explot_pixel(iimg, diameter=20):
    iimg_b = iimg.copy().astype(np.bool_)

    pix = np.round(diameter/2).astype(np.int32)
    strel = np.ones((pix, pix))
    iimg_b = binary_dilation(iimg_b, structure=strel).astype(np.int32)
    return iimg_b

def img2pos(img, thred=0):
    pos = np.where(img>thred)
    value = img[pos]
    return np.c_[(*pos, value)]