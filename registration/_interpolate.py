from scipy.ndimage import distance_transform_edt,binary_dilation,center_of_mass
from scipy.interpolate import interpn,RegularGridInterpolator
import matplotlib.pyplot as plt
# import imageio.v3 as iio
import numpy as np
import skimage as ski
from tqdm import tqdm

import multiprocessing
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage.interpolation import shift
from typing import Union

num_cores = multiprocessing.cpu_count()
'''
# https://stackoverflow.com/questions/47775621/interpolate-resize-3d-array
import skimage as ski
new_img = ski.transform.resize(imageS, (32, 2000, 2000, 3))
from scipy.ndimage import zoom
new_img = zoom(imageS, (2, 1, 1, 1))
new_img = ants.resample_image(img,spacing,False,1)

import torchio as tio
image = tio.ScalarImage(sFileName)
resample = tio.Resample(1)
resampled = resample(image)

import pyinterp
import pyinterp

mesh = pyinterp.RTree()


SIZE = 2000
X0, X1 = 80, 170
Y0, Y1 = -45, 30
lons = np.random.uniform(low=X0, high=X1, size=(SIZE, ))
lats = np.random.uniform(low=Y0, high=Y1, size=(SIZE, ))
data = np.random.uniform(low=-1.0, high=1.0, size=(SIZE, ))

mesh.packing(np.vstack((lons, lats)).T, data)

STEP = 1 / 32
mx, my = np.meshgrid(np.arange(X0, X1 + STEP, STEP),
                        np.arange(Y0, Y1 + STEP, STEP),
                        indexing='ij')

idw, neighbors = mesh.inverse_distance_weighting(
    numpy.vstack((mx.ravel(), my.ravel())).T,
    within=False,  # Extrapolation is forbidden
    k=11,  # We are looking for at most 11 neighbors
    num_threads=0)
idw = idw.reshape(mx.shape)
'''

def ski_resize(images, *args, **kargs):
    return ski.transform.resize(images, *args, **kargs)

def interpotez_3d(images, 
                 zdists=None, 
                 interpv=0.5, 
                 dtype=None,
                 interpd=None,
                 block=None, 
                 n_jobs=3,
                 method="linear",
                 datasize='middle',
                 bounds_error=False,
                 backend='loky',
                 verbose=2,
                 fill_value=0):
    """
    Interpolate 3d
    images : [z, x, y] or [z, x, y, c] 
    zdists : list
    """
    images = np.array(images).copy()

    if images.ndim == 3:
        images = images[:,:,:,np.newaxis]
    if images.ndim == 4:
        thick, high, width, chanel = images.shape
    else:
        raise ValueError('images must be 3 or 4 dims.')

    maxv = np.max(images)
    rdtype = images.dtype
    if maxv <=1 :
        dtype = images.dtype
        dvalue = 1
    elif 1 < maxv <= 255 :
        dtype = np.uint8
        dvalue = 255
    elif 255 < maxv <= 65535:
        dtype = np.uint16
        dvalue = 65535
    elif 65535 < maxv <= 4294967295:
        dtype = np.uint32
        dvalue = 4294967295
    else:
        dtype = images.dtype
        dvalue = images.max()

    if zdists is None:
        zticks = np.repeat(1, thick )
    else:
        if len(zdists) != thick:
            raise ValueError('zdists must have same length as images.shape[0]')
        zticks = np.arange(zdists)
    zticks = np.array(zticks)
    z_axes = np.cumsum(zticks) -zticks[0]
    x_axes = np.linspace(0, high-1, high)
    y_axes = np.linspace(0, width-1, width)
    c_axes = np.arange(chanel)

    if interpd is None:
        if isinstance(interpv, (float)) and (interpv<=1):
            z_space = np.array([0, interpv])
        elif isinstance(interpv, (int)) and (interpv>=1):
            z_space = np.linspace(0,1, interpv+1, endpoint=False)
        elif type(interpv, (list, np.ndarray)):
            z_space = np.array(interpv)
            assert np.all(z_space<1), 'each element of interpv list have be between 0 and 1.'
        else:
            print('Both interpd and interpv cannot be empty at the same time.')
            print('interpv will set to 0.5.')
            z_space = np.array([0, 0.5])
        z_interp = z_axes[:, np.newaxis] + zticks[:, np.newaxis] * z_space
        z_interp = np.unique(np.clip(z_interp, z_axes.min(), z_axes.max()).flatten())
    else:
        z_interp = np.array(interpd)
    x_interp = np.linspace(0, high-1, high)
    y_interp = np.linspace(0, width-1, width)
    c_interp = np.arange(chanel)

    if datasize == 'high':
        imagen = []
        for ich in tqdm(c_interp):
            rginterp = RegularGridInterpolator((z_axes, x_axes, y_axes), 
                                                images[...,ich], 
                                                method=method,
                                                bounds_error=bounds_error)
            iitps = []
            for iz in z_interp:
                zg, xg ,yg = np.meshgrid(iz, x_interp, y_interp, indexing='ij', sparse=False)
                positions = np.vstack([zg.ravel(), xg.ravel(), yg.ravel()]).T
                iitp = rginterp(positions)

                if np.issubdtype(dtype, np.integer):
                    iitp = np.round(iitp).astype(dtype)
                else:
                    iitp = iitp.astype(dtype)
                iitps.append(iitp)
            iitps = np.stack(iitps, axis=0).reshape(len(z_interp), len(x_interp), len(y_interp))
            imagen.append(iitps)
        imagen = np.stack(imagen, axis=-1)
    elif datasize == 'middle':
        zg, xg ,yg = np.meshgrid(z_interp, x_interp, y_interp, indexing='ij', sparse=False)
        positions = np.vstack([zg.ravel(), xg.ravel(), yg.ravel()]).T
        imagen = []
        for ich in range(chanel):
            rginterp = RegularGridInterpolator((z_axes, x_axes, y_axes), 
                                            images[...,ich], 
                                            method=method,
                                            bounds_error=bounds_error)
            iitp = rginterp(positions)
            iitp= iitp.reshape(len(z_interp), len(x_interp), len(y_interp))
            if np.issubdtype(dtype, np.integer):
                iitp = np.round(iitp).astype(dtype)
            else:
                iitp = iitp.astype(dtype)
            imagen.append(iitp)
        imagen = np.stack(imagen, axis=-1)
    elif datasize == 'low':
        zg, xg ,yg, cg = np.meshgrid(z_interp, x_interp, y_interp, c_interp, indexing='ij', sparse=False)
        positions = np.vstack([zg.ravel(), xg.ravel(), yg.ravel(), cg.ravel()]).T
        rginterp = RegularGridInterpolator((z_axes, x_axes, y_axes, c_axes), 
                                        images, 
                                        method=method,
                                        bounds_error=bounds_error)
        imagen = rginterp(positions)
        # imagen = interpn((z_axes, x_axes, y_axes, c_axes),
        #                                 images, 
        #                                 positions,
        #                                 method=method,
        #                                 bounds_error=bounds_error,)
        imagen = imagen.reshape(len(z_interp), len(x_interp), len(y_interp), len(c_interp))
        if np.issubdtype(dtype, np.integer):
            imagen = np.round(imagen).astype(dtype)
        else:
            imagen = imagen.astype(dtype)
    return imagen

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

    # volumns = []
    # for ic in tqdm(range(chanel)):
    #     if n_jobs ==1:
    #         volumn = images[..., ic]
    #         for ipos in tqdm(range(block)):
    #             volumn= interp_image(volumn, 
    #                                   points, positions[ipos],
    #                                     method=method,
    #                                     bounds_error=bounds_error,
    #                                     fill_value=fill_value)
    #         #volumn.append(ivolumn)
    #     else:
    #         volumn = Parallel(n_jobs= n_jobs, 
    #                             backend=backend, 
    #                             verbose=verbose)\
    #                     (delayed(interp_image)
    #                         (images[..., ic], points, positions[ipos],
    #                             method=method,
    #                             bounds_error=bounds_error,
    #                             fill_value=fill_value)
    #                     for ipos in tqdm(range(block)))
    #         volumn = np.vstack(volumn)
    #     volumns.append(volumn)
    # if chanel==1:
    #     volumns = volumns[0]
    # else:
    #     volumns = np.moveaxis(volumns, 0, -1)

    # if (volumns.max()>1) & np.issubdtype(images.dtype, np.integer):
    #     volumns = np.clip(np.round(volumns, 0), 0,255).astype(images.dtype)
    # return volumns

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

def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    # if n not in (4,8):
    #     raise ValueError('mahotas.bwperim: n must be 4 or 8')
    # print(bw.shape)
    # rows,cols = bw.shape
    #
    # # Translate image by one pixel in all direct
    # # ions
    # north = np.zeros((rows,cols))
    # south = np.zeros((rows,cols))
    # west = np.zeros((rows,cols))
    # east = np.zeros((rows,cols))
    #
    # north[:-1,:] = bw[1:,:]
    # south[1:,:]  = bw[:-1,:]
    # west[:,:-1]  = bw[:,1:]
    # east[:,1:]   = bw[:,:-1]
    # idx = (north == bw) & \
    #       (south == bw) & \
    #       (west  == bw) & \
    #       (east  == bw)
    # plt.imshow(idx.astype(int))
    # plt.title('move')
    # plt.show()
    #
    # if n == 8:
    #     north_east = np.zeros((rows, cols))
    #     north_west = np.zeros((rows, cols))
    #     south_east = np.zeros((rows, cols))
    #     south_west = np.zeros((rows, cols))
    #     north_east[:-1, 1:]   = bw[1:, :-1]
    #     north_west[:-1, :-1]  = bw[1:, 1:]
    #     south_east[1:, 1:]    = bw[:-1, :-1]
    #     south_west[1:, :-1]   = bw[:-1, 1:]
    #     idx &= (north_east == bw) & \
    #            (south_east == bw) & \
    #            (south_west == bw) & \
    #            (north_west == bw)
    # return ~idx * bw
    return binary_dilation(bw) - bw

def bwdist(im):
    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im

def signed_bwdist(im):
    '''
    Find perim and return masked image (signed/reversed)
    '''
    imbp = bwperim(im)
    imbd = bwdist(imbp)
    ima = -imbd * np.logical_not(im) + imbd*im
    return ima

def interp_shape(top, bottom, precision, use_bw=False):
    '''
    Interpolate between two contours

    Input: top
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if precision>2:
        print("Error: Precision must be between 0 and 1 (float)")
    top_and_bottom = np.stack((top, bottom))

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1 + precision

    if use_bw:
        top = signed_bwdist(top)
        bottom = signed_bwdist(bottom)
        r, c = top.shape
        # rejoin top, bottom into a single array of shape (2, r, c)
        top_and_bottom = np.stack((top, bottom))


    # create ndgrids
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))
    xi = np.c_[np.full((r*c), precision), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    #out = out > 0

    return out

def morphological_interpolation(
    labeled_stack: np.ndarray, stack_depth_spacing: Union[float, int]
):
    import itk
    # # python -m pip install itk-morphologicalcontourinterpolation
    # itk.morphological_contour_interpolator
    # python -m pip uninstall itk itk-core itk-filtering ... <other ITK modules here>
    # python -m pip install itk==v5.3rc3 itk-morphologicalcontourinterpolation

    spacing_array = np.zeros(
        (round(stack_depth_spacing), *labeled_stack.shape[1:]),
        dtype=labeled_stack.dtype,
    )
    spaced_stack = []
    for xy_plane in labeled_stack:
        spaced_stack.append(np.expand_dims(xy_plane, axis=0))
        spaced_stack.append(spacing_array)
    spaced_stack.pop(-1)
    spaced_stack = np.concatenate(spaced_stack)
    spaced_stack.astype(labeled_stack.dtype, copy=False)

    return itk.GetArrayFromImage(
        itk.morphological_contour_interpolator(itk.GetImageFromArray(spaced_stack))
    )

def test1():
    # Run interpolation
    images = iio.imread('D:/04GNNST/02Analysis/02LM_Visium/stak.reg.rgb.2000.2000.tif')
    image_1 = images[0]
    image_2 = images[-1]
    print(image_1.shape, image_2.shape)

    image_1 = sk.color.rgb2gray(image_1)
    image_2 = sk.color.rgb2gray(image_2)
    image_1 = sk.transform.resize(image_1, (1000, 1000))
    image_2 = sk.transform.resize(image_2, (1000, 1000))

    image_1 = (image_1>0).astype(int)
    image_2 = (image_2>0).astype(int)

    # image_1 = np.clip(image_1*255, 0, 255).astype(np.uint8)
    # image_2 = np.clip(image_2*255, 0, 255).astype(np.uint8)

    print(image_1.shape, image_2.shape)

    out = interp_shape(image_1,image_2, -1, use_bw=True)

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(image_1)
    ax[1].imshow(image_2)
    ax[2].imshow((out>0).astype(int))
    plt.show()
