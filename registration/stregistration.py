import pandas as pd
import numpy as np
import scanpy as sc

import sys
import imageio
import rembg 
from pystackreg import StackReg

import matplotlib
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage as ski
from PIL import Image
import dipy

import cv2
import itk
# from itkwidgets import view
# from itkwidgets import view, compare, checkerboard
# from itkwidgets.widget_viewer import Viewer
# import ipywidgets
# from ipywidgets import Button, Label, HBox, VBox

from .plotting import qview

def get_force_peak(img, layout='rgb', figsize=(20,5), 
                   logyscale=True,
                   bin = None, iterval=None, show=True,):
    #from scipy.signal import find_peaks
    #find_peaks(counts, distance =10, width =3)
    peaks= []

    iimg = img.copy()
    if len(iimg.shape) ==2:
        iimg = iimg[:,:,np.newaxis]
    if np.round(iimg.max())<=1:
        iterval=(0, 1)
        bins=100
        xtick = np.round(np.linspace(0,1,bins+1, endpoint=True), 2)
    else:
        iterval=(0, 255)
        bins=255
        xtick = list(range(0,256,5))
    iimg = iimg[:,:,:]

    fig, ax = plt.subplots(1,1, figsize=figsize)
    for i in range(iimg.shape[2]):
        x = iimg[:,:,i].flatten()
        counts, values=np.histogram(x, bins=bins, range=iterval)
        max_value = int(values[np.argmax(counts)])
        peaks.append(max_value)
        xrange = np.array([values[:-1], values[1:]]).mean(0)
        ax.plot(xrange, counts, label=f"{i} {layout[i]} {max_value}", color=layout[i])
        ax.axvline(x=max_value, color=layout[i], linestyle='-.')

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
    return np.array(peaks)

def pixfullr(iimg, thred=10, fill=255, show=True):
    img_np = iimg.copy()
    idx = np.all(img_np>255-thred, axis=-1)
    img_np[idx,] = [fill, fill, fill]
    
    img_rr = img_np.copy()
    rvf = [255-fill, 255-fill, 255-fill]
    img_rr[np.all(img_rr==fill, axis=-1)] = rvf

    img_rk = img_np.copy()
    img_rk[ img_rk != iimg ] = 0
    if show:
        fig, axs = plt.subplots(1,2,figsize=(8,16))
        axs[0].imshow(img_rk)
        axs[1].imshow(img_rr)
        fig.show()
    return img_np


def fill_clip(clip):
    if isinstance(clip, int):
        clip = [clip]*3
    elif isinstance(clip, float):
        clip = [int(clip)]*3
    elif isinstance(clip, list):
        clip = clip[:3]
    return np.int64(clip[:3])

def pixfull0(iimg,
            fore_clip=None,
            fore_error=0,
            back_clip=0,
            bgcolor=(0, 0, 0),
            layout='rgb',
            show_peak = True,
            figsize_peak=(20,5),
            figsize=(10,20),
            show=True):
    
    fore_peak= get_force_peak(iimg, layout=layout, figsize=figsize_peak, show=show_peak)
    if fore_clip is None:
        fore_clip= fore_peak

    fore_clip = fill_clip(fore_clip)
    fore_error= fill_clip(fore_error)
    back_clip = fill_clip(back_clip)
    bgcolor   = fill_clip(bgcolor)

    fore_clip = fore_clip - fore_error

    img_np = iimg.copy()
    fore_idx = np.all(img_np>=fore_clip, axis=-1)
    back_idx = np.all(img_np<=back_clip, axis=-1)
    idx = (fore_idx | back_idx)
    img_np[idx,] = bgcolor

    #img_rk = iimg.copy()
    #img_rk[idx,] = 255
    #img_rk[idx,] = (100,200,255)
    if show:
        fig, axs = plt.subplots(1,2,figsize=figsize)
        axs[0].imshow(iimg)
        axs[1].imshow(img_np)
        fig.show()
    return img_np

def pixfull(iimg,
            clips=None,
            error=None,
            bgcolor=(0, 0, 0),
            layout='rgb',
            show_peak = True,
            figsize_peak=(20,5),
            figsize=(10,20),
            show=True):
    
    fore_peak= get_force_peak(iimg.copy(), layout=layout, figsize=figsize_peak, show=show_peak)
    if clips is None:
        clips = [fore_peak]
    if error is None:
        error = [[18]*6]
    
    img_np = iimg.copy()
    mask = np.zeros(img_np.shape[:2], dtype=bool)
    for iclip,ierr in zip(clips,error):
        iclip = np.array(iclip)
        ierr = np.array(ierr)  
        bclip = iclip - ierr[:3]
        fclip = iclip + ierr[3:6]

        bidx = np.all(img_np>=bclip, axis=-1)
        fidx = np.all(img_np<=fclip, axis=-1)
        mask = (mask | (bidx & fidx))
    img_np[mask,] = bgcolor

    img_rk = img_np.copy()
    img_rk[ img_rk != iimg ] = 0
    #img_rk[idx,] = 255
    #img_rk[idx,] = (100,200,255)
    if show:
        fig, axs = plt.subplots(1,2,figsize=figsize)
        axs[0].imshow(mask.astype(np.int64), cmap='gray')
        #axs[0].imshow(img_rk)
        axs[1].imshow(img_np)
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        fig.show()
    return img_np, mask

def cv2show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def RotPic(iimg, ort=-180, scale=1):
    iimg = iimg.copy()
    h,w = iimg.shape[:2]
    center = (w//2, h//2)
    M_1 = cv2.getRotationMatrix2D(center, ort, scale)
    iimg_r = cv2.warpAffine(iimg, M_1, (w, h))
    return iimg_r

from scipy.ndimage.morphology import binary_dilation
def get_img_spatial(adata, img_key="hires", basis = None, 
                    library_id=None, get_pix_loc=True, scale_img=False):
    
    basis = 'spatial' if basis is None else basis
    library_id = list(adata.uns[basis].keys())[0] if (library_id is None) else library_id
    iimg = adata.uns[basis][library_id]['images'][img_key]
    scale_factor = adata.uns[basis][library_id]['scalefactors'][f'tissue_{img_key}_scalef']
    
    if scale_img:
        rsize = np.ceil(iimg.shape[:2]/scale_factor)[::-1].astype(np.int32)
        iimg = cv2.resize(iimg.copy(), rsize, interpolation= cv2.INTER_LINEAR)
        st_loc = np.round(adata.obsm[basis], decimals=0).astype(np.int32)
    else:
        st_loc = np.round(adata.obsm[basis] * scale_factor, decimals=0).astype(np.int32)

    iimg = np.round(iimg*255)
    iimg = np.clip(iimg, 0, 255).astype(np.uint8)
    st_img = np.zeros(iimg.shape[:2], dtype=bool)
    st_img[st_loc[:,1], st_loc[:,0]] = True
    st_loc = st_img.copy().astype(np.int32)

    if get_pix_loc:
        spot_diameter_fullres = adata.uns[basis][library_id]['scalefactors']['spot_diameter_fullres']
        pix = np.round(spot_diameter_fullres*scale_factor/2).astype(np.int32)
        strel = np.ones((pix, pix))
        st_img = binary_dilation(st_img, structure=strel).astype(np.int32)
    else:
        st_img = st_loc
    return [iimg, st_loc, st_img]

from matplotlib.patches import Rectangle
def get_topn_index(array, n=10):
    arr_f = array.flatten()
    arr_s = arr_f.argsort().argsort()
    arr_f = arr_f>= np.sort(arr_f)[-n]
    arr_s = arr_s[arr_f] .argsort()
    arr_f = np.vstack(np.where(arr_f.reshape(array.shape)))
    return arr_f, arr_s

def matchFlame(img, temp, 
               show_method=True,
               top_n=1,
               sel_n=None,
               methods = ['cv.TM_CCOEFF_NORMED','cv.TM_CCORR_NORMED','cv.TM_SQDIFF_NORMED'] ):
    methodsr = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    methods = methodsr if methods is None else methods
    img = img.copy()
    #img = np.transpose(img, axes=[1,0,-1])
    temp = temp.copy()
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(temp.copy(), cv2.COLOR_BGR2GRAY)
   
    w, h = temp.shape[:2][::-1]
    recdict = {}

    if show_method:
        F, AX = plt.subplots(1,len(methods), figsize=(5*(len(methods)), 5))
    for i, meth in enumerate(methods):
        method = eval(meth)

        # Apply temp Matching
        #from skimage.feature import match_template
        #result = match_template(image, coin)
        res = cv2.matchTemplate(img_gray, temp_gray, method)
        if top_n >1:
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left, top_idx = get_topn_index(-res, n=top_n)
            else:
                top_left, top_idx = get_topn_index(res, n=top_n)
            top_left = top_left[::-1,]
        else:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            top_left= np.array([top_left]).T
            top_idx = np.array([0])
        
        if not sel_n is None:
            sidx = np.isin(top_idx, sel_n)
            top_idx = top_idx[sidx]
            top_left= top_left[:,sidx]
            print(top_idx, top_left)
        
        bottom_right = (top_left[0,] + w, top_left[1,] + h)
        #rect=cv.rectangle(img,top_left, bottom_right, (0,255,0),30)
        recdict[meth] = [top_left, top_idx, bottom_right]
        if show_method:
            AX[i].imshow(img,cmap = 'gray')

            AX[i].scatter(top_left[0,],top_left[1,], color='red', s=30)
            for _n in range(len(top_idx)):
                _xy= top_left[:,_n]
                _t  = top_idx[_n]
                AX[i].add_patch(Rectangle(_xy, w, h, edgecolor="green", facecolor='none', linewidth=2))
                AX[i].annotate(_t, _xy)
                    
            AX[i].set_axis_off()
            AX[i].set_title(f'{meth}')
    if show_method:
        F.show()

    return recdict

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

def plt_match(img, recdict, figscale = 5, show=True):
    from matplotlib.patches import Rectangle
    n_rec = len(recdict)
    fig, axes = plt.subplots(1,n_rec, figsize=(figscale*n_rec, figscale))
    for n, (k, pos) in enumerate(recdict.items()):
        ax = axes[n] if n_rec>1 else axes
        ax.imshow(img,cmap = 'gray')

        ax.scatter(pos[:,0],pos[:,1], color='red', s=30)
        ax.scatter(pos[:,0] +pos[:,2] ,pos[:,1] +pos[:,3], color='blue', s=30)
        for n in range(pos.shape[0]):
            ipos = pos[n,:]
            xy= ipos[:2]
            w = ipos[2]
            h = ipos[3]
            ax.add_patch(Rectangle(xy, w, h,
                                   edgecolor="green", facecolor='none', linewidth=2))
            ax.annotate(n, xy)

        ax.set_axis_off()
        ax.set_title(f'{k}')
    if show:
        fig.show()
    else:
        return (fig, axes)

def matchFlame(img, temp, 
               show_match=True,
               top_n=1,
               sel_n=None,
               methods = ['cv.TM_CCOEFF_NORMED','cv.TM_CCORR_NORMED','cv.TM_SQDIFF_NORMED'] ):
    methodsr = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    methods = methodsr if methods is None else methods
    img = img.copy()
    #img = np.transpose(img, axes=[1,0,-1])
    temp = temp.copy()
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(temp.copy(), cv2.COLOR_BGR2GRAY)
   
    w, h = temp.shape[:2][::-1]
    recdict = {}

    for i, meth in enumerate(methods):
        print(meth)
        method = eval(meth)
        # Apply temp Matching
        #from skimage.feature import match_template
        #result = match_template(image, coin)
        res = cv2.matchTemplate(img_gray, temp_gray, method)
        # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        #     top_left = min_loc
        # else:
        #     top_left = max_loc

        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            res_pos = sort_array(-res)
        else:
            res_pos = sort_array(res)

        res_top = res_pos[:,[1,0,2]]
        if not top_n is None:
            res_top = res_top[:top_n,:]
        if not sel_n is None:
            res_top = res_top[sel_n,:]
        res_top = np.c_[ res_top, np.repeat([[w, h]], res_top.shape[0], axis=0)][:,[0,1,3,4,2]]
        recdict[meth] = res_top
    
    if show_match:
        plt_match(img, recdict)

    return recdict

def get_corner_pos(res_pos, corner, method=None, use_glob=True):
    method = list(res_pos.keys()) if method is None else method
    corns = np.vstack([res_pos[imt] for imt in method])
    
    tlpos = corns[:,:2].copy()
    w, h = corns[0,2:4]

    if corner in ['00', 0, 'tl']:
        tpos, lpos = tlpos.min(0)
    elif corner in ['01',1, 'tr']:
        tpos, lpos = tlpos[:,0].max(), tlpos[:,1].min()
    elif corner in ['10',2,'bl']: 
        tpos, lpos = tlpos[:,0].min(), tlpos[:,1].max()
    elif corner in ['11',3, 'br']:
        tpos, lpos = tlpos[:,0].max(), tlpos[:,1].max()

    if not use_glob:
        xyidx = np.argmin(np.sum([(tlpos[:,0]-tpos)**2, (tlpos[:,1]-lpos)**2], axis=0))
        tpos, lpos = tlpos[xyidx,:2]

    if corner in ['00', 0,'tl']:
        xpos, ypos = tpos, lpos
    elif corner in ['01', 1, 'tr']:
        xpos, ypos = tpos + w, lpos
    elif corner in ['10', 2, 'bl']: 
        xpos, ypos = tpos, lpos + h
    elif corner in ['11', 3,'br']:
        xpos, ypos = tpos+w, lpos + h
    return [tpos, lpos, w, h, xpos, ypos]

def get_loc_img(adata, img_key="hires", basis = None, 
                    library_id=None, get_pix_loc=True, scale_img=False):
    basis = 'spatial' if basis is None else basis
    library_id = list(adata.uns[basis].keys())[0] if (library_id is None) else library_id
    iimg = adata.uns[basis][library_id]['images'][img_key]
    scale_factor = adata.uns[basis][library_id]['scalefactors'][f'tissue_{img_key}_scalef']
    
    if scale_img:
        rsize = np.ceil(iimg.shape[:2]/scale_factor)[::-1].astype(np.int32)
        iimg = cv2.resize(iimg.copy(), rsize, interpolation= cv2.INTER_LINEAR)
        st_loc = np.round(adata.obsm[basis], decimals=0).astype(np.int32)
    else:
        st_loc = np.round(adata.obsm[basis] * scale_factor, decimals=0).astype(np.int32)
    
    st_loc = np.c_[st_loc, np.arange(1, adata.shape[0]+1) ]
    iimg = np.round(iimg*255)
    iimg = np.clip(iimg, 0, 255).astype(np.uint8)
    st_img = np.zeros(iimg.shape[:2], dtype=np.int64)
    st_img[st_loc[:,1], st_loc[:,0]] = st_loc[:,2]
    return [iimg, st_loc, st_img]

def get_spatial_info(adata, img_key="hires", basis = None, 
                    library_id=None, get_pix_loc=False, rescale=None):

    adata = adata.copy()
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
        iimg = cv2.resize(iimg.copy(), rsize, interpolation= cv2.INTER_LINEAR)        
    locs = adata.obsm[basis] * scales
    
    st_loc = np.round(locs, decimals=0).astype(np.int32)
    iimg = np.round(iimg*255)
    iimg = np.clip(iimg, 0, 255).astype(np.uint8)
    st_img = np.zeros(iimg.shape[:2], dtype=bool)
    st_img[st_loc[:,1], st_loc[:,0]] = True

    if get_pix_loc:
        from scipy.ndimage.morphology import binary_dilation
        pix = np.round(spot_diameter_fullres*scale_factor/2).astype(np.int32)
        strel = np.ones((pix, pix))
        st_img = binary_dilation(st_img, structure=strel).astype(np.int32)
    else:
        st_img = st_loc
    return {"img":iimg, "locs":locs, 
            'loc_img':st_img,
            'scale_factor':scale_factor, 
            'spot_size':spot_diameter_fullres }

def resize_loc(img, sf_hw):
    raw_wh = img.shape[:2]
    new_wh = np.round(raw_wh * sf_hw).astype(np.int64)    
    new_img = np.zeros(new_wh, dtype=np.int64)

    locs = sort_array(img.copy(), ascending=True)
    locs[:,0] = locs[:,0] * sf_hw[0]
    locs[:,1] = locs[:,1] * sf_hw[1]
    
    new_img[locs[:,0], locs[:,1]] = locs[:,2]
    return(new_img)

def resize_loc(img, sf_hw):
    raw_wh = img.shape[:2]
    new_wh = np.round(raw_wh * sf_hw).astype(np.int64)    
    new_img = np.zeros(new_wh, dtype=np.int64)

    locs = sort_array(img.copy(), ascending=True)
    locs[:,0] = locs[:,0] * sf_hw[0]
    locs[:,1] = locs[:,1] * sf_hw[1]
    
    new_img[locs[:,0], locs[:,1]] = locs[:,2]
    return(new_img)

def resize_pos(locs, sf_hw):
    locs = locs.copy()
    locs[:,0] = locs[:,0] * sf_hw[0]
    locs[:,1] = locs[:,1] * sf_hw[1]
    return(locs)

from scipy.ndimage.morphology import binary_dilation
def explot_pixel(iimg, diameter=20):
    iimg_b = iimg.copy().astype(np.bool_)

    pix = np.round(diameter/2).astype(np.int32)
    strel = np.ones((pix, pix))
    iimg_b = binary_dilation(iimg_b, structure=strel).astype(np.int32)
    return iimg_b

def get_homo(img1, img2, togray=False,
             feature_method = 'sift', match_method='knn', 
             Filter=False,
             lines=2000,
             min_matches = 8,
             verify_ratio = 0.7, reprojThresh = 5.0):
    img1 = img1.copy()
    img2 = img2.copy()
    
    if Filter:
        #cv2.medianBlur(imageS[0], ksize=5).shape
        img1 = cv2.bilateralFilter(img1, 15,85,85 )
        img2 = cv2.bilateralFilter(img2, 15,85,85 )
    
    if togray:
        img1_g = cv2.cvtColor(img1
                              , cv2.COLOR_RGB2GRAY)
        img2_g = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_g = img1
        img2_g = img2

    if feature_method == 'sift':
        sift = cv.SIFT_create(nOctaveLayers =9, contrastThreshold=0.09, edgeThreshold=30, sigma =1.6)
        k1, d1 = sift.detectAndCompute(img1_g, None)
        k2, d2 = sift.detectAndCompute(img2_g, None)
        nomaltype = cv.NORM_L2
    elif feature_method == 'surf':
        minHessian = 400
        surf = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
        k1, d1 = surf.detectAndCompute(img1_g, None)
        k2, d2 = surf.detectAndCompute(img2_g, None)
        nomaltype = cv.NORM_L2

    elif feature_method == 'orb':
        orb = cv.ORB_create()
        k1, d1 = orb.detectAndCompute(img1_g, None)
        k2, d2 = orb.detectAndCompute(img2_g, None)
        nomaltype = cv.NORM_HAMMING

        
    if match_method=='cross':
        bf = cv2.BFMatcher( nomaltype, crossCheck=True) #cv2.NORM_HAMMING,
        matches = bf.match(d1, d2)
        top_match = sorted(matches, key=lambda x:x.distance)
        img3 = cv2.drawMatches(img1, k1,img2, k2, top_match[:lines], None, (0,0,255), flags=2)
        verify_matches = matches

    elif match_method=='knn':       
        bf = cv2.BFMatcher(nomaltype)
        matches = bf.knnMatch(d1, d2, k=2)
        #return img3

        verify_matches = []
        for m1, m2 in matches:
            if m1.distance < verify_ratio * m2.distance:
                verify_matches.append(m1)
        img3 = cv2.drawMatches(img1, k1, img2, k2, verify_matches[:lines], None, (0,255,0), flags=2)

    elif match_method=='flann':
        # FLANN parameters
        if feature_method == 'orb':
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                                 table_number = 6, # 12
                                 key_size = 12, # 20
                                 multi_probe_level = 1) #2
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(d1, d2,k=2)
        
        verify_matches = []
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m1,m2) in enumerate(matches):
            if m1.distance < verify_ratio * m2.distance:
                matchesMask[i]=[1,0]
                verify_matches.append(m1)

        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1, k1, img2, k2,matches,None,**draw_params)
        #img3 = cv2.drawMatches(img1, k1, img2, k2, verify_matches, None, (0,0,255), flags=2)

    print(len(verify_matches))
    assert len(verify_matches)> min_matches
    img1_pts = []
    img2_pts = []
    for m in verify_matches:
        img1_pts.append(k1[m.queryIdx].pt)
        img2_pts.append(k2[m.trainIdx].pt)

    img1_pts = np.array(img1_pts).astype(np.float64) #.reshape(-1,1,2)
    img2_pts = np.array(img2_pts).astype(np.float64) #.reshape(-1,1,2)
    H, msk = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, reprojThresh)
    return H, msk, img1_pts, img2_pts, img3,verify_matches


def opencv_registration(fixed_img, moving_img,
                         method = 'rigid',
                         match_method='knn', 
                         feature_method='sift',
                         verify_ratio=0.9,
                         out_shape=None,
                         togray=True, 
                         Filter=False,
                         clip=True,
                         preserve_range=False,
                         show=True):
    import numpy as np
    from skimage import transform as skitf
    if out_shape is None:
        out_shape = fixed_img.shape[:2]
    K = get_homo(moving_img.copy(), fixed_img.copy(),
                 match_method=match_method,
                 feature_method=feature_method,
                 verify_ratio=verify_ratio,
                 togray=togray, 
                 Filter=Filter)
    H, msk, mov_pts, fix_pts, img3, verify_matches  = K

    if method.lower() in ['rigid', 'euclidean']:
        #@tform = skitf.estimate_transform('euclidean', fix_pts, mov_pts)
        #cv2.warpAffine(img1, np.array(tform)[:2,],img1.shape[:2][::-1])
        tform = skitf.EuclideanTransform()
    elif method.lower() in ['similarity']:
        tform = skitf.SimilarityTransform()
    elif method.lower() in ['projective']:
        tform = skitf.ProjectiveTransform()
    elif method.lower() in ['affine']:
        tform = skitf.AffineTransform()

    tform.estimate(fix_pts,  mov_pts)
    mov_out = skitf.warp(moving_img, tform, 
                         output_shape=out_shape,
                         preserve_range = preserve_range,
                         clip=clip)
    if show:
        qview(fixed_img, moving_img, mov_out, img3, 
                ncols=3,
                titles = ['fixed_img', 'moving_img', 'mov_out', 'img_matches'])
    return [mov_out, tform]

def transform_points(locs, tform, inverse=False):
    new_locs = locs.copy()
    if inverse:
        new_locs[:,:2] =  tform.inverse(new_locs[:,:2])
        return  new_locs
    else:
        new_locs[:,2] = 1
        new_locs =  new_locs @ np.linalg.inv(tform).T
        new_locs[:,2] = locs[:,2]
        return  new_locs

def loc2img(locus, size):
    img = np.zeros(size, dtype=np.int64)
    img[np.round(locus[:,0]).astype(np.int64), 
        np.round(locus[:,1]).astype(np.int64), ] = locus[:,2]
    return img

import numpy as np
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk

def skiregistraion_tlv1(fixed_img, moving_img, 
                        togray=True, 
                        attachment=15,
                        tightness=0.3,
                        num_warp=5,
                        num_iter=10,
                        tol=0.0001,
                        prefilter=False,
                        **kargs):
    import numpy as np
    from skimage.transform import warp
    from skimage.registration import optical_flow_tvl1, optical_flow_ilk
    import skimage as ski

    if togray:
        fixed_img = ski.color.rgb2gray(fixed_img)
        moving_img = ski.color.rgb2gray(moving_img)

    V, U = optical_flow_tvl1(fixed_img.astype(np.float32), 
                            moving_img.astype(np.float32),
                            attachment=attachment, #5
                            tightness=tightness, 
                            num_warp=num_warp, #5
                            num_iter=num_iter,
                            tol=tol, #0.001
                            prefilter=prefilter)
    nr, nc = fixed_img.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), 
                                         np.arange(nc),
                                         indexing='ij')
    coords = np.array([row_coords + V, col_coords + U])
    return coords

def transform_oftvl1(image, coords, mode='edge', order=1, use_ski=True, **kargs):
    from skimage import transform as skitf
    from scipy.ndimage.interpolation import map_coordinates
    from skimage.color import rgb2gray
    if len(image.shape)>2:
        image = rgb2gray(image)

    if use_ski:
        image1_warp = skitf.warp(image, coords, mode=mode, order=order, **kargs)
    else:
        image1_warp = map_coordinates(image, coords, prefilter=False, 
                                      order=order, 
                                      mode='nearest', **kargs)
    return image1_warp

def transform_rgb_oftvl1(image, coords, mode='edge', order=1, use_ski=True, **kargs):
    imagergb = []
    for i in range(image.shape[2]):
        iimg = transform_oftvl1(image[:,:,i], 
                                    coords, 
                                    mode=mode,
                                    order=order, 
                                    use_ski=use_ski,
                                    **kargs)
        imagergb.append(iimg[:,:,np.newaxis])
    imagergb = np.concatenate(imagergb, axis=2)
    return(imagergb)


def transform_points_oftvl1(points, coords, scale_max = 10):
    points = points[:,[1,0,2]].copy()
    pointimg = np.arange(np.prod(coords[0].shape)).reshape(coords[0].shape)
    points = np.c_[points, pointimg[points[:,0], points[:,1]]]

    from scipy.ndimage.interpolation import map_coordinates
    imgwarp = map_coordinates(pointimg, coords,
                            prefilter=False,
                            mode='nearest',
                            order=0, cval=0.0)
    def get_cub_postion(point, scale):
        XX, YY = np.meshgrid(np.arange(point[0] - scale, point[0] + scale +1),
                            np.arange(point[1] - scale, point[1] + scale +1))
        return(XX, YY)

    pointsnew = []
    for i in range(points.shape[0]):
        iloc = points[i]
        scale = 0
        X, Y = [], []
        while (len(X)==0) and (scale<=scale_max):
            ipos = pointimg[get_cub_postion(iloc[:2], scale)].flatten()
            X, Y= np.where(np.isin(imgwarp, ipos))
            scale +=1
        if len(X)>0:
            X = np.mean(X)
            Y = np.mean(Y)
        else:
            X, Y = np.inf, np.inf
        if scale>5:
            print(i, scale)
        iloc = list(iloc) + [X, Y]
        pointsnew.append(iloc)
    pointsnew = np.array(pointsnew)
    return pointsnew[:,[5,4,2]]

def GetInfo(sitkimg):
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

def points2images(points, imgsize, mask=True):
    img = np.zeros(imgsize, dtype=np.int64)
    if mask:
        img[np.round(points[:,1]).astype(np.int64), 
            np.round(points[:,0]).astype(np.int64), ] = 1
        return img
    else:
        img[np.round(points[:,1]).astype(np.int64), 
            np.round(points[:,0]).astype(np.int64), ] = points[:,2]
        return img

def pixelimgmatch(pixelimg, image):
    Redb = matplotlib.colors.LinearSegmentedColormap.from_list("Redb", ['white', 'red'])
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.imshow(image)
    ax.imshow(explot_pixel(pixelimg), cmap=Redb, alpha=.8)
    fig.show()

def map_points_oftvl1(points, coords):
    points = points.copy()
    from scipy.ndimage.interpolation import map_coordinates
    hw = coords[0].shape[:2]
    imgpos = points2images(points, hw, mask=False)

    imgwarp = map_coordinates(imgpos, coords,
                            prefilter=False,
                            mode='nearest',
                            order=0, cval=0.0)
    pointn = sort_array(imgwarp, ascending=True)[:,[1,0,2]]
    return pointn

def transform_points_oftvl1_error(points, coords):
    points = points.copy()
    pointnew  = map_points_oftvl1(points, coords)
    miss_idx = np.where(~np.isin(points[:,2], pointnew[:,2]))[0]

    pos8 = np.array([[1,0], [0,1], [-1,0], [0,-1], [1,1],[-1,-1], [1,-1], [-1,1]])
    pos8 = np.c_[pos8, np.zeros(8)]
    istep = 1
    pointmiss = np.empty((0,3))
    while len(miss_idx)>0:
        print(len(miss_idx), istep)
        sift = pos8*istep
        print(sift)
        sift_points = np.concatenate([sift+points[idx] for idx in miss_idx], axis=0)
        sift_points = map_points_oftvl1(sift_points, coords)

        pointmiss = np.r_[pointmiss, sift_points]
        miss_idx = np.setdiff1d(miss_idx, pointmiss[:,2])
        istep +=1
    return pointnew
    #pointnew = np.append(pointnew, poinfind, axis=0)
    #idx_miss = np.setdiff1d(pointmiss[:,2], pointnew[:,2])
    #istep +=1

def transform_points_oftvl1(points, coords,  scale_max = 10):
    points = points.copy()
    pointimg = np.arange(np.prod(coords[0].shape)).reshape(coords[0].shape)
    points = np.c_[points, pointimg[points[:,0], points[:,1]]]

    from scipy.ndimage.interpolation import map_coordinates
    imgwarp = map_coordinates(pointimg, coords,
                            prefilter=False,
                            mode='nearest',
                            order=0, cval=0.0)

    def get_cub_postion(point, scale):
        XX, YY = np.meshgrid(np.arange(point[0] - scale, point[0] + scale +1),
                            np.arange(point[1] - scale, point[1] + scale +1))
        return(XX, YY)

    pointsnew = []
    scale = scale_max
    for i in range(points.shape[0]):
        iloc = points[i]
        scale = 0
        X, Y = [], []
        while (len(X)==0) and (scale<=trip):
            ipos = pointimg[get_cub_postion(iloc[:2], scale)].flatten()
            X,Y = np.where(np.isin(imgwarp, ipos))
            scale +=1
        if len(X)>0:
            X = np.mean(X)
            Y = np.mean(Y)
        else:
            X, Y = np.inf, np.inf
        if scale>5:
            print(i, scale)
        iloc = list(iloc) + [X, Y]
        pointsnew.append(iloc)
    pointsnew = np.array(pointsnew)
    return pointsnew[:,[3,4,2]]

def stackreg_3d(image, 
             trans_type=StackReg.RIGID_BODY, 
             reference='previous',
             **kargs):
    
    imageg = np.array([cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in image])
    
    sr = StackReg(trans_type)
    tmats = sr.register_stack(imageg, reference=reference, **kargs)
    #np.save('transformation_matrices.npy', tmats)
    #tmats = np.load('transformation_matrices.npy')

    #out_gray = sr.transform_stack(imageLg, tmats=tmats)
    out_gray = sr.transform_stack(imageg)

    image_rgb = np.copy(image)
    for color in range(image.shape[3]):
        image_rgb[:, :, :, color] = sr.transform_stack(image[:, :, :, color])
    return out_gray, image_rgb, tmats

def stack_3d(image, 
             refer_idx=None,
             trans_type=StackReg.RIGID_BODY, 
             reference='previous',
             **kargs):
    if refer_idx is None:
        refer_idx = int(image.shape[0]/2)
        
    bimgs = image[:(refer_idx+1),]
    fimgs = image[refer_idx:,]
    
    grays, rgbs, tmats = [], [], []
    if refer_idx > 0:
        bgray, brgb, btmat = stackreg_3d(bimgs[::-1,], 
                                          trans_type=trans_type, 
                                          reference=reference,
                                          **kargs)
        grays  = bgray[::-1,] 
        rgbs   = brgb[::-1,]
        tmats  = btmat[::-1,]
    if refer_idx < image.shape[0]:
        fgray, frgb, ftmat = stackreg_3d(fimgs, 
                                          trans_type=trans_type, 
                                          reference=reference,
                                          **kargs)
        grays = fgray if grays == [] else np.vstack([grays[:-1,], fgray])
        rgbs  = frgb  if rgbs  == [] else np.vstack([rgbs[:-1,], frgb])
        tmats = ftmat if tmats == [] else np.vstack([tmats[:-1,], ftmat])
    return [grays, rgbs, tmats]

def transform_points(locs, tform, inverse=False):
    new_locs = locs.copy()
    if inverse:
        new_locs[:,:2] =  tform.inverse(new_locs[:,:2])
        return  new_locs
    else:
        new_locs[:,2] = 1
        new_locs =  new_locs @ np.linalg.inv(tform).T
        new_locs[:,2] = locs[:,2]
        return  new_locs

def show_images(img_ref, img_warp, fig_name):
    fig, axarr = plt.subplots(ncols=2, figsize=(12, 5))
    axarr[0].set_title('warped image & reference contour')
    axarr[0].imshow(img_warp, cmap='viridis')
    axarr[0].contour(img_ref, colors='r')
    ssd = np.sum((img_warp - img_ref) ** 2)
    axarr[1].set_title('difference, SSD=%.02f' % ssd)
    im = axarr[1].imshow(img_warp - img_ref,  cmap='viridis')
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(fig_name + '.png')

def registerParaObj(transforms=None, resolutions=None, GridSpacing=None, verb=False):
    parameters = itk.ParameterObject.New()

    transforms = ['rigid', 'bspline' ] if transforms is None else transforms
    resolutions = [4]*len(transforms) if resolutions is None else resolutions
    GridSpacing = [10]*len(transforms) if GridSpacing is None else GridSpacing
    for i, itran in enumerate(transforms):
        ires = resolutions[i]
        igrid= GridSpacing[i]
        if itran=='rigid':
            default_rigid = parameters.GetDefaultParameterMap("rigid", ires, igrid)
            parameters.AddParameterMap(default_rigid)

        if itran=='affine':
            default_affine = parameters.GetDefaultParameterMap("affine", ires, igrid)
            parameters.AddParameterMap(default_affine)

        if itran=='bspline':
            default_bspline = parameters.GetDefaultParameterMap("bspline", ires, igrid)
            parameters.AddParameterMap(default_bspline)

        if itran=='groupwise':
            groupwise_parameter_map = parameters.GetDefaultParameterMap('groupwise', ires, igrid)
            parameters.AddParameterMap(groupwise_parameter_map)

    parameters.RemoveParameter("ResultImageFormat")
    if verb:
        print(parameters)
    return parameters

def loc2img(locus, size, value=None):
    img = np.zeros(size, dtype=np.int64)
    value = locus[:,2] if value is None else 1
    img[np.round(locus[:,0]).astype(np.int64), 
        np.round(locus[:,1]).astype(np.int64), ] = value
    return img

def locwrite(locate, out):
    f = open(out, 'w')
    f.write('point\n')
    f.write(f'{locate.shape[0]}\n')
    for i in range(locate.shape[0]):
        iline = ' '.join(locate[i].astype(str))
        f.write(f'{iline}\n')
    f.close()

def load_sample(sid):
    import sys
    import importlib
    sys.path.append('/share/home/zhonw//JupyterCode')
    import SCFunc
    importlib.reload(SCFunc)
    from SCFunc import Preprocession
    
    data_path = '/share/home/zhonw//WorkSpace/11Project/04GNNST/01DataBase/01ZB_10xVisum_8'
    file = f'{data_path}/H5AD/{sid}.h5ad'
        
    adata = sc.read_h5ad(file)
    adata.obs_names = adata.obs_names+':'+sid

    adata.var_names_make_unique()
    #adata = Preprocession(adata).Normal() 
    #adata.layers['count'] = adata.raw.X.copy()
    #adata = sc.pp.subsample(adata, n_obs=8000, random_state=0, copy=True)
    print(adata.shape)
    
    adata.obs['SID'] = sid
    adata.obsm['X_spatial']=adata.obsm['spatial'].copy()   
    adata.obsm['spatialr']=adata.obsm['spatial'][:,::-1].copy()  
    adata.raw= adata
    return adata

def pointimgmatch(image, points,  s=1, color='red', alpha=1, **kargs):
    #import matplotlib
    #Redb = matplotlib.colors.LinearSegmentedColormap.from_list("Redb", ['white', 'red'])

    fig, ax = plt.subplots(1,1, figsize=(7,7))
    ax.imshow(image)
    ax.scatter(points[:,0], points[:,1], s=s, color=color, alpha=alpha, **kargs)
    fig.show()
