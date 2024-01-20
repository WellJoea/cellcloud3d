try:
    from pystackreg import StackReg
except:
    pass
import skimage as ski
from joblib import Parallel, delayed
import numpy as np

class turbostack():
    TRANS = {
        'translation': StackReg.TRANSLATION,
        'rigid': StackReg.RIGID_BODY,
        'silimarity': StackReg.SCALED_ROTATION,
        'affine': StackReg.AFFINE,
        'bilinear': StackReg.BILINEAR,
    }
    def __init__(self, transtype = None, tmats=None):
        self.transtype = 'rigid' if transtype is None else transtype
        self.sr = StackReg(self.TRANS[self.transtype])
        self.tmats = tmats 
    
    def regist_stack(self, image, 
                 reference='previous',
                 **kargs):
        tmats = self.sr.register_stack(image, reference=reference, **kargs)
        return tmats

    def regist(self,
                     images, 
                     isscale=None,
                     refer_idx=None,
                     reference='previous',
                     **kargs):
        if refer_idx is None:
            refer_idx = int(images.shape[0]/2)
        refer_idx = int(min(refer_idx, images.shape[0]-1))

        isscale = isscale or self.onechannel(images)
        if not isscale:
            imaget = ski.color.rgb2gray(images)
        else:
            imaget = images

        bimgs = imaget[:(refer_idx+1),]
        fimgs = imaget[refer_idx:,]
        tmats = []
        if refer_idx > 0:
            btmat = self.regist_stack(bimgs[::-1,], 
                                  reference=reference,
                                  **kargs)
            tmats  = btmat[::-1,]
        if refer_idx < imaget.shape[0]:
            ftmat = self.regist_stack(fimgs, 
                                  reference=reference,
                                  **kargs)
            tmats = ftmat if len(tmats) == 0 else np.concatenate([tmats[:-1,], ftmat])
        self.new_tmats = np.float64(tmats)
        self.images = images
        return self

    def transform(self,
                   images = None, 
                   tmats =None,
                    isscale=None,
                    trans_name = 'skimage',
                    n_jobs = 10,
                    backend="multiprocessing"):
        images = self.images if images is None else images
        tmats = self.new_tmats if tmats is None else tmats
        isscale = isscale or self.onechannel(images)

        if trans_name=='stackreg':
            if isscale:
                #mov_imgs = sr.transform_stack(images, tmats=tmats)
                mov_imgs = Parallel(n_jobs=n_jobs, backend=backend)\
                                    (delayed( self.sr.transform )
                                    (images[i], tmat=tmats[i]) for i in range(images.shape[0]))
                mov_imgs = np.array(mov_imgs)
            else:
                #mov_imgs = []
                #for color in range(images.shape[3]):
                #    iimg = sr.transform_stack(images[:, :, :, color], tmats=tmats)
                #    mov_imgs.append(image_mc)
                colors = images.shape[3]
                volumns =images.shape[0]
                mov_imgs = []
                for color in range(colors):
                    iimages = [ i[..., color]/255 for i in  images ]
                    imov_img = Parallel(n_jobs=n_jobs, backend=backend)\
                                       (delayed(self.sr.transform )
                                        (iimages[i], tmat=tmats[i]) for i in range(len(iimages)))
                    for i in range(volumns):
                        imov_img[i] = np.clip(np.round(imov_img[i] * 255), 0, 255).astype(np.uint8)[...,np.newaxis]
                    imov_img = np.array(imov_img)
                    mov_imgs.append(imov_img)
                mov_imgs = np.concatenate(mov_imgs, axis=3)
        elif trans_name=='skimage':
            mov_imgs = []
            volumns =images.shape[0]
            imgtype = images.dtype
            for i in range(images.shape[0]):
               iimg = ski.transform.warp(images[i], tmats[i])
               if np.issubdtype(imgtype, np.integer) or (imgtype in [np.uint8, np.uint16, np.uint32, np.uint64]):
                   iimg = np.clip(np.round(iimg * 255), 0, 255).astype(np.uint8)
               mov_imgs.append(iimg)
            # mov_imgs = Parallel(n_jobs=n_jobs, backend=backend)\
            #                     (delayed( ski.transform.warp )
            #                               (images[i].copy(), tmats[i].copy()) for i in range(images.shape[0]))
            mov_imgs = np.array(mov_imgs)
        self.mov_out = mov_imgs
        return mov_imgs

    def transform_points(self, locs, tmats=None, inverse=False):
        tmats = self.new_tmats if tmats is None else tmats
        assert len(locs) == len(tmats), 'the length between locs and tmats must be same.'
        new_locs = [ self.trans_point2d(locs[i], tmats[i], inverse=inverse)
                     for i in range(len(locs))]
        self.new_locs=new_locs
        return new_locs

    def regist_transform(self, images, 
                            tmats =None,
                            locs=None, 
                            isscale=None,
                            refer_idx=None,
                            reference='previous',
                            trans_name = 'skimage',
                            n_jobs = 10,
                            backend="multiprocessing",
                            **kargs):
        isscale = isscale or self.onechannel(images)
        self.regist( images, 
                     isscale=isscale,
                     refer_idx=refer_idx,
                     reference=reference,
                     **kargs)
        self.transform(
                   tmats = tmats,
                    isscale=isscale,
                    trans_name = trans_name,
                    n_jobs = n_jobs,
                    backend=backend)

        if not locs is None:
            self.transform_points(locs, inverse=False)
        return self

    @staticmethod
    def trans_point2d(locs, tmat, inverse=False):
        if locs is None:
            return locs
        locs = np.asarray(locs).copy()
        new_locs = locs.copy()[:,:2]
        new_locs = np.c_[new_locs, np.ones(new_locs.shape[0])]

        if inverse:
            new_locs =  new_locs @ tmat.T
        else:
            new_locs =  new_locs @ np.linalg.inv(tmat).T

        locs[:,:2] = new_locs[:,:2]
        return locs

    @staticmethod
    def onechannel(images):
        if images.ndim==3:
            return True
        elif images.ndim==4:
            return False
        else:
            raise ValueError('the images must have 3 or 4 dims.')

    @staticmethod
    def scaledimg(images):
        if (np.issubdtype(images.dtype, np.integer) or
            (images.dtype in [np.uint8, np.uint16, np.uint32, np.uint64])) and \
            (images.max() > 1):
            return False
        else:
            return True
