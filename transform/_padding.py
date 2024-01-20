import numpy as np
from cellcloud3d.io import read_image_info, write_image_info

def padding_spatial(
                adatas,
                img_key="hires",
                basis = 'spatial', 
                library_id=None,
                inplace=True,
                resize = None):

    if not inplace:
        adatas = [ idata.copy() for idata in adatas ]
    images = []
    locs = []
    sfs = []
    for i in range(len(adatas)):
        iadata = adatas[i]
        imginfo = read_image_info(iadata, img_key=img_key,
                                    basis = basis, 
                                    library_id=library_id,
                                    get_pix_loc=False, 
                                    rescale=None)
        images.append(imginfo['img'])
        locs.append(imginfo['locs'].values)
        sfs.append(imginfo['rescale'])

    maxhw = np.vstack([i.shape[:2] for i in images]).max(0)
    resize = maxhw if resize is None  else resize
    print(f'all the image will set to the same size: {resize}.')

    cpd = padding()
    cpd.fit_transform(images, points=locs, resize=resize)
    for i in range(len(adatas)):
        isf = sfs[i]
        nimg = cpd.imagesT[i]
        nlocs = cpd.pointsT[i]/isf

        write_image_info(adatas[i], 
                         image = nimg, 
                         locs = nlocs, 
                        img_key=img_key,
                        basis = basis, 
                        library_id=library_id,
                        keepraw=False)
    if not inplace:
        return adatas

class padding():
    def __init__(self):
        pass

    def fit(self, images, resize=None, paddims=None, verbose=False):
        if paddims is None:
            if resize is None:
                paddims = images[0].ndim
            else:
                paddims = len(resize)
            paddims = range(paddims)
        if verbose:
            print(f'padding dims are {paddims}')

        if resize is None:
            resize = np.array([i.shape for i in images]).max(0)
        print(f'resize shape is {resize}')

        pad_width = []
        pad_front = []
        for img in images:
            iwidth = [(0,0)] * img.ndim
            ifront = [0] * img.ndim
            for idim in paddims:
                iwid = img.shape[idim]
                twid = resize[idim]
                befor = (twid - iwid)//2
                after = twid - iwid - befor
                iwidth[idim] = (befor, after)
                ifront[idim] = befor
            pad_width.append(iwidth)
            pad_front.append(ifront)

        self.images = images
        self.pad_width = pad_width
        self.pad_front = pad_front
        self.resize = resize
        self.pad_dims = paddims

    def transform(self, images=None, pad_width=None, **kargs):
        images = self.images if images is None else images
        pad_width = self.pad_width if pad_width is None else pad_width
        imagesT = [ self.padcrop(images[i], pad_width[i], **kargs)
                            for i in range(len(images)) ]
        self.imagesT = imagesT
        return imagesT

    def transform_points(self, points, pad_front=None, inversehw=True):
        if points is None:
            pointsT = None
        else:
            pad_front = self.pad_front if pad_front is None else pad_front
            pointsT = [ self.padpos(points[i], pad_front[i], inversehw=inversehw) 
                            for i in range(len(points)) ]
        self.pointsT = pointsT
        return pointsT

    def fit_transform(self, images, points=None, resize=None,
                        inversehw=True, verbose=False,
                        paddims=None,  **kargs):
        self.fit( images, resize=resize, paddims=paddims, verbose=verbose)
        self.transform()
        self.transform_points(points, inversehw=inversehw)
        return self

    @staticmethod
    def pad(img, 
                pad_width=([30,30],[30,30]),
                constant_values= 0,
                mode ='constant',
                use_np=True,
                **kargs):
        if use_np:
            return np.pad( img, pad_width , mode ='constant', constant_values=constant_values)
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

    @staticmethod
    def padcrop(img, 
            pad_width=([30,30],[30,30]),
            constant_values= 0,
            mode ='constant',
            **kargs):
        if np.array(pad_width).min()>=0:
            return np.pad( img, pad_width , mode =mode, constant_values=constant_values, **kargs)
        else:
            iimg = img.copy()
            crop = np.clip(pad_width, None, 0)
            pad =  np.clip(pad_width, 0, None)
            sl = [slice(None)] * iimg.ndim
            for i in range(crop.shape[0]):
                sl[i]= slice(np.abs(crop[i][0]),
                            None if crop[i][1]==0 else crop[i][1], 
                            None)
            iimg = iimg[tuple(sl)]
            iimg = np.pad( iimg, pad , mode =mode, constant_values=constant_values, **kargs)
            return iimg.astype(img.dtype)

    @staticmethod
    def padpos(pos, pad_front=[30,30], inversehw=True):
        if len(pad_front)<pos.shape[1]:
            padfull = [0]*pos.shape[1]
            padfull[:len(pad_front)] = pad_front
        else:
            padfull = pad_front[:pos.shape[1]]
        if inversehw:
            padfull = np.asarray(padfull)
            padfull[[0,1]]=padfull[[1,0]]
        return pos + padfull
        # tp, lf = tl
        # ipos = pos.copy()
        # ipos[:,0] += lf
        # ipos[:,1] += tp
        # return ipos.astype(pos.dtype)