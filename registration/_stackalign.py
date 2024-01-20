import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from cellcloud3d.registration._stackreg import stackregist
from cellcloud3d.registration._turbostack import turbostack
from cellcloud3d.utilis import list_get
from cellcloud3d.io import read_image_info, write_image_info, write_image_infos
from cellcloud3d.transform import (padding, rescale_points, homotransform_points, homotransforms, rescales)
#from .. import log

class spatialalign3d():
    def __init__(self, align_type = 'rigid', transformRecords=[], ):
        self.align_type = align_type
        self.transformRecords = transformRecords
        self.aligner = None,

    def preparing(self, 
                  adatas = None,
                  images = None,
                  locs = None,
                    img_key="hires",
                    basis = None, 
                    library_id=None,
                    get_pix_loc=False, 
                    resize = None,
                    order=None, 
                    add_id = 'cc3d',
                    update_obs_index=True,
                    inplace=True,
                    scale=None, 

                    scale_method = 'skimage',
                    **kargs):
        #log.CI('check data and fatch images and other information.')
        if adatas is None:
            self.adatas = None
        else:
            self.adatas = adatas if inplace else [idata.copy() for idata in adatas]

        if images is None:
            if self.adatas is None:
                raise ValueError('adatas is None, please provide images and locs.')
            images = []
            locs = []

            for iadata in self.adatas:
                imginfo = read_image_info(iadata, img_key=img_key,
                                            basis = basis, 
                                            library_id=library_id,
                                            get_pix_loc=get_pix_loc, 
                                            rescale=None)
                images.append(imginfo['img'])
                locs.append(imginfo['locs'].values)
        maxhw = max(max(i.shape[:2]) for i in images)
        resize = [maxhw, maxhw] if resize is None  else resize
        print(f'all the image will set to the same size: {resize}.')

        cpd = padding()
        cpd.fit_transform(images, points=locs, resize=resize)

        self.scale = scale
        self.images_raw = np.array(cpd.imagesT)
        self.imgres = self.images_raw.dtype
        self.locs_raw = cpd.pointsT

        if not self.adatas is None:
            self.adatas = self.adatacat(self.adatas, add_id = add_id, update_obs_index=update_obs_index)
            self.adatas.obs['zaxis'] = self.adatas.obs[add_id].astype(np.float32)

        if not scale is None:
            if type(scale) in [int, float]:
                scale = [scale] * len(self.images_raw)
            self.images = rescales(self.images_raw, scale, order=order, method=scale_method, **kargs )
            self.locs = None if self.locs_raw is None else rescale_points(self.locs_raw, scale)
        else:
            self.images = self.imgres.copy()
            self.locs = self.locs_raw

        if np.issubdtype(self.imgres, np.integer):
            scale_max = self.images.max()
            self.images = np.round(self.images * 255./scale_max).astype(np.uint8)
            # if self.images.ndim == 3:
            #     self.images /= scale_max
            # elif self.images.ndim == 4:
            #     self.images = np.round(self.images * 255./scale_max).astype(np.uint8)

        print(f'use size {self.images.shape} to registration.')

    def init_align(self, method='cv2homo', align_type=None, resolutions = 30, GridSpacing = 10):
        align_type = align_type or self.align_type
        self.aligner_method = method
        self.align_type = align_type

        if method == 'turbo':
            self.aligner = turbostack(transtype = align_type)
        elif method == 'cv2homo':
            self.aligner  = stackregist(transtype = self.align_type)
        elif method == 'elastix':
            self.aligner = stackregist(transtype = self.align_type, resolutions=resolutions, GridSpacing=GridSpacing)
        else:
            raise ValueError('the current methods only include "turbo, cv2homo, elastix".')

    def align(self, 
               images=None, 
                tmats =None,
                locs=None, 
                isscale=None,

                reference='previous',
                trans_name = 'skimage',
                n_jobs = 10,
                backend="multiprocessing",

                regist_pair=None,
                step=1,
                root=None,
                showtree=False,
                broadcast=True,
                layout="spectral",
                cv2global ={},
                cv2local = [],
                itkarglist = [],
                itk_threds=5,

                verbose=0,
                **kargs):
        images = self.images if images is None else images
        locs =  self.locs if locs is None else  locs

        if self.aligner_method == 'turbo':
            self.aligner.regist_transform(images, 
                                tmats =tmats,
                                locs=locs, 
                                isscale=isscale,
                                refer_idx=root,
                                reference=reference,
                                trans_name = trans_name,
                                n_jobs = n_jobs,
                                backend=backend)

        elif self.aligner_method == 'cv2homo':
            self.aligner.regist_transform(images, 
                                    locs=locs,
                                    tmats =tmats,
                                    regist_method='opencv',
                                    regist_pair=regist_pair,
                                    isscale=isscale,
                                    step=step,
                                    root=root,
                                    showtree=showtree,
                                    broadcast=broadcast,
                                    layout=layout,
                                    cv2global = cv2global,
                                    cv2local = cv2local,
                                    itkarglist = itkarglist,
                                    n_jobs = 1,
                                    itk_threds=itk_threds,
                                    backend=backend,
                                    verbose=verbose,
                                    trans_name = trans_name,
                                    **kargs)

        elif self.aligner_method == 'elastix':
            self.aligner.regist_transform(images, 
                                    locs=locs,
                                    tmats =tmats,
                                    regist_method='itk',
                                    regist_pair=regist_pair,
                                    isscale=isscale,
                                    step=step,
                                    root=root,
                                    showtree=showtree,
                                    broadcast=broadcast,
                                    layout=layout,
                                    cv2global = cv2global,
                                    cv2local = cv2local,
                                    itkarglist = itkarglist,
                                    n_jobs = 1,
                                    itk_threds=itk_threds,
                                    backend=backend,
                                    verbose=verbose,
                                    trans_name = trans_name,
                                    **kargs)
        
    def records(self, tmats=True, locs=True, mov_out=False, regist_pair=True):
        recodict = {}
        if tmats:
             recodict.update({'tmats' : getattr(self.aligner, 'new_tmats', None)})
        if locs:
             recodict.update({'locs' : getattr(self.aligner, 'new_locs', None)})
        if mov_out:
             recodict.update({'mov_out' : getattr(self.aligner, 'mov_out', None)})
        if regist_pair:
             recodict.update({'regist_pair' : getattr(self.aligner, 'trans_pairs', None)})
        self.transformRecords.append(recodict)

    @property
    def get_records(self):
        return self.transformRecords

    def del_records(self, idx):
        if type(idx) in [int]:
            del self.transformRecords[idx]
        elif type(idx) in [list]:
            for iidx in idx:
                del self.transformRecords[iidx]

    def trans_tmats(self,  transformRecords, rescale = None):
        from functools import reduce
        tmats = [ itrans['tmats'] for itrans in transformRecords]
        tmats = reduce(lambda x,y:x @ y, tmats)

        if not rescale is None: # error for different shape
            rescale_matrx = np.tile(np.eye(3), (tmats.shape[0],1, 1))
            rescale_matrx[:, [0,1], [0,1]] = rescale #x y same scale
            tmats = tmats @ rescale_matrx
        self.final_tmats = tmats
        return tmats

    def trans_align(self, images = None,
                    locs = None,
                    tmats = None,
                    padingsize =None,
                    n_jobs = 10, 
                    order=None,
                    keeptype=True,
                    trans_name = 'skimage',
                     backend='loky', **kargs):
        
        if not images is None:
            imagen = homotransforms(images, tmats,
                                            order=order,
                                            keeptype=keeptype,
                                            trans_name = trans_name,
                                            n_jobs = n_jobs,
                                            backend= backend,
                                            verbose=0, **kargs)
        else:
            imagen = None

        if not locs is None:
            locsn = homotransform_points(locs, tmats, inverse=False)
        else:
            locsn = None
        if (not padingsize is None) and (not imagen is None):
            cpd = padding()
            cpd.fit_transform(imagen, points=locsn, resize=padingsize)
            imagen = np.array(cpd.imagesT)
            locsn =  cpd.pointsT

            self.pad_width = cpd.pad_width
            self.pad_front = cpd.pad_front
            self.pad_resize = cpd.resize
        # self.final_images = imagen
        # self.final_locs = locsn
        return [imagen, locsn]

    def update_adata(self, 
                    images = None, 
                    locs = None, 
                    zdists = None,
                    img_key="hires",
                    basis = 'spatial', 
                    library_ids=None, 
                    metadata=None,
                    keepraw=False,
                    scale = None,
                    tissue_hires_scalef = None,
                    tissue_lowres_scalef = None,
                    **kargs):

        spatial = None if locs is None else np.concatenate(locs, axis=0) 
        if (not zdists is None) and (not spatial is None):
            if type(zdists) in (int, float):
                zaxis = self.adatas.obs['zaxis'].astype(np.float32) * zdists
            elif type(zdists) in (list, np.ndarray):
                assert len(zdists) == len(locs) , f'the zdists lenght must be {len(locs)}'
                zaxis =  np.repeat(zdists, [len(i) for i in locs])
                kargs.update(zaxis = zaxis)
            self.adatas.obsm[f'{basis}3d'] = np.c_[spatial, zaxis]

        write_image_infos(self.adatas, 
                        images = images, 
                        locs = spatial, 
                        library_ids=library_ids, 
                        inplace = True,
    
                        img_key=img_key,
                        basis = basis, 
                        metadata=metadata,
                        keepraw = keepraw,

                        tissue_hires_scalef=tissue_hires_scalef,
                        tissue_lowres_scalef=tissue_lowres_scalef,
                        rescale = scale,
                        **kargs)

    def update(self, transformRecords = None, padingsize = 'auto',
                pad_tola = 50, n_jobs = 10,  order=None,
                keeptype=True,
                img_key = 'hires',
                trans_name = 'skimage', backend='loky',
                zdists = None, basis = 'spatial', 
                library_ids =None, 
                keepraw=False,
                lowscale = None, **kargs):

        transformRecords = self.transformRecords if transformRecords is None else transformRecords
        self.tmats = self.trans_tmats( transformRecords, rescale=None)
        if padingsize == 'auto' and (not self.locs is None):
            locsn = homotransform_points(self.locs, self.tmats, inverse=False)
            locsn = np.concatenate(locsn, axis=0) 
            imghw = np.array(self.images.shape[1:3])
            maxout  = min((imghw - locsn.max(0)).min(), locsn.min())
            if maxout <0:
                padingsize = np.round([pad_tola-maxout+ imghw[0], pad_tola-maxout+imghw[1]], 0).astype(np.int64)
                print(f'auto set padding set to {padingsize}')
            else:
                padingsize = None
        else:
            padingsize = None
        images, locs = self.trans_align( images =  self.images, 
                                        locs = self.locs  ,
                                        tmats = self.tmats,
                                        padingsize =padingsize,
                                        n_jobs = n_jobs, 
                                        order=order,
                                        keeptype=keeptype,
                                        trans_name = trans_name,
                                        backend=backend)
        self.images = images
        self.locs = locs

        if not self.adatas is None:
            self.zdists =  max(images.shape[1:3])/images.shape[0] if zdists is None else zdists
            self.update_adata(images = images,
                                locs = locs, 
                                zdists = zdists,
                                img_key=img_key,
                                basis = basis, 
                                library_ids=library_ids, 
                                keepraw=keepraw,
                                **kargs)

            self.adatas.uns['tmats'] = self.tmats
            self.adatas.uns['zdists'] = self.zdists
            self.adatas.uns['locs'] = np.vstack(locs)
            self.adatas.uns['images'] = images

            if (not lowscale is None) and (0< lowscale <1) and (img_key=='hires'):
                imagel = rescales(images, lowscale )
                self.update_adata( images = imagel, 
                                    locs = None, 
                                    zdists = zdists,
                                    img_key='lowres',
                                    tissue_lowres_scalef = lowscale,
                                    basis = basis, 
                                    library_ids=library_ids, 
                                    keepraw = False,
                                    inplace = True,
                                    **kargs)
            print(f'finished: added to `.obsm["{basis}3d"]`')
            print(f'          added to `.uns["tmats"]`')
            print(f'          added to `.uns["zdists"]`')
            print(f'          added to `.uns["locs"]`')
            print(f'          added to `.uns["images"]`')

    @staticmethod
    def adatacat(adatas,
                  add_id = 'cc3d',
                  update_obs_index=True,
                 **kargs):
        adata = ad.concat(adatas, label=add_id,  **kargs)
        adata.var.index.name = 'Gene'
        if update_obs_index:
            adata.obs_names = adata.obs_names + '_' + adata.obs[add_id].astype(str)
        return adata