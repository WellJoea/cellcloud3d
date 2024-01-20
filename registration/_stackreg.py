from cellcloud3d.registration._cv2reg import cv2regist
from cellcloud3d.registration._itkreg import itkregist
from cellcloud3d.tools._search import searchmatch
from cellcloud3d.plotting import qview
from cellcloud3d.utilis._arrays import isidentity
from cellcloud3d.transform import homotransform_point, homotransform_points

import skimage as ski

import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm
import copy

class stackregist():
    def __init__(self,transtype='rigid', 
                 resolutions = 20, 
                 GridSpacing=10):
        self.transtype = transtype
        self.resolutions = resolutions
        self.GridSpacing = GridSpacing
        self.__cv2args = dict(
            transtype=transtype,
            estimate_method = 'skimage',
            skicolor=None,
            filter=None,

            kpds_fix = None, 
            kpds_mov = None,
            nomaltype = None,
            feature_args = {'drawfeature':False},

            max_trials=1000,
            CI = 0.95,

            fixmov_pts = None,
            verify_matches = None,
            matchesMask = None,
            match_args = {},
            drawmatch =False,
            drawm_args = {},
            locs=None, 
        )
        self._cv2args = copy.deepcopy(self.__cv2args)
        self._itkargs = itkregist.initparameters(trans=self.transtype,  
                                              resolutions=self.resolutions,
                                              GridSpacing=self.GridSpacing)
        self.itkargs = self._itkargs

    def buildidx(self, n_node,
                        step=1, 
                        root=None, 
                        regist_pair=None,
                        showtree=False, 
                        layout="spectral"):
        align_pair, trans_pairs = searchmatch.searchidx(n_node,
                                                labels=None,
                                                step=step, 
                                                root=root, 
                                                regist_pair=regist_pair,
                                                full_pair=False,
                                                showtree=showtree, 
                                                keep_self = False,
                                                layout=layout)
        self.align_pair  = align_pair
        self.trans_pairs = trans_pairs
        return [self.align_pair, self.trans_pairs]
        # assert  (not root is None) or (not regist_pair is None), 'A least one of root and regist_pair is not None.'

        # align_idx = None
        # trans_idx = None
        # if not regist_pair is None:
        #     align_idx = searchmatch().buildidx(n_node=n_node, 
        #                                         step=step, 
        #                                         root=None, 
        #                                         edges=regist_pair,
        #                                         showtree=showtree, 
        #                                         layout=layout).dfs_edges

        # if not root is None:
        #     trans_idx = searchmatch().buildidx(n_node=n_node, 
        #                                 step=step, 
        #                                 root=root, 
        #                                 edges=None,
        #                                 showtree=showtree, 
        #                                 layout=layout).dfs_edges
        # self.align_pair = trans_idx if align_idx is None else align_idx
        # self.trans_pairs = align_idx if trans_idx is None else trans_idx

        # return [self.align_pair, self.trans_pairs]

    @property
    def cv2args(self):
        return self._cv2args
    def set_cv2args_global(self, values):
        self.__cv2args.update(values or {})
        self._cv2args = copy.deepcopy(self.__cv2args)
    def set_cv2args_local(self, values):
        '''
        kargs={
            "feature_args":{'method':'sift','drawfeature':False, 
                            'contrastThreshold':0.03,
                            'nOctaveLayers':12,
                            'edgeThreshold':50,
                            'sigma':1.8}, 
            "match_args":{'method':'knn', 'verify_ratio':0.9},
        }
        kargs={
            "feature_args":{'drawfeature':False}, 
            "match_args":{'method':'flann', 'verify_ratio':0.8},
        }
        '''
        self._cv2args.update(values or {})
    def opencv(self, fix, mov):
        rg = cv2regist(transtype= self.cv2args['transtype'])
        #rg.regist_transform(fix, mov, **self.cv2args)
        rg.regist(fix, mov, **self.cv2args)
        return rg

    @property
    def init_itkargs(self):
        self.itkargs = self._itkargs
        if self.transtype=='rigid':
            #self.itkargs.SetParameter(0, "Optimizer", "RegularStepGradientDescent")
            self.itkargs.SetParameter(0, "MaximumNumberOfIterations", "2000")
            # self.itkargs.SetParameter(0, "FixedImagePyramidRescaleSchedule", "64 64 32 32 8 8 4 4 2 2 1 1")
            # self.itkargs.SetParameter(0, "MovingImagePyramidRescaleSchedule", "64 64 32 32 8 8 4 4 2 2 1 1")
            # self.itkargs.SetParameter(0, "MaximumStepLength", "1")
        elif self.transtype=='similarity':
            self.itkargs.SetParameter(0, "MaximumNumberOfIterations", "2000")
            # self.itkargs.SetParameter(0, "FixedImagePyramidRescaleSchedule", "64 64 32 32 8 8 4 4 2 2 1 1")
            # self.itkargs.SetParameter(0, "MovingImagePyramidRescaleSchedule", "64 64 32 32 8 8 4 4 2 2 1 1")
            # self.itkargs.SetParameter(0, "MaximumStepLength", "20")
            # self.itkargs.SetParameter(0, "Optimizer", "RegularStepGradientDescent")
            # self.itkargs.SetParameter(0, "NumberOfHistogramBins", "32")
            #itkt.params.SetParameter(0, "NumberOfSpatialSamples", "4000")
    def set_itkargs(self,
                     resolutions=None,
                     GridSpacing=None, 
                     verb=False):
        # https://elastix.lumc.nl/doxygen/parameter.html
        self.itkargs = itkregist.initparameters(trans=self.transtype,
                                             resolutions=resolutions,
                                             GridSpacing=GridSpacing,
                                             verb=verb
                                             )
    def itk(self, fix, mov, **kargs):
        rg =  itkregist(transtype=self.transtype)
        rg.params = self.itkargs
        rg.regist(fix, mov, **kargs)
        return rg

    def regist(self,
                     images, 
                     locs=None,
                     regist_method='opencv',
                     init_tmats=None,
                     regist_pair=None,
                     isscale=None,
                     step=1,
                     root=None,
                     showtree=False,
                     broadcast=True,
                     layout="spectral",
                     cv2global = {},
                     cv2local = [],
                     itkarglist = [],
                     n_jobs = 1,
                     backend="multiprocessing",
                     verbose=1,
                     itk_threds=50,
                     **kargs):
        align_pair, trans_pairs = self.buildidx(len(images), 
                                step=step, 
                                root=root, 
                                regist_pair=regist_pair,
                                showtree=showtree, 
                                layout=layout)

        if len(cv2local)>0:
            cv2local = [{} if i is None else i for i in cv2local]
            assert len(cv2local) == len(align_pair), \
                f'the lenght of cv2local must be {len(align_pair)}.'
        else:
            cv2local = [{}] * len(align_pair)

        if len(itkarglist)>0:
            assert len(itkarglist) == len(align_pair), \
                f'the lenght of cv2local must be {len(align_pair)}.'
        else:
            itkarglist = [dict()] * len(align_pair)

        if  locs is None:
            locs = [None] * len(images)
        else:
            assert len(locs) == len(images), \
                f'the lenght of locs must be {len(images)}.'

        tmats = [np.identity(3)] * len(images) if init_tmats is None else init_tmats
        if regist_method=='opencv':
            self.set_cv2args_global(cv2global)
            if n_jobs==1:
                for _n,(fix, mov) in tqdm(enumerate(align_pair)):
                    verbose and print(f'fix {fix} -> mov {mov}')
                    self.set_cv2args_local(cv2local[_n])
                    irg = self.opencv(images[fix], images[mov], **kargs)
                    tmats[mov] = irg.tmat
                    #tmats[mov] = np.matmul(irg.tmat, tmats[mov])
                    #tmats[mov] = np.matmul(irg.tmat, tmats[fix])

            else:
                agws = []
                for iarg in cv2local:
                    self.set_cv2args_local(iarg)
                    agws.append(self.cv2args)
                mtxs = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)\
                                    (delayed( stackregist.opencv_pipe )
                                    (images[fix], images[mov], **agws[_n], **kargs) 
                                    for _n,(fix,mov) in enumerate(align_pair))
                for _n,(fix, mov) in enumerate(align_pair):
                    tmats[mov] = mtxs[_n]
                    #tmats[mov] = np.matmul(mtxs[_n], tmats[mov]) 

        if regist_method=='itk':
            if n_jobs==1:
                for _n,(fix, mov) in tqdm(enumerate(align_pair)):
                    verbose and print(f'fix {fix} -> mov {mov}')
                    irg = self.itk(images[fix], 
                                   images[mov],
                                    number_of_threads=itk_threds,
                                    **itkarglist[_n], **kargs)
                    tmats[mov] = irg.tmat[0]

        elif regist_method=='dipy':
            pass
        elif regist_method=='ants':
            pass
        elif regist_method=='sitk':
            pass

        self.images = images
        self.raw_tmats = tmats
        self.align_pair = align_pair
        self.new_tmats = self.update_tmats(trans_pairs, tmats) if broadcast else tmats

    def transform(self,
                   images = None, 
                   tmats =None,
                    isscale=None,
                    trans_name = 'skimage',
                    n_jobs = 10,
                    backend="multiprocessing"):
        images = self.images if images is None else images
        tmats = self.new_tmats if tmats is None else tmats
        isscale = self.scaledimg(images) if isscale is None else isscale

        if trans_name=='skimage':
            mov_imgs = []
            imgtype = images.dtype
            for i in range(images.shape[0]):
                if isidentity(tmats[i]):
                    iimg = images[i]
                else:
                    iimg = ski.transform.warp(images[i], tmats[i])
                    if np.issubdtype(imgtype, np.integer):
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
        new_locs = homotransform_points(locs, tmats, inverse=inverse)
        self.new_locs=new_locs
        return new_locs

    def regist_transform(self, images, 
                            locs=None,
                            tmats =None,
                            regist_method='opencv',
                            regist_pair=None,
                            isscale=None,
                            step=1,
                            root=None,
                            showtree=False,
                            broadcast=True,
                            layout="spectral",
                            cv2global ={},
                            cv2local = [],
                            itkarglist = [],
                            n_jobs = 1,
                            itk_threds=5,
                            backend="multiprocessing",
                            verbose=1,
                            trans_name = 'skimage',
                            **kargs):
        isscale = self.scaledimg(images) if isscale is None else isscale
        self.regist( images, 
                     locs=locs,
                     regist_method=regist_method,
                     regist_pair=regist_pair,
                     isscale=isscale,
                     step=step,
                     root=root,
                     showtree=showtree,
                     broadcast=broadcast,
                     layout=layout,
                     cv2global=cv2global,
                     cv2local = cv2local,
                     itkarglist = itkarglist,
                     n_jobs = n_jobs,
                     itk_threds=itk_threds,
                     backend= backend,
                     verbose= verbose,
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
    def update_tmats(trans_pairs, raw_tmats):
        try:
            new_tmats = raw_tmats.copy()
        except:
            new_tmats = [ imt.copy() for imt in raw_tmats]
        
        for ifov, imov in trans_pairs:
            new_tmats[imov] = np.matmul(new_tmats[imov], new_tmats[ifov])
        return new_tmats

    @staticmethod
    def opencv_pipe(fix, mov, **kargs):
        rg = cv2regist(transtype= kargs['transtype'])
        #rg.regist_transform(fix, mov, **kargs)
        rg.regist(fix, mov, **kargs)
        return rg.tmat

    @staticmethod
    def scaledimg(images):
        if (np.issubdtype(images.dtype, np.integer) or
            (images.dtype in [np.uint8, np.uint16, np.uint32, np.uint64])) and \
            (images.max() > 1):
            return False
        else:
            return True