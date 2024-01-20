import skimage as ski
import scipy as sci
import skimage.transform as skitf
from skimage.measure import LineModelND
import numpy as np
try:
    import cv2
except ImportError:
    pass

from cellcloud3d.plotting import qview
class cv2regist():
    def __init__(self, transtype=None):
        self.transtype = 'rigid' if transtype is None else transtype
        self.TRANS = {
            'skimage':{
                'rigid':skitf.EuclideanTransform,
                'euclidean':skitf.EuclideanTransform,
                'similarity':skitf.SimilarityTransform,
                'affine':skitf.AffineTransform,
                'projective':skitf.ProjectiveTransform,
            }
        }
        self.feature_args = {
                        'method': 'sift',
                        'nfeatures':None,
                        'nOctaveLayers':12,
                        'contrastThreshold':0.03,
                        'edgeThreshold':50,
                        'hessianThreshold': 400,
                        'nOctaves':4, 
                        'sigma' : 1.6,
                        'surf_nOctaveLayers':3,
                        'orb_nfeatures':2000,
                        'scaleFactor':1.2,
                        'nlevels':8,
                        'orb_edgeThreshold':31,
                        'firstLevel':0,
                        'WTA_K':2,
                        #scoreType=0, 
                        'patchSize':31, 
                        'fastThreshold':20,
                        'extended':False, 
                        'upright':False,
                        'drawfeature':True,
                        'color':(255,0,0),
                        'flags':cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS}
        self.match_args = {
                        'method':'knn',
                        'drawpoints':2000,
                        'verify_ratio' : 0.8,
                        'reprojThresh' : 5.0,
                        'feature_method' : None,
                        'table_number' : 6, # 12
                        'key_size' : 12, # 20
                        'multi_probe_level' : 1,
                        'trees':5,
                        'checks':50,
                        'verbose':1}
        self.ransac_args = {
                    'is_data_valid': None, 
                    'is_model_valid': None, 
                    'stop_sample_num': np.inf,
                    'stop_residuals_sum': 0, 
                    'stop_probability': 1, 
                    'residual_trials': 100,
                    'min_pair':10,
                    'stop_merror': 1e-3,
                    'verbose':0,
                    'initial_inliers': None}

        self.drawm_args = {
                        'drawpairs':2000,
                        'matchColor': (0,255,0),
                        'matchesThickness':2,
                        'singlePointColor' : (255,0,0),
                        'flags':0}

    def regist(self, fixed_img, moving_img,
                    transtype=None,
                    estimate_method = 'skimage',
                    skicolor=None,
                    filter=None,

                    kpds_fix = None, 
                    kpds_mov = None,
                    nomaltype = None,
                    feature_args = {},

                    use_Ransac = True,
                    min_samples=3, 
                    residual_threshold=1, 
                    max_trials=1000,
                    CI = 0.95,
                    ransac_args = {},

                    fixmov_pts = None,
                    verify_matches = None,
                    matchesMask = None,
                    min_matches = 8,
                    match_args = {},
                    drawmatch =True,
                    drawm_args = {},
                    **kargs):

        self.feature_args.update(feature_args)
        self.drawm_args.update(drawm_args)
        self.match_args.update(match_args)
        self.ransac_args.update(ransac_args)

        transtype =self.transtype if transtype is None else transtype
        tfmethod = self.TRANS[estimate_method][self.transtype]
        # import inspect
        # sig = inspect.signature(self.regist)
        # params = sig.parameters
        # print(params)

        img1 = moving_img.copy()
        img2 = fixed_img.copy()
        if not filter is None:
            img1 = self.filters(img1, filter=filter)
            img2 = self.filters(img2, filter=filter)
        if not skicolor is None:
            img1_g  = self.colortrans(img1, skicolor=skicolor)
            img2_g  = self.colortrans(img2, skicolor=skicolor)
        else:
            img1_g = img1
            img2_g = img2

        if kpds_fix is None:
            kp1, ds1, nomaltype = self.features(img1_g, **self.feature_args)
        else:
            kp1, ds1 = kpds_fix

        if kpds_mov is None:
            kp2, ds2, nomaltype = self.features(img2_g, **self.feature_args)
        else:
            kp2, ds2 =kpds_mov

        self.match_args['nomaltype'] = nomaltype
        self.match_args['feature_method'] = self.feature_args.get('method', None)
        if fixmov_pts is None:
            verify_matches, matchesMask, fix_pts_raw, mov_pts_raw= \
                self.matchers(kp1, ds1, kp2, ds2, min_matches=min_matches,
                              **self.match_args)
        else:
            fix_pts_raw, mov_pts_raw = fixmov_pts

        if use_Ransac:
            fix_pts, mov_pts, inliers, model = self.autoresidual(
                    fix_pts_raw, mov_pts_raw, model_class=tfmethod,
                    CI = CI,
                    min_samples=min_samples,
                    max_trials=max_trials,
                    **self.ransac_args)

        else:
            fix_pts, mov_pts = fix_pts_raw, mov_pts_raw
            inliers = [True] * len(fix_pts)

        #print(len(fix_pts), len(mov_pts), model)
        assert len(verify_matches)> min_matches, f'lower than {min_matches} paired matches.'
        if drawmatch:
            drawm_args['matchesMask'] = matchesMask
            self.drawMatches(img1, kp1, #kp1[inliers], 
                             img2, kp2, #kp2[inliers], 
                             verify_matches,
                             inliers = inliers,
                             **self.drawm_args)

        tform = self.estimate(fix_pts, mov_pts,
                                tfmethod=tfmethod,
                                transtype=transtype)
        # print(model, tform)

        self.fixed_img = fixed_img
        self.moving_img = moving_img
        self.kp1 = kp1
        self.ds1 = ds1
        self.kp2 = kp2
        self.ds2 = ds2
        self.nomaltype = nomaltype
        self.verify_matches = verify_matches
        self.matchesMask = matchesMask
        self.fix_pts_raw = fix_pts_raw
        self.mov_pts_raw = mov_pts_raw
        self.fix_pts = fix_pts
        self.mov_pts = mov_pts
        self.inliers = inliers

        self.tform = tform
        self.tmat = np.float64(tform)
        return self

    def transform(self, 
                    moving_img=None, 
                    tform=None, 
                    out_shape=None,
                    preserve_range=False,
                    clip=True, 
                    map_args=None,
                    order=None, 
                    mode='constant', 
                    cval=0.0
                    ):
        moving_img = self.moving_img if moving_img is None else moving_img
        tform = self.tform if tform is None else tform
        mov_out = skitf.warp(moving_img, tform, 
                            output_shape=out_shape,
                            preserve_range = preserve_range,
                            clip=clip,
                            order=order,
                            mode=mode,
                            cval=cval,
                            map_args=map_args)
        if (np.issubdtype(moving_img.dtype, np.integer) or 
            (moving_img.dtype in [np.uint8, np.uint16, np.uint32, np.uint64])):
            mov_out = np.clip(np.round(mov_out * 255), 0, 255).astype(np.uint8)
        self.mov_out = mov_out
        return mov_out

    def regist_transform(self, fixed_img, moving_img, 
                            locs=None, inverse=False,
                            **kargs):
        self.regist(fixed_img, moving_img, **kargs)
        self.transform()
        self.transform_points(locs, self.tform,
                              inverse=inverse)
        return self

    @staticmethod
    def transform_points(locs, tmat, inverse=False):
        if locs is None:
            return locs
        locs = np.asarray(locs).copy()
        new_locs = locs.copy()[:,:2]
        new_locs = np.c_[new_locs, np.ones(new_locs.shape[0])]

        if inverse:
            new_locs =  new_locs @ tmat.T
        else:
            new_locs =  new_locs @ np.linalg.inv(tmat).T

        locs[:,:2] = new_locs[:,:2]/new_locs[:,[2]]
        return locs

    @staticmethod
    def filters(image, filter=None):
        if filter is None:
            image = image
        elif filter.lower() in ['blur', 'mean']:
            image = cv2.blur(image, (5,5))
        elif filter.lower() in ['gaussian']:
            image = cv2.GaussianBlur(image,(5,5),0)
        elif filter.lower() in ['median']:
            image = cv2.medianBlur(image, 5)
        elif filter.lower() == 'bilateral':
            image = cv2.bilateralFilter(image, 15,85,85 )
        return image

    @staticmethod
    def colortrans(image, skicolor='rgb2gray', *args, **kargs):
        colorfunc = eval(f'ski.color.{skicolor}')
        image = colorfunc(image, *args, **kargs)
        return image

    @staticmethod
    def features(image, method='sift',
                 nfeatures = None,
                 nOctaveLayers =15,
                 contrastThreshold=0.09, 
                 edgeThreshold=50,
                 sigma =1.6,

                 hessianThreshold = 400,
                 nOctaves=4, 
                 surf_nOctaveLayers=3, 

                 orb_nfeatures=500,
                 scaleFactor=1.2,
                 nlevels=8,
                 orb_edgeThreshold=31,
                 firstLevel=0,
                 WTA_K=2,
                 #scoreType=0, 
                 patchSize=31, 
                 fastThreshold=20,

                 extended=False, 
                 upright=False,
                 drawfeature=False,
                 color=(255,0,0),
                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
        if method == 'sift':
            detector = cv2.SIFT_create(nfeatures=nfeatures,
                                        nOctaveLayers =nOctaveLayers,
                                         contrastThreshold=contrastThreshold, 
                                         edgeThreshold=edgeThreshold, 
                                         sigma =sigma)
            nomaltype = cv2.NORM_L2
        elif method == 'surf':
            detector = cv2.xfeatures2d_SURF.create(hessianThreshold=hessianThreshold,
                                                    nOctaves=nOctaves, 
                                                    nOctaveLayers=nOctaveLayers, 
                                                    extended=extended, 
                                                    upright=upright)
            nomaltype = cv2.NORM_L2
        elif method == 'orb':
            detector = cv2.ORB_create( nfeatures=orb_nfeatures,
                                       scaleFactor=scaleFactor,
                                        nlevels=nlevels,
                                        edgeThreshold=orb_edgeThreshold,
                                        firstLevel=firstLevel,
                                        WTA_K=WTA_K,
                                        #scoreType=0, 
                                        patchSize=patchSize, 
                                        fastThreshold=fastThreshold)

            nomaltype = cv2.NORM_HAMMING

        kp, ds = detector.detectAndCompute(image, None)
        if drawfeature:
            imgdkp=cv2.drawKeypoints(image,
                                        kp,
                                        image,
                                        color=color, 
                                        flags=flags)

            qview(imgdkp)
        return kp, ds, nomaltype

    @staticmethod
    def matchers(kp1, ds1, kp2, ds2,
                nomaltype=cv2.NORM_L2, 
                method='knn',
                drawpoints=2000,
                min_matches = 8,
                verify_ratio = 0.7,
                reprojThresh = 5.0,
                feature_method = None,
                table_number = 6, # 12
                key_size = 12, # 20
                multi_probe_level = 1,
                knn_k = 2,
                trees=5,
                checks=50,
                verbose=True,
                **kargs):
        verbose = int(verbose)
        nomaltype = nomaltype or cv2.NORM_L2
        if method=='cross':
            bf = cv2.BFMatcher(nomaltype, crossCheck=True)
            matches = bf.match(ds1, ds2)
            matches = sorted(matches, key=lambda x:x.distance)

        elif method=='knn':
            bf = cv2.BFMatcher(nomaltype)
            matches = bf.knnMatch(ds1, ds2, k=knn_k)

        elif method=='flann':
            if feature_method == 'orb':
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                                    table_number = table_number, # 12
                                    key_size = key_size, # 20
                                    multi_probe_level = multi_probe_level) #2
            else:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)

            search_params = dict(checks=checks)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(ds1, ds2,k=2)

        verify_matches = []
        matchesMask = [[0,0] for i in range(len(matches))]
        for i,(m1,m2) in enumerate(matches):
            if m1.distance < verify_ratio * m2.distance:
                matchesMask[i]=[1,0]
                verify_matches.append(m1)

        verbose and print(f'find {len(verify_matches)} matches.')
        assert len(verify_matches)> min_matches, 'low paired matches.'

        fix_pts = []
        mov_pts = []
        for m in verify_matches:
            fix_pts.append(kp1[m.queryIdx].pt)
            mov_pts.append(kp2[m.trainIdx].pt)

        fix_pts = np.array(fix_pts).astype(np.float64) #.reshape(-1,1,2)
        mov_pts = np.array(mov_pts).astype(np.float64) #.reshape(-1,1,2)

        return [verify_matches,matchesMask,fix_pts,mov_pts]

    @staticmethod
    def autoresidual(src_pts, dst_pts, model_class=skitf.AffineTransform,
                    min_samples=3, residual_threshold =1, 
                    residual_trials=100, max_trials=500,
                    seed=200504, CI=0.95, stop_merror=1e-3,
                    min_pair=10,
                    is_data_valid=None, is_model_valid=None,
                    drawhist=False, verbose=0, **kargs):

        verbose = int(verbose)
        src_ptw = src_pts.copy()
        dst_ptw = dst_pts.copy()

        if not residual_threshold is None:
            assert 0<residual_threshold <=1
        model_record = np.eye(3)

        n_feature = len(src_pts)
        Inliers = np.arange(n_feature)
        multsig = sci.stats.norm.ppf((1+CI)/2 , 0, 1)
        dist = np.linalg.norm(src_pts - dst_pts, axis=1)
        threshold = np.max(dist)
        stop_counts = 0

        for i in range(residual_trials):
            model, inbool = ski.measure.ransac(
                    (src_pts, dst_pts),
                    model_class, 
                    min_samples=min_samples,
                    residual_threshold=threshold, 
                    max_trials=max_trials,
                    rng=seed,
                    is_data_valid=is_data_valid, 
                    is_model_valid=is_model_valid, 
                    **kargs )

            residuals = model.residuals(src_pts, dst_pts)
            norm_thred = sci.stats.norm.ppf(CI, np.mean(residuals), np.std(residuals))
            sigma_thred = np.mean(residuals) + multsig*np.std(residuals)
            threshold = np.mean([norm_thred, sigma_thred]) *residual_threshold

            n_inliers = np.sum(inbool)
            if verbose>=2:
                print(f'points states: before {len(inbool)} -> after {n_inliers}. {threshold}')

            assert n_inliers >= min_pair, f'nn pairs is lower than {min_pair}.'
            dst_ptn = cv2regist.transform_points( dst_pts, model, inverse=False)
            src_pts = src_pts[inbool]
            dst_pts = dst_ptn[inbool]
            Inliers = Inliers[inbool]

            model_new = model_class()
            model_new.estimate(src_ptw[Inliers], dst_ptw[Inliers])
            merror = np.abs((np.array(model_new)-np.array(model_record)).sum())

            if verbose>=3:
                print(f'model error: {merror}')
            model_record = model_new
            if  (merror <= stop_merror):
                stop_counts += 1
                if (stop_counts >=2):
                    break

        if verbose:
            print(f'ransacn points states: before {n_feature} -> after {len(Inliers)}.')
        model = model_class(dimensionality=2)
        model.estimate(src_ptw[Inliers], dst_ptw[Inliers])
        return [src_ptw[Inliers], dst_ptw[Inliers], Inliers, model]

    @staticmethod
    def RansacFilter(src_pts, dst_pts, model_class=skitf.AffineTransform,
                    min_samples=3, residual_threshold=100, max_trials=2000,
                    is_data_valid=None, is_model_valid=None, use_cv2=False, **kargs
                    ):
        model, inliers = ski.measure.ransac(
                (src_pts, dst_pts),
                model_class, min_samples=min_samples,
                residual_threshold=residual_threshold, max_trials=max_trials,
                is_data_valid=is_data_valid, is_model_valid=is_model_valid, 
                **kargs )
        dist = np.linalg.norm(src_pts - dst_pts, axis=1)
        residuals = np.abs(model.residuals(*(src_pts, dst_pts)))
        n_inliers = np.sum(inliers)
        print(f'points states: before {len(inliers)} -> after {n_inliers}.')

        if use_cv2:
            src_pts_inter = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            dst_pts_inter = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            pairs_match = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            #image3 = cv2.drawMatches(img1, src_pts_inter, img2, dst_pts_inter, pairs_match, None)

            src_pts = np.float64([ src_pts_inter[m.queryIdx].pt for m in pairs_match ]).reshape(-1, 2)
            dst_pts = np.float64([ dst_pts_inter[m.trainIdx].pt for m in pairs_match ]).reshape(-1, 2)
        else:
            src_pts = src_pts[inliers]
            dst_pts = dst_pts[inliers]
        return [src_pts, dst_pts, inliers, model]

    @staticmethod
    def estimate(fix_pts, mov_pts, dimensionality=2, tfmethod=skitf.AffineTransform, transtype='rigid'):
        # TRANS = {
        #     'translation': StackReg.TRANSLATION,
        #     'rigid': StackReg.RIGID_BODY,
        #     'silimarity': StackReg.SCALED_ROTATION,
        #     'affine': StackReg.AFFINE,
        #     'bilinear': StackReg.BILINEAR,
        #     #'calcOpticalFlowFarneback'
        #     #'calcOpticalFlowPyrLK()'
        #     #'calcOpticalFlowPyrLK '
        # }
        # tform = skitf.estimate_transform(transtype,  mov_pts, fix_pts)
        tform = tfmethod(dimensionality=dimensionality)
        tform.estimate(mov_pts, fix_pts)
        return tform

    @staticmethod
    def drawMatches(img1, kp1, img2, kp2, 
                     verify_matches,
                     inliers = None,
                     matchesMask = None,
                     matchesThickness=2,
                     drawpairs=2000,
                     matchColor= (0,255,0),
                     singlePointColor = (255,0,0),
                     flags=0,):
        draw_params = dict(#matchesThickness=matchesThickness,
                            matchColor = matchColor,
                            singlePointColor = singlePointColor,
                            #matchesMask = matchesMask,
                            flags = flags)
        verify_matchesf = []
        if not inliers is None:
            verify_matchesf = [ imh for imh, inl in zip(verify_matches, inliers) if inl]
        else:
            verify_matchesf = verify_matches

        drawpairs = min(len(verify_matchesf), drawpairs)
        imgdm = cv2.drawMatches(img1, kp1, img2, kp2, 
                                     tuple(verify_matchesf[:drawpairs]), 
                                    None, **draw_params)
        #imgdm = cv2.drawMatches(img1, k1, img2, k2, verify_matches, None, (0,0,255), flags=2)
        qview(imgdm)
        return imgdm
