import numpy as np
import collections
import skimage.transform as skitf
import skimage as ski
import scipy as sci
import matplotlib.pyplot as plt
try:
    import cv2
except ImportError:
    pass
try:
    from ..registration._cv2reg import cv2regist
except:
    pass

from cellcloud3d.tools._neighbors import Neighbors
from cellcloud3d.tools._search import searchmatch
from cellcloud3d.plotting._imageview import drawMatches
from cellcloud3d.transform import homotransform_point, homotransform_points, rescales, homotransforms
from cellcloud3d.utilis._arrays import isidentity

class nnalign():
    def __init__(self):
        self.TRANS = {
                'rigid':skitf.EuclideanTransform, #3
                'euclidean':skitf.EuclideanTransform, #3
                'similarity':skitf.SimilarityTransform, #4
                'affine':skitf.AffineTransform, #6
                'projective':skitf.ProjectiveTransform, # 8
                'fund':skitf.FundamentalMatrixTransform,
                'piecewise':skitf.PiecewiseAffineTransform,
            }

    def build(self,
                locs, group, 
                lables = None,
                hData=None, 
                method='hnsw',
                metric='euclidean',
                n_jobs=-1,
                root=None, 
                regist_pair=None,
                full_pair=False,
                step=1,
                showtree=False, 
                keep_self=True,
                sample = None,
                
                **kargs):
        assert locs.shape[0] == len(group)
        if lables is None:
            try:
                self.lables = group.index.values
            except:
                self.lables = np.arange(locs.shape[0])
        else:
            self.lables = lables
        assert locs.shape[0] == len(self.lables)

        self.cellid = np.arange(locs.shape[0])
        try:
            self.order = group.cat.remove_unused_categories().cat.categories
        except:
            self.order = np.unique(group)

        self.align_pair, self.trans_pair = searchmatch.searchidx(len(self.order), 
                                                        root=root,
                                                        regist_pair=regist_pair,
                                                        full_pair=full_pair,
                                                        keep_self=keep_self,
                                                        step=step,
                                                        showtree=showtree)
        idxdict = collections.OrderedDict()
        for igroup in self.order:
            idxdict[igroup] = (group == igroup)
        self.idxdict = idxdict

        kdts = {}
        kdls = {}
        kdds = {}

        for sid, idx in idxdict.items():
            nnr = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
            rdata = locs[idx] if hData is None else hData[idx]
            nnr.fit(rdata, **kargs)

            kdds[sid] = rdata
            kdts[sid] = nnr
            kdls[sid] = self.cellid[idx]
        self.kdds = kdds
        self.kdls = kdls
        self.kdts = kdts
        self.pos = locs
        self.idxdict = idxdict

    def query(self, rsid, qsid, qdata = None, reverse=False, knn=11, return_dist =False, **kargs):
        rlabel = self.kdls[rsid]
        qlabel = self.kdls[qsid]
        qdata = self.kdds[qsid] if qdata is None else qdata
        if reverse:
            qmax = qdata.max()
            qmin = qdata.min()
            rmax = self.kdds[qsid].max()
            rmin = self.kdds[qsid].min()
            qdata = (1 - (qdata - qmin)/(qmax - qmin))*(rmax-rmin) + rmin
        nnr = self.kdts[rsid]
        qnn = nnr.transform(qdata,  knn=knn, **kargs)

        if return_dist:
            return nnr.translabel(qnn[1], rlabel, qlabel, nnvalue=qnn[0])
        else:
            return nnr.translabel(qnn[1], rlabel, qlabel, nnvalue=None)

    def nnmatch(self, rsid, qsid, rdata = None, qdata=None, knn=6,
                cross=True, direct=False, return_dist=False, **kargs):
        if return_dist:
            qrnn = self.query(rsid, qsid, qdata=qdata, knn=knn, return_dist=True, **kargs) #src->dst
            rqnn = self.query(qsid, rsid, qdata=rdata, knn=knn, return_dist=True, **kargs) #src->dst

            if direct:
                cross = False
            else:
                rqnn = { tuple(k[::-1]):v for k,v in rqnn.items() } 

            if cross:
                mnn = set(qrnn.keys()) & set(rqnn.keys())
                mnn = np.array([ [*inn, qrnn[inn]] for inn in mnn ])
            else:
                qrnn.update(rqnn)
                mnn = np.array([ [*k, v] for k,v in qrnn.items() ])
        else:
            qrnn = self.query(rsid, qsid, qdata=qdata, knn=knn, return_dist=False, **kargs)
            rqnn = self.query(qsid, rsid, qdata=rdata, knn=knn, return_dist=False, **kargs)
            if direct:
                cross = False
            else:
                rqnn = [ (v,k) for k,v in rqnn ]

            if cross:
                mnn = set(qrnn) & set(rqnn)
                mnn = np.array(list(mnn))
            else:
                mnn = set(qrnn) | set(rqnn)
                mnn = np.array(list(mnn))
        return mnn

    def negative_self(self, kns=10, seed = None, exclude_edge_index = None):
        nnn_idx = []
        for rsid in self.order:
            rlabel = self.kdls[rsid]
            nnn = self.negative_sampling(rlabel, kns=kns, seed=seed)
            nnn_idx.extend(nnn)
        if not exclude_edge_index is None:
            nnn_idx = list(set(nnn_idx) - set(exclude_edge_index))
        return np.array(nnn_idx)

    def negative_hself(self, edge_index, kns=None, seed = None):
        nnn_idx = []
        for rsid in self.order:
            rlabel = self.kdls[rsid]
            iposidx = np.isin(edge_index[1], rlabel) & np.isin(edge_index[0], rlabel) #src ->dst
            nnn = self.negative_hsampling(edge_index[:, iposidx], rlabel, kns=kns, seed=seed)
            nnn_idx.append(nnn)
        return np.concatenate(nnn_idx, axis=1)

    def pairmnn(self, knn=10, cross=True, return_dist=False, direct=False, **kargs):
        mnn_idx = []
        for i, (ridx, qidx) in enumerate(self.align_pair):
            rsid = self.order[ridx]
            qsid = self.order[qidx]
            imnn = self.nnmatch(rsid, qsid, knn=knn, cross=cross,
                                direct=direct,
                                return_dist=return_dist, **kargs)
            recol = [1,0,2] if return_dist else [1,0]
            imnn = np.vstack([imnn, imnn[:,recol]])
            mnn_idx.append(imnn)
        return np.concatenate(mnn_idx, axis=0)

    def ransacmnn(self, rsid, qsid, rdata = None, qdata=None,
                   model_class='rigid',
                   knn=11,
                   cross=True,
                    min_samples=5, residual_threshold=1., 
                    residual_trials=100, max_trials=500, CI = 0.95,
                    drawmatch=False, verbose=False,
                    line_width=0, line_alpha=0.35,
                    line_sample=None,
                    line_limit=None,
                    size=1,
                    fsize=5,
                    seed=491001,
                    titles = None,
                  **kargs):
        model_class = self.TRANS[model_class] if model_class in self.TRANS else model_class
        if titles is None:
            titles = [rsid, qsid]

        mnnk = self.nnmatch(rsid, qsid, knn=knn,
                            rdata = rdata, qdata=qdata,
                             cross=cross, return_dist=True, **kargs)
        rpos = self.pos[mnnk[:,1].astype(np.int64)]
        qpos = self.pos[mnnk[:,0].astype(np.int64)]

        inliers, model = self.autoresidual(rpos, qpos, model_class,
                                            min_samples=min_samples,
                                            max_trials=max_trials,
                                            CI=CI,
                                            residual_trials=residual_trials,
                                            verbose=verbose,
                                            seed=seed,
                                            residual_threshold=residual_threshold)

        mnnf = mnnk[inliers]
        src_pts = rpos[inliers]
        dst_pts = qpos[inliers]

        rposall = self.pos[self.idxdict[rsid]]
        qposall = self.pos[self.idxdict[qsid]]
        dst_mov = homotransform_point(qposall, model, inverse=False)

        if drawmatch:
            ds3 = homotransform_point(dst_pts, model, inverse=False)
            drawMatches( (src_pts, ds3), bgs =(rposall, dst_mov),
                        line_color = ('r'), ncols=2,
                        pairidx=[(0,1)], fsize=fsize,
                        titles= titles,
                        line_limit=line_limit,
                        line_sample=line_sample,
                        size=size,
                        line_width=line_width, line_alpha=line_alpha)

        return [mnnf, model]

    def cv2mnn(self, rsid, qsid, rdata = None, qdata=None,
                model_class='rigist',  drawmatch=False, 
                min_samples=5, residual_threshold=1., 
                residual_trials=100, max_trials=500,
                line_width=0, line_alpha=0.5, **kargs):
        model_class = self.TRANS[model_class] if model_class in self.TRANS else model_class

        r_pts = self.pos[self.idxdict[rsid]]
        q_pts = self.pos[self.idxdict[qsid]]
        rdata = self.kdds[rsid] if rdata is None else rdata
        qdata = self.kdds[qsid] if qdata is None else qdata

        src_pts = [cv2.KeyPoint(point[0], point[1], 1) for point in r_pts]
        dst_pts = [cv2.KeyPoint(point[0], point[1], 1) for point in q_pts]
        verify_matches, matchesMask, src_pts, dst_pts = cv2regist.matchers(
                        src_pts, rdata, dst_pts, qdata,
                        # nomaltype=cv2.NORM_L2, 
                        # method='knn',
                        # drawpoints=2000,
                        # min_matches = 8,
                        # verify_ratio = 1,
                        # reprojThresh = 5.0,
                        # feature_method = None,
                        # table_number = 6, # 12
                        # key_size = 12, # 20
                        # multi_probe_level = 1,
                        # knn_k = 2,
                        # trees=5,
                        # checks=50,
                        # verbose=True,
                        **kargs)

        rmatchidx = [m.queryIdx for m in verify_matches]
        qmatchidx = [m.trainIdx for m in verify_matches]
        mnnk = np.array([rmatchidx, qmatchidx]).T

        inliers, model  = self.autoresidual(src_pts, dst_pts, model_class,
                                            min_samples=min_samples,
                                            max_trials=max_trials,
                                            residual_trials=residual_trials,
                                            residual_threshold=residual_threshold)
        mnnf = mnnk[inliers]
        src_pts = src_pts[inliers]
        dst_pts = dst_pts[inliers]
        if drawmatch:
            bg1 = self.pos[self.idxdict[rsid]]
            bg2 = self.pos[self.idxdict[qsid]]
            bg3 = homotransform_point(bg2, model, inverse=False)
            ds3 = homotransform_point(dst_pts, model, inverse=False)

            drawMatches( (src_pts, dst_pts, ds3), bgs =(bg1, bg2, bg3),
                        line_color = ('r', 'green'), ncols=2,
                        pairidx=[(0,1),(0,2)], fsize=5,
                        titles=['sorce', 'dest', 'mov'],
                        line_width=line_width, line_alpha=line_alpha)

        return [mnnf, model]

    def regist(self, knn=11, method='rigid', cross=True, CIs = 0.93, 
               broadcast = True, 
               drawmatch=False, line_width=0.5, line_alpha=0.5, **kargs):
        if isinstance(CIs, float):
            CIs = [CIs] * len(self.align_pair)
        else:
            assert isinstance(CIs, list) or isinstance(CIs, np.ndarray)
        assert len(CIs) == len(self.align_pair)

        if isinstance(method, str):
            method = [method] * len(self.align_pair)
        else:
            assert isinstance(method, list)
        assert len(method) == len(self.align_pair)

        tforms = [np.identity(3)] * len(self.order)
        mnnfs = []
        for i, (ridx, qidx) in enumerate(self.align_pair):
            rsid = self.order[ridx]
            qsid = self.order[qidx]
            mnnf, model = self.ransacmnn(rsid, qsid, 
                                         drawmatch=drawmatch, 
                                         model_class=method[i],
                                         CI=CIs[i],
                                         cross=cross,
                                         knn=knn,
                                         line_width=line_width,
                                         line_alpha=line_alpha,
                                         **kargs)
            tforms[qidx] = model
            mnnfs.append(mnnf)
        self.tforms = self.update_tmats(self.trans_pair, tforms) if broadcast else tforms

    def transform_points(self, moving=None, tforms=None, inverse=False):
        tforms = self.tforms if tforms is None else tforms
        if moving is None:
            mov_out = []
            mov_idx = []
            for i,sid in enumerate(self.order):
                itform = tforms[i]
                icid = self.kdls[sid]
                iloc = self.pos[icid]
                nloc = homotransform_point(iloc, itform, inverse=inverse)
                mov_out.append(nloc)
                mov_idx.append(icid)
            mov_out = np.concatenate(mov_out, axis=0)
            mov_idx = np.concatenate(mov_idx, axis=0)
            mov_out = mov_out[np.argsort(mov_idx)] #[self.cellid]
        else:
            mov_out = homotransform_points(moving, tforms, inverse=inverse)
        return mov_out

    def transform_images(self,
                   images = None, 
                   tforms =None,
                    scale=None,
                    trans_name = 'skimage'):
        if images is None:
            return images

        tforms = self.tforms if tforms is None else tforms
        images = images if scale is None else rescales(images, scale)

        if trans_name=='skimage':
            mov_imgs = []
            imgtype = images.dtype
            for i in range(images.shape[0]):
                if isidentity(tforms[i]):
                    iimg = images[i]
                else:
                    iimg = ski.transform.warp(images[i], tforms[i])
                    # if np.issubdtype(imgtype, np.integer):
                    #     iimg = np.clip(np.round(iimg * 255), 0, 255).astype(np.uint8)
                mov_imgs.append(iimg)
            mov_imgs = np.array(mov_imgs)
        return mov_imgs

    @staticmethod
    def update_tmats(trans_pair, raw_tmats):
        try:
            new_tmats = raw_tmats.copy()
        except:
            new_tmats = [ imt.copy() for imt in raw_tmats]

        for ifov, imov in trans_pair:
            new_tmats[imov] = np.matmul(new_tmats[imov], new_tmats[ifov])
        return new_tmats

    @staticmethod
    def split(groups):
        try:
            Order = groups.cat.remove_unused_categories().cat.categories
        except:
            Order = np.unique(groups)

        idxdict = collections.OrderedDict()
        for igroup in Order:
            idxdict[igroup] = (groups == igroup)
        return idxdict

    @staticmethod
    def negative_sampling(labels, kns=10, seed = None, exclude_edge_index = None):
        n_nodes = len(labels)
        rng = np.random.default_rng(seed=seed)
        idx = rng.integers(0, high = n_nodes, size=[n_nodes,kns])
        nnn = [ (labels[v], labels[k]) for k in range(n_nodes) for v in idx[k]] #src->dst
        if not exclude_edge_index is None:
            nnn = list(set(nnn) - set(exclude_edge_index))
        else:
            nnn = list(set(nnn))
        return (nnn)

    @staticmethod
    def negative_hsampling( edge_index, labels, kns=None, seed = 200504):
        pos_lab, counts = np.unique(edge_index[1],return_counts=True)  #src -> dst
        nev_set = []
        seed = [seed] if isinstance(seed, int) else seed
        for i in range(len(pos_lab)):
            ipos = pos_lab[i]
            isize = kns or counts[i]
            iset =  edge_index[0][edge_index[1] == ipos]
            nevs =  list(set(labels) - set(iset) -set([ipos]) )
            rng = np.random.default_rng(seed=[i, *seed])
            inev = rng.choice(nevs, size=isize, replace=False, shuffle=False)
            nev_set.append(inev)
        nev_set= np.concatenate(nev_set, axis=0)
        if kns is None:
            src_set= edge_index[1]
        else:
            src_set= np.repeat(pos_lab, kns)
        neg_sam = np.array([src_set, nev_set])
        return (neg_sam)

    @staticmethod
    def autoresidual(src_pts, dst_pts, model_class,
                    min_samples=5, residual_threshold =None, 
                    residual_trials=100, max_trials=500,
                    seed=200504, CI=0.95, stop_merror=1e-3,
                    min_pair=10,
                    is_data_valid=None, is_model_valid=None,
                    drawhist=False, verbose=False, **kargs):

        src_ptw = src_pts.copy()
        dst_ptw = dst_pts.copy()

        if not residual_threshold is None:
            assert 0<residual_threshold <=1
        model_record = np.eye(3)

        Inliers = np.arange(len(src_pts))
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

            if drawhist:
                fig, ax=plt.subplots(1,1, figsize=(5,5))
                ax.hist(residuals, bins=50)
                ax.axvline(norm_thred, color='red')
                ax.axvline(sigma_thred, color='blue')
                ax.axvline(threshold, color='black')
                plt.show()

            n_inliers = np.sum(inbool)
            if verbose:
                print(f'points states: before {len(inbool)} -> after {n_inliers}. {threshold}')

            assert n_inliers >= min_pair, f'nn pairs is lower than {min_pair}.'
            dst_ptn = homotransform_point( dst_pts, model, inverse=False)
            src_pts = src_pts[inbool]
            dst_pts = dst_ptn[inbool]
            Inliers = Inliers[inbool]

            model_new = model_class()
            model_new.estimate(src_ptw[Inliers], dst_ptw[Inliers])
            merror = np.abs((np.array(model_new)-np.array(model_record)).sum())

            if verbose:
                print(f'model error: {merror}')
            model_record = model_new
            if  (merror <= stop_merror):
                stop_counts += 1
                if (stop_counts >=2):
                    break

        print(f'ransacn nn pairs: {len(Inliers)}')
        model = model_class(dimensionality=2)
        model.estimate(src_ptw[Inliers], dst_ptw[Inliers])
        return Inliers, model

    @staticmethod
    def icp_transform( A, B):
        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t