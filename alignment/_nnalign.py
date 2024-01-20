import numpy as np
import collections
import skimage.transform as skitf
import skimage as ski
import scipy as sci
import matplotlib.pyplot as plt

from cellcloud3d.tools._neighbors import Neighbors
from cellcloud3d.tools._search import searchmatch
from cellcloud3d.tools._sswnn import SSWNN

from cellcloud3d.plotting._imageview import drawMatches
from cellcloud3d.transform import homotransform_point, homotransform_points, rescales, homotransforms
from cellcloud3d.utilis._arrays import isidentity

class nnalign(SSWNN):
    def __init__(self, *arg, **kargs):
        super().__init__(*arg, **kargs)
        self.TRANS = {
                'rigid':skitf.EuclideanTransform, #3
                'euclidean':skitf.EuclideanTransform, #3
                'similarity':skitf.SimilarityTransform, #4
                'affine':skitf.AffineTransform, #6
                'projective':skitf.ProjectiveTransform, # 8
                'fund':skitf.FundamentalMatrixTransform,
                'piecewise':skitf.PiecewiseAffineTransform,
            }

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

    def ransacmnn(self, rsid, qsid,
                   model_class='rigid',
                   m_neighbor=6, e_neighbor =30, s_neighbor =30,
                   lower = 0.01, upper = 0.9,
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
                    hide_axis=False,
                    equal_aspect=True,
                    invert_xaxis=False,
                    invert_yaxis=False,
                    line_color = ('r'), 
                    ncols=2,
                    pargs={},
                  **kargs):
        model_class = self.TRANS[model_class] if model_class in self.TRANS else model_class
        if titles is None:
            titles = [rsid, qsid]

        # mnnk = self.nnmatch(rsid, qsid, knn=knn,
        #                     rdata = rdata, qdata=qdata,
        #                      cross=cross, return_dist=True, **kargs)
        print(f'Match pairs: {rsid} <-> {qsid}')
        print('Match Features...')
        ssnn = self.swmnn(rsid, qsid, rrnn=self.rrnns[rsid], 
                            qqnn = self.rrnns[qsid], 
                            m_neighbor=m_neighbor, e_neighbor =e_neighbor, 
                            s_neighbor =s_neighbor,lower = lower, upper = upper,
                             **kargs)
        mridx, mqidx = ssnn.nonzero()
        mscore = ssnn.data
        ridx, rlabel = self.groupidx[rsid]
        qidx, qlabel = self.groupidx[qsid]
        mridx = rlabel[mridx]
        mqidx = qlabel[mqidx]

        rpos = self.splocs[mridx]
        qpos = self.splocs[mqidx]
        print('Compute Transformation...')
        inliers, model = self.autoresidual(rpos, qpos, model_class,
                                            min_samples=min_samples,
                                            max_trials=max_trials,
                                            CI=CI,
                                            residual_trials=residual_trials,
                                            verbose=verbose,
                                            seed=seed,
                                            residual_threshold=residual_threshold)
        src_pts = rpos[inliers]
        dst_pts = qpos[inliers]

        rposall = self.splocs[ridx]
        qposall = self.splocs[qidx]
        dst_mov = homotransform_point(qposall, model, inverse=False)

        keepidx = np.zeros_like(mridx)
        keepidx[inliers] = 1
        anchors = np.vstack([mridx, mqidx, keepidx, mscore]).T

        if drawmatch:
            ds3 = homotransform_point(dst_pts, model, inverse=False)
            drawMatches( (src_pts, ds3), bgs =(rposall, dst_mov),
                        line_color = line_color, ncols=ncols,
                        pairidx=[(0,1)], fsize=fsize,
                        titles= titles,
                        line_limit=line_limit,
                        line_sample=line_sample,
                        size=size,
                        equal_aspect = equal_aspect,
                        hide_axis=hide_axis,
                        invert_xaxis=invert_xaxis,
                        invert_yaxis=invert_yaxis,
                        line_width=line_width,
                        line_alpha=line_alpha,
                        **pargs)
        return model, anchors

    def regist(self, m_neighbor=6, e_neighbor =30, s_neighbor =30,
               method='rigid', CIs = 0.95, o_neighbor = 60, 
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

        self.rrnns = self.selfsnns(o_neighbor = o_neighbor, s_neighbor=s_neighbor)
        tforms = [np.identity(3)] * len(self.order)
        self.matches = {}
        for i, (ridx, qidx) in enumerate(self.align_pair):
            rsid = self.order[ridx]
            qsid = self.order[qidx]
            model, anchors = self.ransacmnn(rsid, qsid, 
                                         drawmatch=drawmatch, 
                                         model_class=method[i],
                                         CI=CIs[i],
                                         m_neighbor = m_neighbor, 
                                         e_neighbor = e_neighbor, 
                                         s_neighbor = s_neighbor,
                                         line_width=line_width,
                                         line_alpha=line_alpha,
                                         **kargs)
            tforms[qidx] = model
            self.matches[(rsid, qsid)] = anchors

        self.tforms = self.update_tmats(self.trans_pair, tforms) if broadcast else tforms

    def transform_points(self, moving=None, tforms=None, inverse=False):
        tforms = self.tforms if tforms is None else tforms
        if moving is None:
            mov_out = []
            mov_idx = []
            for i,sid in enumerate(self.order):
                itform = tforms[i]
                icid = self.groupidx[sid][1]
                iloc = self.splocs[icid]

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
                    min_samples=5, residual_threshold =1., 
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

def aligner( hData, groups,  position,
                ckd_method='annoy', sp_method = 'sknn',
                use_dpca = True,
                dpca_npca = 50,
                root=None, regist_pair=None, full_pair=False, step=1,
                m_neighbor= 6, 
                e_neighbor= 30,
                s_neighbor= 30,
                o_neighbor = 30, 
                lower = 0.01,
                reg_method = 'rigid', point_size=1,
                CIs = 0.93, 
                drawmatch=False,  line_sample=None,
                line_width=0.5, line_alpha=0.5, line_limit=None,**kargs):
    mnnk = nnalign()
    mnnk.build(hData, groups,
                splocs=position,
                method=ckd_method,
                spmethod=sp_method,
                dpca_npca = dpca_npca,
                root=root,
                regist_pair=regist_pair,
                step=step,
                full_pair=full_pair)
    mnnk.regist(m_neighbor=m_neighbor,
                e_neighbor =e_neighbor,
                s_neighbor =s_neighbor,
                o_neighbor = o_neighbor,
                use_dpca = use_dpca,
                lower = lower,
                method=reg_method,
                CIs = CIs,
                broadcast = True,
                drawmatch=drawmatch,
                line_width=line_width,
                line_alpha=line_alpha,
                line_limit=line_limit,
                line_sample=line_sample,
                fsize=4,
                size=point_size,
                **kargs)
    tforms = mnnk.tforms
    new_pos = mnnk.transform_points()
    Order = mnnk.order
    matches = mnnk.matches
    return [tforms, new_pos, matches, Order]