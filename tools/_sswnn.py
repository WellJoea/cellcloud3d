import numpy as np
import collections
import scipy as sci
import matplotlib.pyplot as plt
from scipy.sparse import csr_array
from typing import List, Optional
from joblib import Parallel, delayed

from cellcloud3d.preprocessing._decomposition import dual_pca
from cellcloud3d.tools._neighbors import Neighbors
from cellcloud3d.tools._search import searchmatch
from cellcloud3d.transform import homotransform_point, homotransform_points, rescales, homotransforms
from cellcloud3d.utilis._arrays import isidentity
from cellcloud3d.plotting._imageview import drawMatches
from cellcloud3d.tools._spatial_edges import similarity_mtx

class SSWNN():
    def __init__(self):
        self.similarity_mtx = similarity_mtx

    def build(self,
                latent, groups,
                splocs=None, 
                lables = None,
                levels = None,

                dpca_npca = 50,
                method='annoy',
                spmethod='sknn',
                metric='euclidean',
                n_jobs=-1,
                root=None, 
                regist_pair=None,
                full_pair=False,
                step=1,
                showtree=False, 
                keep_self=True,

                **kargs):
        assert latent.shape[0] == len(groups)
        if lables is None:
            try:
                self.lables = groups.index.values
            except:
                self.lables = np.arange(latent.shape[0])
        else:
            self.lables = lables
        assert latent.shape[0] == len(self.lables)
        
        if not splocs is None:
            assert latent.shape[0] == splocs.shape[0]

        self.cellid = np.arange(latent.shape[0])
        if levels is None:
            try:
                self.order = groups.cat.remove_unused_categories().cat.categories
            except:
                self.order = np.unique(groups)
        else:
            self.order = levels

        self.align_pair, self.trans_pair = searchmatch.searchidx(len(self.order), 
                                                        root=root,
                                                        regist_pair=regist_pair,
                                                        full_pair=full_pair,
                                                        keep_self=keep_self,
                                                        step=step,
                                                        showtree=showtree)
        groupidx = collections.OrderedDict()
        for igroup in self.order:
            groupidx[igroup] = [groups == igroup, self.cellid[(groups == igroup)]]
        self.groupidx = groupidx

        self.latent = latent
        self.splocs = splocs

        enns = {}
        for sid in self.order:
            idx = self.groupidx[sid][0]
            enn = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
            enn.fit(self.latent[idx], **kargs)
            enns[sid] = enn
        self.enns = enns

        if not self.splocs is None:
            snns = {}
            for sid in self.order:
                idx = self.groupidx[sid][0]
                snn = Neighbors(method=spmethod, metric=metric, n_jobs=n_jobs)
                snn.fit(self.splocs[idx], **kargs)
                snns[sid] = snn
            self.snns = snns
        else:
            self.snns = None

        self.largs = dict(method=method, metric=metric, n_jobs=n_jobs, 
                          dpca_npca=dpca_npca, kargs=kargs)

    def kdtree_dpca(self, rsid, qsid, qdata = None, knn= 11, 
                    dpca_npca=None, dpca_scale=True, **kargs):
        ridx, rlabel = self.groupidx[rsid]
        qidx, qlabel = self.groupidx[qsid]

        rdata = self.latent[ridx]
        qdata = self.latent[qidx] if qdata is None else qdata
        rdata, qdata = dual_pca(rdata, qdata, 
                                n_comps = self.largs.get('dpca_npca', dpca_npca),
                                scale=self.largs.get('dpca_scale', dpca_scale),
                                copy=True, axis=0,
                                zero_center=True)
        kdnn = Neighbors(method=self.largs.get('method', 'annoy'), 
                         metric=self.largs.get('metric', 'euclidean'), 
                         n_jobs=self.largs.get('n_jobs', -1))

        kdnn.fit(rdata, **self.largs.get('kargs', {}))
        cdkout = kdnn.transform(qdata, knn=knn, **kargs)
        return kdnn, cdkout

    def query(self, rsid, qsid, slot = 'enn', qdata = None,
               use_dpca=True, dpca_npca=50,  knn=11, return_type='raw', **kargs):
        ridx, rlabel = self.groupidx[rsid]
        qidx, qlabel = self.groupidx[qsid]

        if use_dpca and (slot == 'enn'):
            kdnn, cdkout = self.kdtree_dpca( rsid, qsid, qdata = qdata, dpca_npca=dpca_npca, knn= knn, **kargs)
        elif slot == 'enn':
            kdnn = self.enns[rsid]
            qdata = self.latent[qidx] if qdata is None else qdata
            cdkout = kdnn.transform(qdata, knn=knn, **kargs)
        elif slot == 'snn':
            kdnn = self.snns[rsid]
            qdata = self.splocs[qidx] if qdata is None else qdata
            cdkout = kdnn.transform(qdata, knn=knn, **kargs)

        return kdnn.translabel(cdkout, rlabel=rlabel, qlabel=qlabel, return_type=return_type)

    def simi_pair(self, rsid, qsid, method = 'cosine', pairidx = None):
        ridx, _ = self.groupidx[rsid]
        qidx, _ = self.groupidx[qsid]

        rdata = self.latent[ridx]
        qdata = self.latent[qidx]

        return self.similarity_mtx(rdata, qdata, method=method, pairidx=pairidx)

    def selfsnns(self, o_neighbor = 60, s_neighbor =30, show_simi = False):
        rrnns = {}
        for sid in self.order:
            if self.splocs is None:
                rrnn = self.query(sid, sid, slot='enn', knn=o_neighbor+1, return_type='sparseidx', use_dpca=False)
            else:
                if o_neighbor and s_neighbor:
                    rrenn = self.query(sid, sid, slot='enn', knn=o_neighbor+1, return_type='sparseidx', use_dpca=False)
                    rrsnn = self.query(sid, sid, slot='snn', knn=s_neighbor+1, return_type='sparseidx')
                    rrnn = rrenn.multiply(rrsnn)
                elif (not o_neighbor) and s_neighbor:
                    rrnn = self.query(sid, sid, slot='snn', knn=s_neighbor+1, return_type='sparseidx')
                elif (not s_neighbor) and o_neighbor:
                    rrnn = self.query(sid, sid, slot='enn', knn=o_neighbor+1, return_type='sparseidx', use_dpca=False)

            if show_simi:
                mridx, mqidx = rrnn.nonzero()
                simis  = self.simi_pair(sid, sid, pairidx=[mridx, mqidx])
                import matplotlib.pylab as plt
                fig, ax = plt.subplots(1,3, figsize=(12,4))
                ax[0].hist(rrnn.sum(0), bins=s_neighbor, facecolor='b', label = f'{np.mean(rrnn.sum(0))}')
                ax[1].hist(rrnn.sum(1), bins=s_neighbor, facecolor='b', label = f'{np.mean(rrnn.sum(1))}')
                ax[2].hist(simis, bins=100, facecolor='b', label = f'{np.mean(simis)}')
                plt.show()
            rrnns[sid] = rrnn
        return rrnns

    def swnnscore(self, ssnn, n_neighbor = None, lower = 0.01, upper = 0.9):
        mhits = ssnn.data

        mhits = np.sqrt(mhits)
        n_neighbor = min(n_neighbor or max(mhits), max(mhits))
        mhits /= n_neighbor

        min_score = np.quantile(mhits, lower)
        max_score = np.quantile(mhits, upper)
        mhits = (mhits-min_score)/(max_score-min_score)
        mhits  = np.clip(mhits, 0, 1)
        return mhits

    def swmnn(self, rsid, qsid, rrnn=None, qqnn = None, m_neighbor=6, e_neighbor =30, 
              s_neighbor =30, lower = 0.01, upper = 0.9, 
              use_dpca=True, **kargs):
        qrnna = self.query(rsid, qsid, slot='enn', knn=e_neighbor, return_type='raw', use_dpca=use_dpca, sort_dist=True, **kargs) #shape[0] == q
        rqnna = self.query(qsid, rsid, slot='enn', knn=e_neighbor, return_type='raw', use_dpca=use_dpca, sort_dist=True, **kargs) #shape[0] == r

        if rrnn is None:
            rrnn = self.query(rsid, rsid, use_dpca=False, slot='snn', knn=s_neighbor+1, return_type='sparseidx', **kargs)
        if qqnn is None:
            qqnn = self.query(qsid, qsid, use_dpca=False, slot='snn', knn=s_neighbor+1, return_type='sparseidx', **kargs)

        qrnn = Neighbors.translabel(qrnna, rsize=rrnn.shape[0], return_type='sparseidx')
        rqnn = Neighbors.translabel(rqnna, rsize=qqnn.shape[0], return_type='sparseidx')

        qrmnn = [qrnna[0][:,:m_neighbor], qrnna[1][:,:m_neighbor] ]
        qrmnn = Neighbors.translabel(qrmnn, rsize=rrnn.shape[0], return_type='sparseidx')
        rqmnn = [rqnna[0][:,:m_neighbor], rqnna[1][:,:m_neighbor] ]
        rqmnn = Neighbors.translabel(rqmnn, rsize=qqnn.shape[0], return_type='sparseidx')

        ssnn = (rqnn.dot(qqnn.transpose())).multiply(rrnn.dot(qrnn.transpose()))
        mnn = rqmnn.multiply(qrmnn.transpose())
        ssnn = ssnn.multiply(mnn) #
        if not ssnn.has_sorted_indices:
            ssnn.sort_indices()

        mridx, mqidx = ssnn.nonzero()
        mhits = self.swnnscore(ssnn, n_neighbor=min(e_neighbor, s_neighbor),
                               lower = lower, upper = upper)
        keepidx = mhits > 0 
        mridx = mridx[keepidx].astype(np.int64)
        mqidx = mqidx[keepidx].astype(np.int64)
        mhits = mhits[keepidx].astype(np.float32)

        ssnn = csr_array((mhits, (mridx, mqidx)), shape=ssnn.shape)
        ssnn.sort_indices()
        ssnn.eliminate_zeros()
        return ssnn

    def swmnns(self, m_neighbor=6, e_neighbor =30, s_neighbor =30, 
                o_neighbor = 50,
                use_dpca=True, drawmatch =False, 
                line_width=0.1, line_alpha=0.35,
                line_sample=None,
                line_limit=None,
                size=1,
                fsize=7,
                verbose=0,
                merge_edges=True,
            **kargs):

        rrnns = self.selfsnns(s_neighbor = s_neighbor, o_neighbor=o_neighbor)
        paris = []
        scores = []
        for i, (rid, qid) in enumerate(self.align_pair):
            if verbose>=1:
                print(f'match: {rid} -> {qid}')
            rsid = self.order[rid]
            qsid = self.order[qid]
            ridx, rlabel = self.groupidx[rsid]
            qidx, qlabel = self.groupidx[qsid]

            ssnn = self.swmnn(rsid, qsid, rrnn=rrnns[rsid], qqnn = rrnns[qsid], 
                              use_dpca = use_dpca,
                              m_neighbor=m_neighbor, e_neighbor =e_neighbor, 
                              s_neighbor =s_neighbor, **kargs)

            mridx, mqidx = ssnn.nonzero()
            if verbose>=1:
                print( f'match pairs: {mridx.shape[0]}' )

            if drawmatch and (not self.splocs is None):
                rposall = self.splocs[ridx]
                qposall = self.splocs[qidx]
                drawMatches( (rposall[mridx], qposall[mqidx]),
                            bgs =(rposall, qposall),
                            line_color = ('r'), ncols=2,
                            pairidx=[(0,1)], fsize=fsize,
                            titles= [rsid, qsid],
                            line_limit=line_limit,
                            line_sample=line_sample,
                            size=size,
                            line_width=line_width, line_alpha=line_alpha)

            mhits = ssnn.data
            if merge_edges:
                mridx = rlabel[mridx]
                mqidx = qlabel[mqidx]
            paris.append(np.array([mridx, mqidx]).astype(np.int64))
            scores.append(mhits.astype(np.float32))

        if merge_edges:
            paris = np.concatenate(paris, axis=1)
            scores = np.concatenate(scores, axis=0)
        return [paris, scores]

    def nnmatch(self, rsid, qsid, knn=6, **kargs):

        qrnn = self.query(rsid, qsid, knn=knn, return_type='lists', **kargs)
        rqnn = self.query(qsid, rsid, knn=knn, return_type='lists', **kargs)
        rqnn = zip(rqnn[1], rqnn[0])
        qrnn = zip(qrnn[0], qrnn[1])

        mnn = set(qrnn) & set(rqnn)
        mnn = np.array(list(mnn))

        return mnn

    def negative_self(self, kns=10, seed = None, exclude_edge_index = None):
        nnn_idx = []
        for rsid in self.order:
            ridx, rlabel = self.groupidx[rsid]
            nnn = self.negative_sampling(rlabel, kns=kns, seed=seed)
            nnn_idx.extend(nnn)
        if not exclude_edge_index is None:
            nnn_idx = list(set(nnn_idx) - set(exclude_edge_index))
        return np.array(nnn_idx)

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

def findmatches( hData, groups, position=None,
                ckd_method='annoy', sp_method = 'sknn',
                root=None, regist_pair=None, full_pair=False, step=1,

                use_dpca = True, dpca_npca = 60,
                m_neighbor=6, e_neighbor =30, s_neighbor =30,
                o_neighbor = 30,
                lower = 0.01, upper = 0.9,

                point_size=1,
                drawmatch=False,  line_sample=None,
                merge_edges=True,
                line_width=0.5, line_alpha=0.5, line_limit=None,**kargs):

    ssnn = SSWNN()
    ssnn.build(hData, groups,
                splocs=position,
                method=ckd_method,
                spmethod=sp_method,
                root=root,
                regist_pair=regist_pair,
                step=step,
                full_pair=full_pair)
    paris, scores = ssnn.swmnns(m_neighbor=m_neighbor,
                                e_neighbor =e_neighbor,
                                s_neighbor =s_neighbor,
                                o_neighbor =o_neighbor,
                                use_dpca = use_dpca,
                                dpca_npca = dpca_npca,
                                lower = lower,
                                upper = upper,
                                drawmatch=drawmatch,
                                line_width=line_width,
                                line_alpha=line_alpha,
                                line_limit=line_limit,
                                line_sample=line_sample,
                                fsize=4,
                                size=point_size,
                                verbose=0,
                                merge_edges = merge_edges,
                                **kargs)
    return [paris, scores]