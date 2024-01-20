from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import numpy as np
from scipy.sparse import issparse, csr_array
from sklearn.neighbors import NearestNeighbors as sknn
from annoy import AnnoyIndex
try:
    from cuml.neighbors import NearestNeighbors as cumlnn
except:
    pass
try:
    from pynndescent import NNDescent
except:
    pass
try:
    import hnswlib
except ImportError:
    pass
try:
    import faiss
except ImportError:
    pass
try:
    import ot
except ImportError:
    pass
import random

class Neighbors():
    def __init__(self, method='annoy',
                  metric='euclidean',
                  n_jobs=-1):
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, data, hnsw_space='l2',
            seed=200504,
            max_elements=None, ef_construction=200, M=20,
            annoy_n_trees=70, pynndescent_n_neighbors=50):
        np.random.seed(seed)
        random.seed(seed)
        data_labels = np.arange(data.shape[0])
        if self.method == 'hnsw':
            ckd = hnswlib.Index(space=hnsw_space, dim=data.shape[1])
            ckd.init_index(max_elements = data.shape[0] if max_elements is None else max_elements, 
                            ef_construction = ef_construction, M = M,  random_seed =seed)
            ckd.add_items(data, data_labels, num_threads = self.n_jobs)

        elif self.method == 'annoy':
            ckd = AnnoyIndex(data.shape[1], metric=self.metric)
            ckd.set_seed(seed)
            for i in np.arange(data.shape[0]):
                ckd.add_item(i,data[i,:])
            ckd.build(annoy_n_trees)

        elif self.method == 'sknn':
            ckd= sknn

        elif  self.method == 'faiss':
            ckd = faiss.IndexFlatL2(data.shape[1])
            ckd.add(data)

        elif  self.method == 'pynndescent':
            ckd = NNDescent(data, metric=self.metric,
                            n_jobs=self.n_jobs,
                            n_neighbors=pynndescent_n_neighbors, 
                            random_state=seed)
            ckd.prepare()

        elif  self.method == 'cKDTree':
            ckd = cKDTree(data)

        elif  self.method == 'KDTree':
            ckd = KDTree(data,metric=self.metric)
        self.ckd = ckd

    def transform(self, data, ckd=None, knn=20, set_ef=60, 
                  search_k=-1, sort_dist=True,
                  include_distances=True):
        ckd = self.ckd if ckd is None else ckd
        if self.method == 'hnsw':
            ckd.set_ef(max(set_ef, knn+10)) #ef should always be > k
            ckd.set_num_threads(self.n_jobs)
            labels, distances = ckd.knn_query(data, k = knn, num_threads=self.n_jobs)
            ckdout = (np.sqrt(distances), labels) if ckd.space =='l2' else (distances, labels)

        elif self.method == 'annoy':
            ckdo_ind = []
            ckdo_dist = []
            for i in np.arange(data.shape[0]):
                holder = ckd.get_nns_by_vector(data[i,:],
                                                knn,
                                                search_k=search_k,
                                                include_distances=include_distances)
                ckdo_ind.append(holder[0])
                ckdo_dist.append(holder[1])
            ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))

        elif self.method == 'sknn':
            nbrs = ckd(n_neighbors=knn, p=2, algorithm='auto', metric='minkowski')
            nbrs.fit(data)
            distances, indices = nbrs.kneighbors(data, knn, return_distance=True)
            ckdout = (distances, indices)

        elif self.method == 'pynndescent':
            ckdout = ckd.query(data, k=knn)
            ckdout = (ckdout[1], ckdout[0])
        elif self.method == 'faiss':
            D, I = ckd.search(data, knn)
            D[D<0] = 0
            ckdout = (np.sqrt(D), I)
        elif self.method == 'cKDTree':
            ckdout = ckd.query(x=data, k=knn, p=2, workers=self.n_jobs)
        elif self.method == 'KDTree':
            ckdout = ckd.query(data, k=knn)

        if sort_dist and ((ckdout[0][:,1:] - ckdout[0][:,:-1]).min()<0):
            idxsort = ckdout[0].argsort(axis=1)
            ckdout[0] = np.take_along_axis(ckdout[0], idxsort, 1)
            ckdout[1] = np.take_along_axis(ckdout[1], idxsort, 1)
        return ckdout

    @staticmethod
    def translabel(ckdout, rlabel=None, qlabel=None, rsize=None, return_type='raw'):
        nnidx = ckdout[1]
        minrnum = nnidx.max() +1
        if not rlabel is None:
            assert len(rlabel) >= minrnum
        if not rsize is None:
            assert rsize >= minrnum
        if not qlabel is None:
            assert len(qlabel) == nnidx.shape[0]
        
        if return_type == 'raw':
            return [ckdout[0], nnidx]
        elif (return_type == 'arrays') and (not rlabel is None):
            rlabel = np.asarray(rlabel)
            return [ckdout[0], rlabel[nnidx]]
        elif return_type in ['lists', 'sparse', 'sparseidx']:
            src = nnidx.flatten('C')
            dst = np.repeat(np.arange(nnidx.shape[0]), nnidx.shape[1])
            dist = ckdout[0].flatten('C')

            if return_type in ['sparse', 'sparseidx']:
                rsize = rsize or (None if rlabel is None else len(rlabel)) or minrnum
                if return_type == 'sparseidx':
                    dist = np.ones_like(dst)
                cdkout = csr_array((dist, (dst, src)), shape=(nnidx.shape[0], rsize))
                if not cdkout.has_sorted_indices:
                    cdkout.sort_indices()
                return cdkout
            else:
                if not rlabel is None:
                    src = np.asarray(rlabel)[src]
                if not qlabel is None:
                    dst = np.asarray(qlabel)[dst]
                return [src, dst, dist]
        else:
            raise ValueError('return_type must be one of "raw", "arrays", "lists", "sparse", "sparseidx"')