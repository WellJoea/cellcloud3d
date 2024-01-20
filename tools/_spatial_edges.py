import anndata as ad
import matplotlib.pyplot as plt
from skimage import filters
import collections
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors as sknn
import scipy as sci
from scipy.sparse import issparse, csr_array
from cellcloud3d.tools._neighbors import Neighbors

def spatial_edges(adata, groupby, basis='spatial',
                add_key=None,
                merge_edges=False,
                n_neighbors=10,
                radius = None,
                radiu_trim='infer',
                return_dsimi= False,
                show_hist = True,
                simi_scale = 'infer',
                dsimi_method = 'exp1',
                remove_loop=False,
                infer_thred=0.99, 

                return_esimi = True,
                esimi_method = 'cosine',
                esimi_thred = 0.1,
                remove_lowesimi = False,
                show_exp = True,
                doscale=False, 
                dopca=False, 
                n_pcs=100, 
                max_value=10,

                use_name =False,
                verbose = True,
                n_jobs=-1,
                **kargs):

    add_key = add_key or f'{basis}_edges'
    if use_name:
        cellidx = adata.obs_names.values
    else:
        cellidx = np.arange(adata.shape[0]).astype(np.int64)

    try:
        groups = adata.obs[groupby].cat.remove_unused_categories().cat.categories
    except:
        groups = adata.obs[groupby].unique()

    if isinstance(n_neighbors, int) or (n_neighbors is None):
        n_neighbors = [n_neighbors] * len(groups)
    
    if type(radius) in [int, float] or (radius is None):
        radius = [radius] * len(groups)

    if not isinstance(radiu_trim, list):
        radiu_trim = [radiu_trim] * len(groups)

    if type(esimi_thred) in [int, float] or (esimi_thred is None):
        esimi_thred = [esimi_thred] * len(groups)

    edges, edges_attr = [], []
    for i, igrp in enumerate(groups):
        idx = (adata.obs[groupby]==igrp)
        icoord = adata.obsm[basis][idx]
        iadata = adata[idx]
        icellid = cellidx[idx]

        edge, dist, simi = coord_2_dist(icoord, 
                                      show_hist=show_hist,
                                      title=groups[i],
                                      n_neighbors=n_neighbors[i],
                                      radiu_trim=radiu_trim[i],
                                      radius=radius[i],
                                      return_dsimi=return_dsimi,
                                      infer_thred=infer_thred,
                                      remove_loop=remove_loop,
                                      simi_scale=simi_scale,
                                      simi_method=dsimi_method,
                                      verbose=verbose,
                                      n_jobs=n_jobs,
                                      **kargs)
        if return_esimi:
            exp_simi, exp_idx = similarity_exp(iadata, edge_index=edge, 
                                        doscale=doscale,
                                        dopca=dopca,
                                        n_pcs=n_pcs, 
                                        max_value=max_value, 
                                        title = groups[i],
                                        method = esimi_method,
                                        esimi_thred =esimi_thred[i], 
                                        show_plot= show_exp)
            print(f'total edges: {exp_idx.shape[0]}; low_threshold edges: {exp_idx.shape[0] - exp_idx.sum()}')
        else:
            exp_simi = np.ones(edge.shape[1])
            exp_idx  = np.ones(edge.shape[1])

        edge_attr = np.array([dist, simi, exp_simi, exp_idx])
        if remove_lowesimi and return_esimi:
            edge = edge[:, exp_idx.astype(np.bool_)]
            edge_attr = edge_attr[:, exp_idx.astype(np.bool_)]

        if merge_edges:
            edges.append(icellid[edge])
        else:
            edges.append(edge)
        edges_attr.append(edge_attr)

    if merge_edges or (groupby is None):
        edges_infor ={'edges':  np.concatenate(edges, axis=1),
                      'edges_attr': np.hstack(edges_attr)}
    else:
        edges_infor ={'edges': dict(zip(groups, edges)), 
                      'edges_attr': dict(zip(groups, edges_attr))}
    adata.uns[add_key] = edges_infor
    adata.obs[groupby] = pd.Categorical(adata.obs[groupby], categories=groups)
    if verbose:
        print('computing spatial edges...\n'
            f"finished: added to `.uns['{add_key}']`")

def similarity_exp(adata, edge_index=None, doscale=True, dopca=True, n_pcs=100, max_value=10, 
                   method = 'cosine', esimi_thred = 0.1, 
                   title = None, show_plot=False):
    import scanpy as sc
    adatasi = adata.copy()
    if doscale:
        sc.pp.scale(adatasi, zero_center=True, max_value=max_value)
    if dopca:
        sc.tl.pca(adatasi, svd_solver='arpack', n_comps=n_pcs)
        similar = adatasi.obsm['X_pca']
    else:
        similar = adatasi.X.toarray() if issparse(adatasi.X) else adatasi.X

    similar = similarity_mtx(similar, similar, method = method, pairidx = edge_index)
    simi_idx =  np.abs(similar) >= esimi_thred
    if (not edge_index is None) and (show_plot):
        fig, ax = plt.subplots(1,2, figsize=(7.5,3))
        ax[0].hist(np.abs(similar).flatten(), bins=100)
        if esimi_thred >0 :
            ax[0].axvline(esimi_thred, color='black', label=f'exp_simi_thred: {esimi_thred :.3f}')
            if similar.min() < 0:
                ax[0].axvline(-esimi_thred, color='gray')

        _, counts = np.unique(edge_index[1][simi_idx], return_counts=True)
        mean_neig = np.mean(counts)
        bins = np.max(counts)
        ax[1].hist(counts, bins=bins, facecolor='b', label=f'mean_neighbors:{mean_neig :.3f}', )
        ax[1].legend()

        if not title is None:
            ax[0].set_title(f'{title} expression similarity distribution')
            ax[1].set_title(f'{title} mean neighbor distribution')
        plt.tight_layout()
        plt.show()
    return similar, simi_idx

def similarity_mtx(mtxa, mtxb, method = 'cosine', pairidx = None):
    if method == 'cosine':
        l2a =  np.linalg.norm(mtxa, ord=None, axis=1)[:, np.newaxis]
        l2b =  np.linalg.norm(mtxb, ord=None, axis=1)[:, np.newaxis]
        l2a[l2a< 1e-8] = 1e-8
        l2b[l2b< 1e-8] = 1e-8
        mtxa = mtxa / l2a
        mtxb = mtxb / l2b
        if pairidx is None:
            return mtxa @ mtxb.T
        else:
            return np.sum(mtxa[pairidx[0]] * mtxb[pairidx[1]], axis=1)
    elif method == 'pearson':
        mtxa = mtxa - mtxa.mean(1)[:, None]
        mtxb = mtxb - mtxb.mean(1)[:, None]
        stda = np.sqrt(np.sum(np.square(mtxa), axis=1))
        stdb = np.sqrt(np.sum(np.square(mtxb), axis=1))
        stda[stda< 1e-8] = 1e-8
        stdb[stdb< 1e-8] = 1e-8
        mtxa = mtxa/stda[:, None]
        mtxb = mtxb/stdb[:, None]
        if pairidx is None:
            return mtxa @ mtxb.T
        else:
            return np.sum(mtxa[pairidx[0]] * mtxb[pairidx[1]], axis=1)

def exp_edges(adata, doscale = True, max_value = 10, dopca = True, n_pcs=100, 
                method='annoy', n_jobs= -1,
                metric='euclidean', n_neighbors=50, **kargs):
    import scanpy as sc
    adatasi = adata.copy()
    if doscale:
        sc.pp.scale(adatasi, zero_center=True, max_value=max_value)
    if dopca:
        sc.tl.pca(adatasi, svd_solver='arpack', n_comps=n_pcs)
        similar = adatasi.obsm['X_pca']
    else:
        similar = adatasi.X.toarray() if issparse(adatasi.X) else adatasi.X

    kdnn = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
    kdnn.fit(similar)
    cdkout = kdnn.transform(similar, knn=n_neighbors, **kargs)
    return cdkout

def min_dist(coord, algorithm='auto', metric = 'minkowski', quantiles = [0.05, 0.95], n_jobs=-1):
    nbrs = sknn(n_neighbors=2,
                p=2,
                n_jobs=n_jobs,
                algorithm=algorithm, metric=metric)
    nbrs.fit(coord)
    distances, indices = nbrs.kneighbors(coord, 2, return_distance=True)
    distances = distances[:, 1].flatten()
    dmin, dmax = np.quantile(distances, quantiles)
    mean_dist = np.mean(distances[( (distances>=dmin) & (distances<=dmax) )])
    return mean_dist

def compute_connectivities_umap(X, knn_indices=None, knn_dists=None,
                                random_state=0, metric=None, # euclidean 
                                n_obs=None, n_neighbors=15, set_op_mix_ratio=1.0,
                                local_connectivity=1.0):
    from scipy.sparse import coo_matrix
    from umap.umap_ import fuzzy_simplicial_set
    if X is None:
        X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(X, n_neighbors, random_state, metric,
                                          knn_indices=knn_indices, knn_dists=knn_dists,
                                          set_op_mix_ratio=set_op_mix_ratio,
                                          local_connectivity=local_connectivity)
    if isinstance(connectivities, tuple):
        connectivities = connectivities[0]
    return connectivities.tocsr()

def disttosimi(distances, method='exp', scale=None ):
    if(scale is None):
        scale = 1
    elif isinstance(scale, str):
        if scale == 'max':
            scale = np.max(distances)
        elif scale == 'mean':
            scale = np.mean(distances)
        elif scale == 'median':
            md = np.ma.masked_where(distances == 0, distances)
            scale = np.ma.median(md)
        elif scale == 'l2':
            scale = np.linalg.norm(distances, axis=None, ord=2)**0.5
        else:
            scale = 1

    distances = distances/scale
    if method == 'linear':
        simi = 1- distances
    elif method == 'negsigmid':
        simi = (2*np.exp(-distances))/(1+np.exp(-distances))
    elif method == 'exp':
        simi = np.exp(-distances)
    elif method == 'exp1':
        nonz_min = np.min(distances[distances>0])
        distances = np.clip(distances, nonz_min, None)
        simi = np.exp(1-distances)
    elif method == 'log':
        nonz_min = np.min(distances[distances>0])
        distances = np.clip(distances-nonz_min, 0, None)
        simi = 1/(1+np.log(1+distances))
    return simi

def trip_edges(vector, filter = 0.95, agg='max'):
    methods = collections.OrderedDict({
                'isodata': filters.threshold_isodata,
                'li': filters.threshold_li,
                'mean': filters.threshold_mean,
                'minimum': filters.threshold_minimum,
                'otsu': filters.threshold_otsu,
                'triangle': filters.threshold_triangle,
                'yen': filters.threshold_yen})
    
    if type(filter) in [int, float] and (0<filter<=1):
        multsig = sci.stats.norm.ppf((1+filter)/2 , 0, 1)
        thred = np.mean(vector) + multsig*np.std(vector)

    elif filter == 'hist7':
        threds = []
        for k,ifilter in methods.items() :
            if k in ['isodata', 'minimum']:
                try:
                    threds.append(ifilter(vector, nbins=50))
                except:
                    pass
            else:
                try:
                    threds.append(ifilter(vector))
                except:
                    pass
        print(threds)
        thred = eval(f'np.{agg}')(np.sort(threds)[1:-1])
    elif filter in methods.kyes():
        thred = methods[filter](vector)
    else:
        raise('the filter must be in one of "float, hist7, isodata, li, mean, minimum, otsu, triangle, yen"')
    return thred

def coord_2_dist(coord, 
                 n_neighbors=None,
                 radius=None,
                  algorithm='auto',
                  radiu_trim = 'infer',
                  infer_thred = 0.95,
                  z_scale = 0.99,
                  max_neighbor = 1111,
                  p = 2,
                  remove_loop = False,
                  show_hist= True,

                  metric = 'minkowski',
                  title = None,
                  return_dsimi = False,
                  simi_method = 'exp1',
                  simi_scale = 'infer',
                  verbose = True,
                  n_jobs = -1,
                  **kargs):
    if coord.shape[1] == 3:
        coord = coord.copy()
        coord[:,2] = z_scale * coord[:,2]

    if radius:
        nbrs = sknn(radius=radius,
                    n_neighbors=max_neighbor,
                    p=p,
                    n_jobs=n_jobs,
                    algorithm=algorithm, metric=metric, **kargs)
        nbrs.fit(coord)
        distances, indices = nbrs.radius_neighbors(coord, radius, return_distance=True)
        src = np.concatenate(indices, axis=0)
        dst = np.repeat(np.arange(indices.shape[0]), list(map(len, indices)))
        dist = np.concatenate(distances, axis=0)
        radiu_trim = None

    elif n_neighbors:
        nbrs = sknn(n_neighbors=n_neighbors+1,
                    p=p,
                    algorithm=algorithm, metric=metric, **kargs)
        nbrs.fit(coord)
        distances, indices = nbrs.kneighbors(coord, n_neighbors+1, return_distance=True)

        src = indices.flatten('C')
        dst = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
        dist = distances.flatten('C')

    if return_dsimi:
        scale_simi = min_dist(coord) if simi_scale=='infer' else simi_scale
        simi = disttosimi(dist, method=simi_method, scale=scale_simi)
    else:
        simi = np.ones(len(dist))

    keep_idx = np.ones(src.shape[0], dtype=bool)
    if remove_loop:
        keep_idx &= (src != dst)

    if radiu_trim == 'infer':
        radiu_trim = trip_edges(dist[keep_idx], filter=infer_thred)

    if not radiu_trim is None:
        keep_idx &= (dist<=radiu_trim)

    dist_raw = dist.copy()
    src = src[keep_idx]
    dst = dst[keep_idx]
    dist = dist[keep_idx]
    simi = simi[keep_idx]

    _, counts = np.unique(dst, return_counts=True)
    mean_neig = np.mean(counts)
    bins = (n_neighbors+1) if n_neighbors else np.max(counts)

    if verbose:
        if title:
            print('*'*10 + title + '*'*10)
        print(f'radius: {radiu_trim or radius}\n'
            f'nodes: {len(coord)}\nedges: {len(dst)}\n'
            f'mean neighbors: {mean_neig :.7}')

    if show_hist:
        cols = 3 if return_dsimi else 2
        fig, ax = plt.subplots(1, cols, figsize=((cols+0.5)*3, 3))

        ax[0].hist(dist_raw, histtype='barstacked', bins=50, facecolor='r', alpha=1)
        if not radiu_trim is None:
            ax[0].axvline(radiu_trim, color='black', 
                          label=f'radius: {radiu_trim :.3f}\nnodes: {len(coord)}\nedges: {len(dst)}')
            ax[0].legend()
        ax[1].hist(counts, bins=bins, facecolor='b', label=f'mean_neighbors:{mean_neig :.3f}', )
        ax[1].legend()

        if return_dsimi:
            ax[2].hist(simi, bins=50, facecolor='grey',
                label=f'mean similarity:{np.mean(simi) :.3f}', )
            ax[2].legend()
        if not title is None:
            ax[0].set_title(f'{title} distance distribution')
            ax[1].set_title(f'{title} mean neighbor distribution')
            if return_dsimi:
                ax[2].set_title(f'{title} mean similarity distribution')
        plt.tight_layout()
        plt.show()

    return [np.array([src, dst]), dist, simi]

def coord_2_dists(corrds, show_hist=True, titles=None, return_dsimi=True,
                  merge_edges = True,
                  n_neighbors=10, radius = None, radiu_trim='infer', **kargs):
    assert type(corrds) == list, 'must input multiple list corrds.'
    start = 0
    edges, dists, simis = [], [], []
    
    if isinstance(n_neighbors, int) or (n_neighbors is None):
        n_neighbors = [n_neighbors] * len(corrds)
    
    if type(radius) in [int, float] or (radius is None):
        radius = [radius] * len(corrds)

    if not isinstance(radiu_trim, list):
        radiu_trim = [radiu_trim] * len(corrds)

    for i in range(len(corrds)):
        coord = corrds[i]
        title = None if titles is None else titles[i]
        edge, dist, simi = coord_2_dist(coord, 
                                      show_hist=show_hist,
                                      title=title,
                                      n_neighbors=n_neighbors[i],
                                      radiu_trim=radiu_trim[i],
                                      radius=radius[i],
                                      return_dsimi=return_dsimi,
                                      **kargs)
        if merge_edges:
            edges.append(edge + start)
        else:
            edges.append(edge)
        dists.append(dist)
        simis.append(simi)
        start += len(coord)
    if merge_edges:
        edges = np.concatenate(edges, axis=1)
        dists = np.hstack(dists)
        simis = np.hstack(simis)
    return [edges, dists, simis]