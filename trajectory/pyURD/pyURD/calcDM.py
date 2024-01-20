import scanpy as sc
from scipy.sparse import csr_matrix, bsr_matrix,csc_matrix, csgraph
from .utils import Time

def scanpyDM(URD, n_comps=60, neighbors_key=None, copy=False):
    dpt = sc.tools._dpt.DPT(URD.adata, neighbors_key=neighbors_key)
    dpt.compute_transitions()
    dpt.compute_eigen(n_comps=n_comps)
    URD.adata.obsm['X_diffmap'] = dpt.eigen_basis
    URD.adata.uns['diffmap_evals'] = dpt.eigen_values
    URD.adata.obsp['diffmap_trans_sym'] = dpt.transitions_sym
    URD.adata.obsp['diffmap_trans'] = dpt.transitions
    return URD if copy else None

def pyDM(URD, n_jobs=-1, algorithm='ball_tree', k=60, epsilon='bgh', **kargs):
    from pydiffmap import diffusion_map as dm
    neighbor_params = {'n_jobs': n_jobs, 'algorithm': algorithm}
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs=1, k=60, epsilon=epsilon, alpha=1.0, 
                                          neighbor_params=neighbor_params, **kargs)

    dmap = mydmap.fit_transform(URD.adata.X.toarray())

def dMaps(URD, adataX=None, n_comps=60, bandwidth=10, copy=False):
    import dmaps
    import numpy as np
    import matplotlib.pyplot as plt

    # Assume we have the following numpy arrays:
    # coords contains the [n, 3] generated coordinates for the Swiss roll dataset.
    # color contains the position of the points along the main dimension of the roll. 
    adataX = URD.adata.X.toarray().copy() if adataX is None else adataX
    dist = dmaps.DistanceMatrix(adataX)
    dist.compute(metric=dmaps.metrics.euclidean)

    # Compute top three eigenvectors. 
    # Here we assume a good value for the kernel bandwidth is known.
    dmap = dmaps.DiffusionMap(dist)
    dmap.set_kernel_bandwidth(bandwidth)
    dmap.compute(n_comps)
    URD.adata.obsm['X_diffmap']= dmap.get_eigenvectors()
    URD.adata.uns['diffmap_evals'] = dmap.get_eigenvalues()
    URD.adata.obsp['diffmap_trans'] = csr_matrix(dmap.get_kernel_matrix())
    return URD if copy else None

def calcdiffmap(URD, method='scanpyDM', **kargs):
    print( f"{Time()} calcdiffmap with {method}." )
    if method=='scanpyDM':
        scanpyDM(URD,  **kargs)
    elif method=='dMaps':
        dMaps(URD,  **kargs)
    print(Time(), "finish calcdiffmap.")

