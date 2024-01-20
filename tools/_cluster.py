import os
import numpy as np
import pandas as pd

def rmclust(adata, nclust=None, use_rep='pca', X=None,  modelNames='EEE', add_key='mclust', copy=False,
             R_HOME = None, R_USER=None, verbose=False,
            random_seed=491001):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    https://mclust-org.github.io/mclust/reference/mclustModelNames.html
    modelNames: A vector of character strings indicating the models to be
          fitted in the EM phase of clustering. The default is:

            • for univariate data (d = 1): ‘c("E", "V")’

            • for multivariate data (n > d): all the models available
              in ‘mclust.options("emModelNames")’
                "EII" "VII" "EEI" "VEI" "EVI" "VVI" "EEE" "VEE" "EVE" 
                "VVE" "EEV" "VEV" "EVV" "VVV"

            • for multivariate data (n <= d): the spherical and
              diagonal models, i.e. ‘c("EII", "VII", "EEI", "EVI",
              "VEI", "VVI")’
    """
    if R_HOME:
        os.environ['R_HOME'] = R_HOME
    if R_USER:
        os.environ['R_USER'] = R_USER
    if copy:
        adata = adata.copy()
    if X is None:
        X = adata.obsm[use_rep]

    np.random.seed(random_seed)
    from rpy2 import robjects as robj
    robj.r.library("mclust")

    import rpy2.robjects.numpy2ri as npr
    npr.activate()
    r_random_seed = robj.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robj.r['Mclust']
    summary = robj.r['summary']
    nclust = robj.rinterface.NULL if nclust is None else nclust #robj.r.seq(1, nclust)

    #mclust::adjustedRandIndex(kowalczyk.integrated$seurat_clusters, cell_info$cell_type_label)
    res = rmclust(npr.numpy2rpy(X), nclust, modelNames=modelNames)
    if verbose:
        print(summary(res))
    try:
        clust = np.int32(res[-2])-1
        adata.obs[add_key] = pd.Categorical(clust.astype(str),
                                    categories=np.unique(clust).astype(str))
    except:
        adata.obs[add_key] = '0'

    if copy:
        return adata
