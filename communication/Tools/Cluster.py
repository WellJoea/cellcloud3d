import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
#from scipy.spatial.distance import pdist, squareform
#from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram, fcluster
def sci_dist(df_data,
             dist_mtx = None,
             cor_method='pearson'):
    if not dist_mtx is None:
        dist_condensed = dist_mtx
    elif cor_method in ['pearson', 'kendall', 'spearman']:
        corr_matrix = df_data.T.corr(method=cor_method)
        dist_condensed = ssd.squareform(1 - corr_matrix)
    elif cor_method in ['sknormal']:
        from sklearn.preprocessing import normalize     
        dist_condensed = normalize(df_data.copy()) 
    else:
        dist_condensed = df_data.copy()
    return dist_condensed

def sci_linkage(dist_mtx,
                    method='complete', 
                    metric='euclidean',
                    fastclust=True,
                    optimal_ordering=False,
                    **kargs):
    if fastclust:
        import fastcluster
        z_var = fastcluster.linkage(dist_mtx, method=method,
                                    metric=metric, **kargs)
    else:
        z_var = sch.linkage(dist_mtx,
                        method=method, 
                        metric=metric,
                        optimal_ordering=optimal_ordering, 
                        **kargs)
    return z_var

def sci_dendrogram(cluster ,labels = None, 
                    color_threshold=None, 
                    leaf_rotation=None, 
                    link_colors = list(map(mpl.colors.rgb2hex, plt.get_cmap('tab20').colors)),
                    **kargs):
    sch.set_link_color_palette(link_colors)
    dendro_info = sch.dendrogram(cluster,
                                labels=labels,
                                leaf_rotation=leaf_rotation,
                                color_threshold=color_threshold, 
                                **kargs)
    return(dendro_info)

def sci_cuttree(Z, n_clusters=None, height=None):
    cutTree = sch.cut_tree(Z, n_clusters=n_clusters, height=height)
    return cutTree