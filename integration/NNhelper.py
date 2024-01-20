from tqdm import tqdm
from annoy import AnnoyIndex
from pynndescent import NNDescent
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import hnswlib
import numpy as np
import pandas as pd
import re
try:
    import faiss
except ImportError:
    pass

def sigment_thred(sigments, verbose=True, threshold_method=None):
    from skimage import filters
    import collections
    methods = collections.OrderedDict({
                'Isodata': filters.threshold_isodata,
                'Li': filters.threshold_li,
                'Mean': filters.threshold_mean,
                'Minimum': filters.threshold_minimum,
                'Otsu': filters.threshold_otsu,
                'Triangle': filters.threshold_triangle,
                'Yen': filters.threshold_yen})

    thred_dict = {}
    for _i in methods.keys():
        _t = methods[_i](sigments)
        thred_dict[_i] = _t
        print('Automaticall threshold of %s: %.4f'%(_i, _t))
    if threshold_method is None:
        threshold = np.median(list(thred_dict.values()))
        threshold_method = [ k for k,v in thred_dict.items() if v==threshold ][0]
    else:
        threshold = methods[threshold_method](sigments)
    if verbose:
        print("Automatically set threshold with %s at doublet score = %.4f"%(threshold_method, threshold))
    return(threshold)

def create_nn(data, method='annoy', 
                metric='euclidean', hnsw_space='l2',
                max_elements=None, ef_construction=200, M=16,
                num_threads=-1, pynndescent_random_state=0,
                annoy_n_trees=70, pynndescent_n_neighbors=50):
    data_labels = np.arange(data.shape[0])
    if method == 'annoy':
        ckd = AnnoyIndex(data.shape[1], metric=metric)
        for i in np.arange(data.shape[0]):
            ckd.add_item(i,data[i,:])
        ckd.build(annoy_n_trees)

    elif method == 'hnsw':
        ckd = hnswlib.Index(space=hnsw_space, dim=data.shape[1])
        ckd.init_index(max_elements = data.shape[0] if max_elements is None else max_elements, 
                        ef_construction = ef_construction, M = M)
        ckd.add_items(data, data_labels, num_threads = num_threads)

    elif  method == 'pynndescent':
        ckd = NNDescent(data, metric=metric,
                        n_jobs=-1,
                        n_neighbors=pynndescent_n_neighbors, 
                        random_state=pynndescent_random_state)
        ckd.prepare()
    elif  method == 'faiss':
        ckd = faiss.IndexFlatL2(data.shape[1])
        ckd.add(data)
    elif  method == 'cKDTree':
        ckd = cKDTree(data)
    elif  method == 'KDTree':
        ckd = KDTree(data,metric=metric)
    return ckd

def query_nn(data, ckd, method='annoy', knn=20, set_ef=30, search_k=-1, 
                num_threads=-1,
                include_distances=True):
    '''
    Query the faiss/cKDTree/KDTree/annoy index with PCA coordinates from a batch. 
    Input
    -----
    data : ``numpy.array``
        PCA coordinates of a batch's cells to query.
    ckd : faiss/cKDTree/KDTree/annoy/hnsw/pynndescent index
    '''
    if method == 'annoy':
        ckdo_ind = []
        ckdo_dist = []
        for i in tqdm(np.arange(data.shape[0])):
            holder = ckd.get_nns_by_vector(data[i,:],
                                            knn,
                                            search_k=search_k,
                                            include_distances=include_distances)
            ckdo_ind.append(holder[0])
            ckdo_dist.append(holder[1])
        ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))
    elif method == 'hnsw':
        ckd.set_ef(max(set_ef, knn+5)) #ef should always be > k
        ckd.set_num_threads(num_threads)
        labels, distances = ckd.knn_query(data, k = knn, num_threads=num_threads)
        ckdout = (np.sqrt(distances), labels) if ckd.space =='l2' else (distances, labels)
    elif method == 'pynndescent':
        ckdout = ckd.query(data, k=knn)
        ckdout = (ckdout[1], ckdout[0])
    elif method == 'faiss':
        D, I = ckd.search(data, knn)
        #sometimes this turns up marginally negative values, just set those to zero
        D[D<0] = 0
        #the distance returned by faiss needs to be square rooted to be actual euclidean
        ckdout = (np.sqrt(D), I)
    elif method == 'cKDTree':
        ckdout = ckd.query(x=data, k=knn, n_jobs=-1)
    elif method == 'KDTree':
        ckdout = ckd.query(data, k=knn)
    return ckdout

def get_predict(ckdout, ref_labels, dist_thred=None, show=True, bins=1000):
    dist_flat = ckdout[0].flatten()
    knn = ckdout[0].shape[1]
    
    if dist_thred is None:
        thred_value = sigment_thred(dist_flat, verbose=True, threshold_method=None)
    elif isinstance(dist_thred, str):
        if re.search('^[0-9]+s(t){0,1}d$', dist_thred): 
            thred_value = float(re.sub('sd|std', '', dist_thred))
            thred_value = dist_flat.max() - thred_value*np.std(dist_flat)
        elif re.search('^[0-9]+cent$', dist_thred): 
            thred_value = float(re.sub('cent', '', dist_thred))
            thred_value = np.percentile(dist_flat, thred_value)
    elif  type(dist_thred) in  [float, int]:
        thred_value = dist_thred
    
    if show:
        fig, ax=plt.subplots(1,1, figsize=(6,6))
        ax.hist(dist_flat,bins=100)
        ax.axvline(x=thred_value, color='red')
        plt.show()
    
    predict_lab = np.array(list(map(lambda x: ref_labels[x], ckdout[1])))
    predict_lab[ckdout[0]>thred_value] = 'unmapping'
    predict_lab = pd.DataFrame([ dict(zip(*np.unique(i, return_counts=True))) for i in predict_lab ])
    predict_lab.fillna(0,inplace=True)
    predict_lab = predict_lab/knn
    
    max_score = predict_lab.max(1)
    predict_id = predict_lab.idxmax(axis="columns")
    
    predict_lab['max_score'] = max_score
    predict_lab['predict_id'] = predict_id
    print(predict_lab['predict_id'].value_counts())
    return(predict_lab)


'''
ref_data = adataI[adataI.obs['species']=='mouse',:].copy()
qua_data = adataI[adataI.obs['species']=='human',:].copy()

ckd = create_nn(ref_data.obsm['X_pca'], method = 'annoy')
ckdout = query_nn(qua_data.obsm['X_pca'], ckd, knn=200,  method = 'annoy')

predict_lab = get_predict(ckdout, ref_data.obs['KeyType'].values, dist_thred=30)
predict_lab.index = qua_data.obs_names
'''

def split_correlation_matrix(adataI, 
                             clust_col='KeyType',
                             splitby='species', splitrow='human', splitcol='mouse',
                             method='average', metric='euclidean', 
                             cbar_pos=(1, 0.5, 0.01, 0.15),
                             size_inches =[10,10],
                             dendrogram_ratio=0.15,
                             figsize=[12,9], 
                             cmap="bwr", 
                             show=True, save=None, **kargs):

    clust_cat = adataI.obs[clust_col].cat.categories
    row_cat = adataI.obs.loc[ (adataI.obs[splitby]==splitrow), clust_col ]\
                    .cat.remove_unused_categories().cat.categories
    col_cat = adataI.obs.loc[ (adataI.obs[splitby]==splitcol), clust_col ]\
                    .cat.remove_unused_categories().cat.categories

    #rep_df = pd.DataFrame(adataI.X, columns=adataI.var_names, index=adataI.obs['KeyType'])
    #mean_df = rep_df.groupby(level=0).mean()
    #cor_matrix = mean_df.T.corr(method='spearman')

    cor_matrix = pd.DataFrame(adataI.uns[f'dendrogram_{clust_col}']['correlation_matrix'],
                              index=clust_cat, columns=clust_cat)
    cor_matrix = cor_matrix.loc[row_cat, col_cat]

    import seaborn as sns
    g = sns.clustermap(cor_matrix,
                       method=method, 
                       metric=metric, 
                       #cbar_kws={location:'left'},
                       cbar_pos=cbar_pos,
                       dendrogram_ratio=dendrogram_ratio, 
                       colors_ratio=1, 
                       tree_kws={'linewidths':1},
                       cmap=cmap, 
                       vmin=-1, vmax=1,
                       linewidths=0.4, linecolor='black',
                       square=True,
                       row_cluster=True, col_cluster=True, 
                       figsize=figsize, **kargs)
    #fig = plt.gcf()
    #fig.set_size_inches(size_inches)
    if save:
        plt.savefig(save, bbox_inches='tight')
        #f'{it_}.correlation_matrix.split.pdf'
    if show:
        plt.show()
    else:
        return(g)