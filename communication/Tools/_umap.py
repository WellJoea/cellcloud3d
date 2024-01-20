import umap
from umap.umap_ import simplicial_set_embedding
from umap.umap_ import find_ab_params
from scipy.sparse import csr_matrix,bsr_matrix,csc_matrix, csgraph
from sklearn.utils import check_random_state, check_array
import seaborn as sns

neighbors = wgcna.TOM.values.copy()
#neighbors = wgcna.adjacency.copy()
#neighbors[neighbors<0.001] = 0
#np.fill_diagonal(neighbors,0)
neighbors = csr_matrix(neighbors) 

spread = 15
n_components = 2
initial_alpha =1 
min_dist = 0.3
a, b = find_ab_params(spread, min_dist)

X_umap, _ = simplicial_set_embedding(
    csr_matrix(adata_exp.X.T),
    neighbors,
    n_components,
    initial_alpha,
    a, b,
    gamma = 1,
    negative_sample_rate = 5,
    n_epochs = 500 ,
    init = 'spectral',
    random_state = check_random_state(1),
    metric = 'euclidean',
    metric_kwds = {},
    densmap=False,
    densmap_kwds={},
    output_dens=False,
)

leiden_g = leiden_cluster(neighbors, resolution=2, directed=False)
print(np.unique(leiden_g).shape)

dynamodule = wgcna.datExpr.var['moduleLabels'] #wgcna.dynamicMods['Value']

umap_df = pd.DataFrame(X_umap, columns=['umap1', 'umap2'], index=wgcna.datExpr.var_names)
umap_df['leiden'] =  pd.Categorical(leiden_g.astype(str), 
                                    categories = np.sort(np.unique(leiden_g)).astype(str))
umap_df['dynamo'] =  pd.Categorical(dynamodule.astype(str), 
                                    categories = np.sort(np.unique(dynamodule)).astype(str))


fig, axis = plt.subplots(1,2, constrained_layout=True, figsize=(14.5,6))
p1 = sns.scatterplot(data=umap_df, x="umap1", y="umap2", hue='leiden', s=15,
                     cmap = 
                     linewidth=0, ax=axis[0])
p2 = sns.scatterplot(data=umap_df, x="umap1", y="umap2", hue='dynamo', s=15, linewidth=0, ax=axis[1])
p1.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, title='leiden')
p2.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, title='dynamo', ncol=2)
plt.tight_layout()


col_list = list(map(mpl.colors.rgb2hex, plt.get_cmap('tab20').colors))
col_dict = dict(zip(list(map(str, range(len(col_list)))), col_list))
umap_anno = pd.concat([ wgcna.datExpr.var, umap_df], axis=1)

fig, axis = plt.subplots(1,2, constrained_layout=True, figsize=(14.5,6))
p1 = sns.scatterplot(data=umap_anno, x="umap1", y="umap2", hue='leiden', s=15,
                     #palette = COLORS['CellType_colors'][:umap_anno['leiden'].cat.categories.shape[0]],
                     linewidth=0, ax=axis[0])
p2 = sns.scatterplot(data=umap_anno, x="umap1", y="umap2", hue='dynamo', 
                     #palette = dict(zip(umap_anno['dynamo'], umap_anno['dynamicColors'])),
                     s=15, linewidth=0, ax=axis[1])
p1.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, title='leiden')
p2.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, title='dynamo', ncol=2)
plt.tight_layout()