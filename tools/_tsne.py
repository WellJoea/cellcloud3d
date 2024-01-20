import numpy as np

def cutsne():
    import cudf
    import cuml
    import cudf, requests
    import cupy

def ftsne(adata=None,
            x_train=None,
            use_rep='X_pca', 
            add_key='X_tsne',
            copy=False, 
            perplexity = 30, 
            n_components=2,
            n_jobs=5,
            metric="euclidean", #cosine
            initialization='pca', #random
            learning_rate= 'auto', #'auto',
            early_exaggeration_iter=250,
            early_exaggeration='auto',
            n_iter=500,
            exaggeration=None,
            initial_momentum=0.8,
            final_momentum=0.8,
            verbose=True,
            random_state=19491001,
            **kargs):

    X = adata.obsm[use_rep].copy() if x_train is None else x_train.copy()

    if type(perplexity) in [int, float]:
        import openTSNE
        tsne = openTSNE.TSNE(
            n_components=n_components,
            n_jobs=n_jobs,
            perplexity=perplexity,
            metric=metric,
            initialization=initialization,
            learning_rate=learning_rate,
            early_exaggeration_iter=early_exaggeration_iter,
            early_exaggeration=early_exaggeration,
            n_iter=n_iter,
            exaggeration=exaggeration,
            initial_momentum=initial_momentum,
            final_momentum=final_momentum,
            random_state=random_state,
            verbose=verbose,
            **kargs
        )
        embedding_train = tsne.fit(X)
    else:
        import openTSNE
        affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
            X,
            perplexities=perplexity,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        if initialization=='pca':
            init = openTSNE.initialization.pca(X, n_components=n_components, random_state=random_state)
        else:
            init = openTSNE.initialization.random(X, n_components=n_components,  random_state=random_state)

        tsne = openTSNE.TSNE(n_jobs=n_jobs,
                            n_components=n_components,
                            metric=metric,
                            learning_rate=learning_rate,
                            early_exaggeration_iter=early_exaggeration_iter,
                            early_exaggeration=early_exaggeration,
                            n_iter=n_iter,
                            exaggeration=exaggeration,
                            initial_momentum=initial_momentum,
                            final_momentum=final_momentum,
                            random_state=random_state,
                            verbose=verbose,
                            **kargs
            )
        embedding_train = tsne.fit(
            affinities=affinities_multiscale_mixture,
            initialization=init,
        )
    
    if (not adata is None):
        adata = adata.copy() if copy else adata
        adata.obsm[add_key] = np.array(embedding_train)
        if copy:
            return adata

    if (not x_train is None):
        return embedding_train