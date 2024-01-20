import leidenalg as la
import louvain as lo
import igraph as ig
import networkx as nx
import scipy as sci
import numpy as np
import pandas as pd

def pandas2adjacency(df, gtype='igraph', drop_neg_weights=False ):
    df = df.copy()
    col = df.columns 
    row = df.index
    if drop_neg_weights:
        df[df<0] = 0
    if (len(col) == len(row)) and (row != row).sum() ==0: # fast
        if gtype in ['nx', 'networkx']:
            G = nx.from_pandas_adjacency(df)
        elif gtype in ['nx2ig']:
            nG = nx.from_pandas_adjacency(df)
            G = ig.Graph.from_networkx(nG, vertex_attr_hashable="name")
        elif gtype in ['igwa']:
            G = ig.Graph.Weighted_Adjacency(df, 
                                            attr='weight',
                                            mode='undirected')
        elif gtype in ['iga']:
            G = ig.Graph.Adjacency(df, 
                                    mode='undirected')

        elif gtype in ['ig', 'igraph']:
            adj = sci.sparse.csr_matrix(df.values)
            sources, targets = adj.nonzero()
            weights = adj[sources, targets]
            if isinstance(weights, np.matrix):
                weights = weights.A1
            G = ig.Graph(directed=None)
            G.add_vertices(adj.shape[0])
            G.add_edges(list(zip(sources, targets)))
            G.vs["name"] = col
            G.es['weight'] = weights

    else:
        g_df = df.melt(ignore_index=False).reset_index()
        g_df.columns = ['source', 'target', 'value']

        if gtype in ['nx', 'networkx']:
            G = nx.Graph()
            G.add_weighted_edges_from(g_df.to_numpy())
            #H = nx.relabel_nodes(G, rename_label)
        elif gtype in ['nx2ig']:
            G = nx.Graph()
            G.add_weighted_edges_from(g_df.to_numpy())
            G = ig.Graph.from_networkx(G, vertex_attr_hashable="name")
        elif gtype in ['ig','igraph']:
            G = ig.Graph.DataFrame(g_df, directed=False, use_vids=False)
            G.es["weight"] = g_df['value']

        if G.is_bipartite():
            G.vs['type'] = np.where(np.isin(G.vs['name'], list(row)), 0, 1)
    return G

def merge_small_cluster(partition, optimiser=None, min_comm_size=100, max_comm_num=150):
    optimiser = la.Optimiser() if optimiser is None else optimiser
    optimiser.consider_empty_community = False

    cluster, sizes = np.unique(partition.membership, return_counts=True)
    if min_comm_size is None:
        min_comm_size = np.min(sizes)
    if max_comm_num is None:
        max_comm_num = len(sizes)

    cont = 1
    while min_comm_size > np.min(sizes) or max_comm_num < len(sizes):
        aggregate_partition = partition.aggregate_partition(partition)
        cluster, sizes = np.unique(partition.membership, return_counts=True)
        minidx = np.where(sizes == np.min(sizes))[0][-1]
        smallest = cluster[minidx]

        fixed = np.setdiff1d(cluster, smallest)
        max_diff = -np.inf
        cluster_idx = -1
        for fix in fixed:
            diff = aggregate_partition.diff_move(smallest, fix)
            if diff > max_diff:
                max_diff = diff
                cluster_idx = fix

        aggregate_partition.move_node(smallest, cluster_idx)
        if cont <11:
            print(f'merge cluster {smallest} -> {cluster_idx}')
        elif cont ==11:
            print(f'merge cluster ... -> ...')
        cont += 1

        if not (aggregate_partition.sizes()[-1] == 0):
            optimiser.optimise_partition(aggregate_partition)
        partition.from_coarse_partition(aggregate_partition)
        sizes = partition.sizes()
    # optimiser.optimise_partition(partition)
    return partition


def leiden_cluster(
    G: ig.Graph ,
    drop_neg_weights=False,
    graph_type=None,
    resolution: float = 1,
    random_state: int = 0,
    directed: bool = False,
    use_weights: bool = True,
    n_iterations: int = -1,
    max_comm_size: int = 0,
    initial_membership: np.array = None,
    max_comm_num: int = None,
    min_comm_size: int = None,
    layer_weights=None,
    is_membership_fixed=None,
    partition_type: la.VertexPartition.MutableVertexPartition = None,
    **partition_kwargs,
) -> np.array:

    if graph_type == 'multiplex':
        drop_neg_weights = False
        G_pos = G.subgraph_edges(G.es.select(weight_gt = 0), delete_vertices=False)
        G_neg = G.subgraph_edges(G.es.select(weight_lt = 0), delete_vertices=False)
        G_neg.es['weight'] = [-w for w in G_neg.es['weight']]
        part_pos = la.RBConfigurationVertexPartition(G_pos, weights='weight', resolution_parameter=resolution)
        part_neg = la.RBConfigurationVertexPartition(G_neg, weights='weight', resolution_parameter=resolution)

        optimiser = la.Optimiser()
        optimiser.set_rng_seed(random_state)
        optimiser.consider_empty_community = False
        diff = optimiser.optimise_partition_multiplex([part_pos, part_neg],
                                                      layer_weights=[1,-1],
                                                      n_iterations=n_iterations,
                                                      is_membership_fixed=is_membership_fixed)
        part = part_pos #[part_pos, part_neg]
    elif graph_type == 'bipartite':
        p_01, p_0, p_1 = la.CPMVertexPartition.Bipartite(G,
                                                 initial_membership=initial_membership,
                                                 weights = np.array(G.es['weight']).astype(np.float64),
                                                 degree_as_node_size=False,
                                                 resolution_parameter_01=resolution,
                                                 resolution_parameter_0=0,
                                                 resolution_parameter_1=0)
        optimiser = la.Optimiser()
        optimiser.set_rng_seed(random_state)
        optimiser.consider_empty_community = False
        diff = optimiser.optimise_partition_multiplex([p_01, p_0, p_1],
                                                      layer_weights=[1, -1, -1],
                                                      n_iterations=n_iterations,
                                                      is_membership_fixed=is_membership_fixed)
        part = p_01 #[part_pos, part_neg]

    else:
        partition_kwargs = dict(partition_kwargs)
        if partition_type is None:
            partition_type = la.RBConfigurationVertexPartition

        partition_kwargs['n_iterations'] = n_iterations
        partition_kwargs['max_comm_size'] = max_comm_size
        partition_kwargs['initial_membership'] = initial_membership

        if resolution is not None:
            if partition_type==la.CPMVertexPartition.Bipartite:
                partition_kwargs['resolution_parameter_01'] = resolution
                partition_kwargs['degree_as_node_size'] = False
            else:
                partition_kwargs['resolution_parameter'] = resolution
        # clustering proper
    
        if drop_neg_weights:
            G = G.subgraph_edges(G.es.select(weight_gt = 0), delete_vertices=False)

        if use_weights:
            partition_kwargs['weights'] = np.array(G.es['weight']).astype(np.float64)

        partition_kwargs['seed'] = random_state
        part = la.find_partition(G, partition_type, **partition_kwargs)
        if not((min_comm_size is None) and (max_comm_num is None)):
            part = merge_small_cluster(part, 
                                        optimiser=la.Optimiser(),
                                        min_comm_size=min_comm_size,
                                        max_comm_num=max_comm_num)
        groups = np.array(part.membership)
    return part

