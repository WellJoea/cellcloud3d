import numpy as np
import networkx as nx

class searchmatch():
    def __init__(self):
        pass

    def digraph(self, edges, root=None, showtree=True, 
                prog="twopi", layout="kawai", font_size=8, node_size=100):
        G = nx.DiGraph()
        G.add_edges_from(edges)

        if showtree:
            if layout == "spectral":
                nplt = nx.draw_spectral
            elif layout == 'spring':
                nplt = nx.draw_spring
            elif layout == 'circular':
                nplt = nx.draw_circular
            elif layout == 'kawai':
                nplt = nx.draw_kamada_kawai
            #nplt(G, with_labels=True, font_weight='bold')
            import matplotlib.pyplot as plt
            from networkx.drawing.nx_pydot import graphviz_layout

            pos = graphviz_layout(G, prog=prog, root=root)
            nx.draw(G, pos, node_size=node_size, alpha=1, font_size = font_size,
                    #connectionstyle="arc3,rad=-0.2",
                    linewidths  = 2, node_color="red", with_labels=True)
            plt.show()

        self.G = G
        self.root = root
        self.edges = edges
        self.dfs_edges = list(nx.dfs_edges(G, source=root,
                                            depth_limit=None))
        try:
            self.bfs_edges = list(nx.bfs_edges(G, source=root, 
                                                depth_limit=None, 
                                                sort_neighbors=None))
        except:
            self.bfs_edges = []
            roots = list(v for v, d in G.in_degree() if d == 0)
            for iroot in roots:
                self.bfs_edges += list(nx.bfs_edges(G, source=iroot, 
                                                    depth_limit=None, 
                                                    sort_neighbors=None))
        return self

    def buildidx(self, n_node=None, 
                 step=1, 
                 root=None, 
                 edges=None,
                 showtree=True, 
                 layout="spectral", **kargs):
        '''
        edges = [(1,3), (1,2), (4,5), (5,6), (4,7)]
        KK = searchmatch().buildidx( edges=edges, step=3, layout="circular")
        list(KK.dfs_edges)
        '''
        if edges is None:
            source, target = self.treeidx(n_node, 
                                          step=step, 
                                          root=root )
            edges = zip(source, target)

        self.digraph(edges, 
                    root=root,
                    showtree=showtree, 
                    layout=layout, **kargs)
        return self

    @staticmethod
    def searchidx(n_node,
                  labels=None,
                    step=1, 
                    root=None, 
                    regist_pair=None,
                    full_pair=False,
                    showtree=False, 
                    keep_self = False,
                    layout="spectral", 
                    **kargs):
        if not full_pair:
            assert  (not root is None) or (not regist_pair is None), 'A least one of root and regist_pair is not None.'

        align_idx = None
        trans_idx = None

        if full_pair:
            align_idx =  [ (i,j) for i in range(n_node) for j in range(i+1, n_node) ]
    
        if not regist_pair is None:
            align_idx = searchmatch().buildidx(n_node=n_node, 
                                                step=step, 
                                                root=None, 
                                                edges=regist_pair,
                                                showtree=showtree, 
                                                layout=layout, **kargs).dfs_edges
            if keep_self:
                loop_self = [ (i,j) for i,j in regist_pair if i == j ]
                align_idx = loop_self + align_idx
        if not root is None:
            trans_idx = searchmatch().buildidx(n_node=n_node, 
                                        step=step, 
                                        root=root, 
                                        edges=None,
                                        showtree=showtree, 
                                        layout=layout, **kargs).dfs_edges

        align_pair = trans_idx if align_idx is None else align_idx
        trans_pairs = align_idx if trans_idx is None else trans_idx
        if not labels is None:
            align_pair = [ (labels[k], labels[v]) for k,v in align_pair ]

        return [align_pair, trans_pairs]

    @staticmethod
    def treeidx(n_node, step=1, root=None ):
        root = n_node // 2 if root is None else root

        target1 = np.arange(0, root)[::-1]
        source1 = sum([ [i+1]+[i]*(step-1) for i in range(root-1, -1, -step) ], [])
        source1 = np.array(source1)[:root]

        target2 = np.arange(root, n_node, 1)
        source2 = sum([ [i-1]+[i]*(step-1)  for i in range(root, n_node, step) ], [])
        source2 = np.array(source2)[:(n_node-root)]
        source2[0] = root

        source = np.concatenate([source1, source2])
        target = np.concatenate([target1, target2])
        sortidx= np.argsort(target)
        source = np.int64(source[sortidx])
        target = np.int64(target[sortidx])
        return [source, target]

    @staticmethod
    def neighboridx(start, end, top=None, values=None, showtree=False):
        top = 0 if top is None else top
        if not values is None:
            start = 0
            end = len(values)-1
            top = list(values).index(top)
        back = np.arange(top, start, -1)
        forw = np.arange(top,end)
        source= np.concatenate([back, [top], forw])
        target= np.concatenate([back-1, [top], forw+1])
        if not values is None:
            source = np.array(values)[source]
            target = np.array(values)[target]
        G = nx.DiGraph()
        G.add_edges_from(zip(source, target))
        if showtree:
            nx.draw(G, with_labels=True, font_weight='bold', layout="spring")
        return G