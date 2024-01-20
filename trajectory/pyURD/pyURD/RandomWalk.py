import scanpy as sc
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse import csr_matrix, csgraph
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse import csr_matrix,bsr_matrix,csc_matrix, csgraph
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc,isspmatrix_coo
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

class calcDM:
    def __init__(self, adata, root_cells=[0], floods=None):
        self.adata = adata.copy()
        self.root_cells=root_cells.copy()
        self._floods=floods
        
    @property
    def floods(self):
        return self._floods
    @floods.setter
    def floods(self, values):
        self._floods = values
    @floods.deleter
    def floods(self):
        del self._floods

    def Time(self):
        import time
        return(time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime()))

    def corstate(self, x,y, methd='pearsonr'):
        from minepy import MINE
        import numpy as np
        import scipy.stats

        if methd =='pearsonr':
            return(scipy.stats.pearsonr(x, y)[0])
        elif methd =='spearmanr':
            return(scipy.stats.spearmanr(x, y)['correlation'])
        elif methd =='kendalltau':
            return(scipy.stats.kendalltau(x, y)['correlation'])
        elif methd=='mic':
            m = MINE()
            m.compute_score(x, y)
            return(m.mic())

    def sigmoid(self, x, x0, k, c=1):
        return(c/(1 + np.exp(-1 * k * (x - x0))))

    def sc_diffmap(self, n_comps=60, neighbors_key=None):
        dpt = sc.tools._dpt.DPT(self.adata, neighbors_key=neighbors_key)
        dpt.compute_transitions()
        dpt.compute_eigen(n_comps=n_comps)
        self.adata.obsm['X_diffmap'] = dpt.eigen_basis
        self.adata.uns['diffmap_evals'] = dpt.eigen_values
        self.adata.uns['diffmap_trans_sym'] = dpt.transitions_sym    
        self.adata.uns['diffmap_trans'] = dpt.transitions

    def checkcsr(self, csr_cm):
        if not issparse(csr_cm):
            csr_cm = csr_matrix(csr_cm)
        if isspmatrix_csc(csr_cm):
            csr_cm = csr_cm.transpose()
        if isspmatrix_coo(csr_cm):
            csr_cm = csr_cm.tocsr()
        if not isspmatrix_csr(csr_cm):
            print(type(csr_cm))
            raise('the diffmap matrix must in csr_matrix type!')
        else:
            return(csr_cm)

    def floodBuildTM(self, dm_symtrans):
        tm_flood = dm_symtrans.copy()
        tm_sum_max = tm_flood.sum(1).max()
        tm_flood = tm_flood / tm_sum_max
        return(tm_flood)

    def subcsxm(self, csxm):
        rowidx, colidx = csxm.nonzero()
        cmdata = csxm.data
        
    def combine_probs0(self, tm_flood, cells_not_visited, cells_visited):
        visitation_prob = tm_flood[np.ix_(cells_not_visited,cells_visited)].toarray()
        visitation_prob = 1 - (1-visitation_prob).prod(1)
        return(visitation_prob)

    def combine_probs(self, csx_mtx, rowcall, colcall):
        csx_mtx = csx_mtx[np.ix_(rowcall,colcall)].copy()
        rowidx, colidx = csx_mtx.nonzero() #note reindex
        cmdata = 1-csx_mtx.data #used for combine.probs function (1 - prod(1 - x))
        visitation_prob = [1-cmdata[(rowidx==irow)].prod() for irow in range(len(rowcall))] #(1-np.array([])).prod()=1
        return np.array(visitation_prob)

    def floodPseudotimeCalc(self, tm_flood, start_cells, min_flooded=2, irep=0):
        # Initialize with starting cells
        print("Starting flood number", irep)
        tm_flood = tm_flood.copy()
        
        all_cells = np.arange(tm_flood.shape[0])
        pseudotime = np.empty(len(all_cells))
        pseudotime[:] = np.nan
        pseudotime[start_cells] = 0
        
        cells_visited = start_cells
        cells_not_visited = np.setdiff1d(all_cells, cells_visited)
        cells_newly_visited = np.empty(min_flooded)

        i =0
        while ((len(cells_visited) < len(all_cells)-1) & 
            (len(cells_newly_visited) >= min_flooded)):
            i +=1
            print("Flooding step %s: %.2f = %s of %s cells visited"%(i, len(cells_visited)/len(all_cells), 
                                                                    len(cells_visited), len(all_cells)))
            # Calculate visitation probability for each cell
            visitation_prob = self.combine_probs(tm_flood, cells_not_visited, cells_visited)
            # Figure out which cells are newly visited
            cells_newly_visited = cells_not_visited[np.bool_(np.random.binomial(1, visitation_prob, size=len(visitation_prob)))]
            # Record the visited cells
            cells_visited = np.concatenate((cells_visited, cells_newly_visited), axis=0)
            cells_not_visited = np.setdiff1d(cells_not_visited, cells_newly_visited)
            pseudotime[cells_newly_visited] = i
        return(pseudotime)

    def floodPseudotime(self, sym_trans=None, rep=30, min_flooded=2, tm_flood=None, njobs=-1):
        sym_trans = self.adata.uns['diffmap_trans_sym'].copy() if sym_trans is None else sym_trans
        tm_flood = self.floodBuildTM(sym_trans)
        tm_flood = self.checkcsr(tm_flood)

        if njobs in [0,1]:
            floods = []
            for irep in range(rep):
                print("Starting flood number", irep)
                iflood = self.floodPseudotimeCalc(tm_flood, start_cells = self.root_cells,
                                            min_flooded = min_flooded)
                floods.append(iflood)
        else:
            floods = Parallel(n_jobs=njobs, verbose=1)(delayed(self.floodPseudotimeCalc)
                                                    (tm_flood, start_cells = self.root_cells, 
                                                        min_flooded = min_flooded, irep=irep) 
                            for irep in range(rep))
        self.floods = np.array(floods).T

    def floodPseudotimeProcess(self, max_frac_NA=0.7, stat_fun='mean', stability_div=10):
        floods =self.floods
        '''
        cell_bool = (np.isnan(floods).sum(1)/floods.shape[1] > max_frac_NA)
        print('%s cells will be removed with large NA values.' %(sum(cell_bool)))
        floods[cell_bool,:] = 0
        '''
    
        floods = floods/(np.nanmax(floods, axis=0)[np.newaxis,:])
        floods_division = np.ceil(np.arange(1, stability_div+1) / stability_div*floods.shape[1]).astype(int)

        self.walks_per_cell = np.array([ np.nansum(floods[:,:idiv], axis=1) for idiv in floods_division ]).T
        self.pseudotime_stability = np.array([ np.nanmean(floods[:,:idiv], axis=1) for idiv in floods_division ]).T  
        
        self.final_visit_freq = self.walks_per_cell[:, -1]
        self.final_visit_freq_log = np.log10(self.walks_per_cell[:,-1] + 1)
        self.final_pseudotime = self.pseudotime_stability[:,-1]

        print('add URD pseudotime in adata.obs["URD_pseudotime"]')
        print('add URD visit_freq in adata.obs["URD_visit_freq"]')
        self.adata.obs['URD_pseudotime'] = self.final_pseudotime
        self.adata.obs['URD_visit_freq'] = self.final_visit_freq

    def pseudotimePlotStabilityOverall(self):
        floods = self.floods
        pseudotime_stability = self.pseudotime_stability
        stability_rate = np.diff(pseudotime_stability, axis=1)
        Changes = np.abs(stability_rate).sum(0)
        Walk = np.linspace(1, floods.shape[1], pseudotime_stability.shape[1])[1:]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.plot(Walk, Changes, '-or')
        plt.show()

    def pseudotimeDetermineLogistic(self, tm_flood, pseudotime, optimal_cells_forward=20, max_cells_back=40,
                                    sort_dec=True, do_plot=True, print_values=True, asymptote=0.01):
        pseudotime_vec = np.sort(pseudotime)[::-1] if sort_dec else np.sort(pseudotime)

        mean_pseudotime_back = np.mean(pseudotime_vec[:-max_cells_back] - pseudotime_vec[max_cells_back:])
        mean_pseudotime_forward = np.mean(pseudotime_vec[optimal_cells_forward:] - pseudotime_vec[:-optimal_cells_forward])
        
        x0 =  np.mean([mean_pseudotime_back, mean_pseudotime_forward])
        k  =  np.log(asymptote) / (x0 - mean_pseudotime_forward)

        if do_plot:
            from scipy.stats import logistic
            import matplotlib.pyplot as plt
            x = np.linspace(2*mean_pseudotime_back, 2*mean_pseudotime_forward, 100)
            plt.scatter(x, self.sigmoid(x, x0, k), s=15, c='black')
            plt.axvline(x=0, color='red')
            plt.axvline(x=mean_pseudotime_back, color='blue')
            plt.axvline(x=mean_pseudotime_forward, color='blue')
            plt.xlabel("Delta pseudotime")
            plt.ylabel("Delta pseudotime")

        if print_values:
            print("Mean pseudotime back (~", max_cells_back, " cells) ", mean_pseudotime_back)
            print("Chance of accepted move to equal pseudotime is ", self.sigmoid(0, x0, k))
            print("Mean pseudotime forward (~", optimal_cells_forward, " cells) ", mean_pseudotime_forward)

        self.sigm_intpt = x0
        self.sigm_coef  = k

    def pseudotimeWeightTransitionMatrix(self, transitions=None, pseudotime=None, x0=None, k=None, max_records=225e6, nwidth=2000):
        x0 = self.sigm_intpt if x0 is None  else x0
        k  = self.sigm_coef  if k is None else k
        pseudotime = self.final_pseudotime if pseudotime is None else pseudotime
        transitions = self.adata.uns['diffmap_trans_sym'].copy() if transitions is None else transitions
        transitions = self.checkcsr(transitions)
        pseudotime_vec = pseudotime.copy()
        cells_at_a_time = np.floor(max_records / len(pseudotime_vec))
        if cells_at_a_time>= len(pseudotime_vec):
            cells_at_a_time = nwidth
        n_step = np.arange(0, len(pseudotime_vec), cells_at_a_time).astype(int)
        if n_step[-1] < len(pseudotime_vec):
            n_step = np.append(n_step, len(pseudotime_vec)).astype(int)
        print(self.Time(), "Processing in ", len(n_step)-1, " groups.")

        time_transition_matrices = []
        for _i in range(len(n_step)-1):
            print(self.Time(), "Do Processing ", _i, " groups.")
            _s, _e = n_step[_i], n_step[_i+1]
            ipseud = pseudotime_vec[_s: _e]
            itrans = transitions[:, _s:_e].copy()
            rowidx, colidx = itrans.nonzero()
            itdata = itrans.data
            ippseu = itrans.copy()
            ipdata = []
            for _n in range(len(pseudotime_vec)):
                ncol_idx = colidx[(rowidx==_n)]
                npseudo = ipseud[ncol_idx]
                nsigm = self.sigmoid((npseudo - pseudotime_vec[_n]), x0, k)
                ipdata.append(nsigm)
            ipdata = np.concatenate(ipdata)
            ippseu.data = ipdata
            ipseu_trans = ippseu.multiply(itrans)
            time_transition_matrices.append(ipseu_trans)

        import scipy.sparse as sp
        time_transition_matrices = sp.hstack(time_transition_matrices)
        if time_transition_matrices.shape[0] != time_transition_matrices.shape[1]:
            raise('Wrong dimension in final time transition matrix.')
        self.time_transition_matrices = time_transition_matrices
