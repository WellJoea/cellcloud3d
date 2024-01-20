import numpy as np
from joblib import Parallel, delayed
import numpy as np
from scipy.sparse import csr_matrix,bsr_matrix,csc_matrix, csgraph
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc,isspmatrix_coo

from .utils import checkcsr
def floodBuildTM(dm_symtrans):
    tm_flood = dm_symtrans.copy()
    tm_sum_max = tm_flood.sum(1).max()
    tm_flood = tm_flood / tm_sum_max
    return(tm_flood)

def combine_probs0(tm_flood, cells_not_visited, cells_visited):
    visitation_prob = tm_flood[np.ix_(cells_not_visited,cells_visited)].toarray()
    visitation_prob = 1 - (1-visitation_prob).prod(1)
    return(visitation_prob)

def combine_probs(csx_mtx, rowcall, colcall):
    csx_mtx = csx_mtx[np.ix_(rowcall,colcall)].copy()
    rowidx, colidx = csx_mtx.nonzero() #note reindex
    cmdata = 1-csx_mtx.data #used for combine.probs function (1 - prod(1 - x))
    visitation_prob = [1-cmdata[(rowidx==irow)].prod() for irow in range(len(rowcall))] #(1-np.array([])).prod()=1
    return np.array(visitation_prob)

def trimvisites(visitation_prob, n_sample=20):
    visites= np.random.binomial(n_sample, visitation_prob, size=len(visitation_prob))
    visiteidx = visites.nonzero()[0]
    if n_sample>1:
        nvisite = visites[visites.nonzero()]
        svisites= np.bool_(np.random.binomial(1, nvisite/n_sample, size=len(nvisite)))
        #np.random.choice
        visiteidx=visiteidx[svisites]
    return visiteidx

def floodPseudotimeCalc(tm_flood, start_cells, min_flooded=2, irep=0, n_sample=15):
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
        visitation_prob = combine_probs(tm_flood, cells_not_visited, cells_visited)
        # Figure out which cells are newly visited
        #cells_newly_visited = cells_not_visited[np.bool_(np.random.binomial(1, visitation_prob, size=len(visitation_prob)))]
        cells_newly_visited = cells_not_visited[trimvisites(visitation_prob, n_sample=n_sample)]
        # Record the visited cells
        cells_visited = np.concatenate((cells_visited, cells_newly_visited), axis=0)
        cells_not_visited = np.setdiff1d(cells_not_visited, cells_newly_visited)
        pseudotime[cells_newly_visited] = i
    return(pseudotime)

def floodPseudotime(URD, root_cells=None, sym_trans=None, rep=60, min_flooded=2, tm_flood=None, n_jobs=-1, n_sample=20):
    sym_trans = URD.adata.obsp['diffmap_trans'].copy() if sym_trans is None else sym_trans
    tm_flood = floodBuildTM(sym_trans)
    tm_flood = checkcsr(tm_flood)

    URD.root_cells = root_cells if not root_cells is None else URD.root_cells
    if n_jobs in [0,1]:
        floods = []
        for irep in range(rep):
            print("Starting flood number", irep)
            iflood = floodPseudotimeCalc(tm_flood, start_cells = URD.root_cells,
                                        min_flooded = min_flooded, n_sample=n_sample)
            floods.append(iflood)
    else:
        floods = Parallel(n_jobs=n_jobs, verbose=1)(delayed(floodPseudotimeCalc)
                                                (tm_flood, start_cells = URD.root_cells, 
                                                    min_flooded = min_flooded, irep=irep, n_sample=n_sample) 
                        for irep in range(rep))
    URD.floods = np.array(floods).T

def floodPseudotimeProcess(URD, max_frac_NA=0.7, stat_fun='mean', stability_div=10):
    floods =URD.floods.copy()
    '''
    cell_bool = (np.isnan(floods).sum(1)/floods.shape[1] > max_frac_NA)
    print('%s cells will be removed with large NA values.' %(sum(cell_bool)))
    floods[cell_bool,:] = 0
    '''

    floods = floods/(np.nanmax(floods, axis=0)[np.newaxis,:])
    floods_division = np.ceil(np.arange(1, stability_div+1) / stability_div*floods.shape[1]).astype(int)

    #walks_per_cell = np.array([ np.nansum(floods[:,:idiv], axis=1) for idiv in floods_division ]).T
    walks_per_cell = np.array([ np.sum(~np.isnan(floods[:,:idiv]), axis=1) for idiv in floods_division ]).T
    pseudotime_stability = np.array([ np.nanmean(floods[:,:idiv], axis=1) for idiv in floods_division ]).T

    final_visit_freq = walks_per_cell[:, -1]
    final_visit_freq_log = np.log10(walks_per_cell[:,-1] + 1)
    final_pseudotime = pseudotime_stability[:,-1]

    URD.stable['floods_walks_per_cell'] = walks_per_cell
    URD.stable['floods_pseudotime_stability'] = pseudotime_stability
    URD.stable['floods_division'] = floods_division

    URD.diff['floods_visit_freq'] = final_visit_freq
    URD.pseud['floods_pseudotime'] = final_pseudotime

    print('add floods pseudotime in adata.obs["URD_pseudotime"]')
    print('add floods visit_freq in adata.obs["URD_visit_freq"]')
    URD.adata.obs['URD_pseudotime'] = final_pseudotime
    URD.adata.obs['URD_visit_freq'] = final_visit_freq

def pseudotimePlotStabilityOverall(URD):
    floods = URD.floods
    pseudotime_stability = URD.stable['floods_pseudotime_stability']
    stability_rate = np.diff(pseudotime_stability, axis=1)
    Changes = np.abs(stability_rate).sum(0)
    Walk = np.linspace(1, floods.shape[1], pseudotime_stability.shape[1])[1:]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,7))
    plt.plot(Walk, Changes, '-or')
    plt.xticks(rotation=45)
    plt.show()
