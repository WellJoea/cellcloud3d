import numpy as np
from termcolor import colored
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, bsr_matrix, csc_matrix, csgraph
from .utils import scale_prob, checkcsr, Time
        
def simulateRandomWalki(start_cells=None, end_cells=None,  transition_matrix=None, irep=0,
                        end_visits=1, trans_step=1, max_steps=1000):
    current_cell = np.random.choice(start_cells, size=trans_step, replace = False)
    diffusion_path = current_cell
    stops_in_endzone = 0
    n_steps = 0
    while stops_in_endzone < end_visits:
        current_cell = np.random.choice(range(transition_matrix.shape[0]), size=trans_step, replace =False, 
                                        p=scale_prob(transition_matrix[current_cell,:].toarray()).flatten())
        diffusion_path = np.concatenate([diffusion_path, current_cell])

        if np.isin(current_cell, end_cells).any():
            stops_in_endzone +=1

        n_steps +=1
        if (n_steps > max_steps):
            print(colored(f"Warning: Walk {irep} length greater than {max_steps} so returning NULL!", "red"))
            return(None)
    return(diffusion_path)

def processRandomWalki(URD, walks, walks_name, ncells, aggregate_fun='np.mean', n_subsample=10, copy=False, save_tipswalk=True):
    walklen = len(walks)
    walks = [ _i for _i in walks if not _i is None ]
    print(f'{Time()} Tip_{walks_name} get {len(walks)} vilid path in all {walklen} walks')

    walks_in_division = np.ceil(np.arange(1, n_subsample+1) / n_subsample * len(walks)).astype(int)
    walk_lengths = list(map(len, walks))

    walks_flat = np.concatenate(walks)
    hops_flat  = np.concatenate([np.linspace(0,1,len(_iw)+1, endpoint=True)[1:] for _iw in walks])

    rowidx, colidx, pdata, wdata= [],[],[],[]
    for section in range(n_subsample):
        flat_rows = sum(walk_lengths[:walks_in_division[section]])
        visit_freq = walks_flat[:flat_rows]
        visit_rank = hops_flat[:flat_rows]

        unique_, counts_ = np.unique(visit_freq, return_counts=True)
        hops_relative = [eval(aggregate_fun)(visit_rank[visit_freq==_iu]) for _iu in unique_]
        rowidx =np.concatenate([rowidx, unique_])
        colidx =np.concatenate([colidx, np.repeat(section, len(unique_))])
        wdata  =np.concatenate([wdata, counts_])
        pdata  =np.concatenate([pdata, hops_relative])
    pseudotime_stability = csr_matrix((pdata, (rowidx, colidx)),  shape=(ncells, n_subsample))
    walks_per_cell = csr_matrix((wdata, (rowidx, colidx)),  shape=(ncells, n_subsample))

    if save_tipswalk:
        URD.stable[f'tip_{walks_name}_walks_per_cell'] = walks_per_cell
        URD.stable[f'tip_{walks_name}_pseudotime_stability'] = pseudotime_stability
        URD.stable[f'tip_{walks_name}_division'] = walks_in_division

    final_visit_freq = walks_per_cell[:, -1].toarray().flatten()
    final_pseudotime = pseudotime_stability[:, -1].toarray().flatten()
    final_pseudotime[(final_visit_freq==0)] = np.nan
    URD.pseud[f'tip_{walks_name}'] = final_pseudotime
    URD.diff[f'tip_{walks_name}'] = final_visit_freq
    return URD if copy else None

def simulateRandomWalksFromTips(URD, tip_clust, root_cells, transition_matrix=None, n_per_tip=10000,
                                root_visits=1, max_steps=None, n_jobs=-1, copy=False, save_tipswalk=True):
    transition_matrix = URD.adata.obsp['time_transition_matrix'].copy() if transition_matrix is None else transition_matrix
    transition_matrix = checkcsr(transition_matrix)

    max_steps = URD.adata.X.shape[1] if max_steps is None else max_steps
    all_tips = tip_clust[~tip_clust.isna()].unique()

    tip_walks = {}
    for iclut in all_tips:
        print(Time(), " - Starting random walks from tip ", iclut)
        icells = np.flatnonzero(tip_clust==iclut)
        if n_jobs in [0,1]:
            walks  = [ simulateRandomWalki(start_cells=icells, 
                                        transition_matrix=transition_matrix, 
                                        end_cells=root_cells, 
                                        irep=_i, 
                                        end_visits=root_visits, 
                                        max_steps=max_steps)
                        for _i in range(n_per_tip) ]
        else:
            walks = Parallel(n_jobs=n_jobs, verbose=1)(delayed(simulateRandomWalki)
                               (start_cells=icells, 
                                transition_matrix=transition_matrix, 
                                end_cells=root_cells, 
                                irep=_i, 
                                end_visits=root_visits, 
                                max_steps=max_steps)
                            for _i in range(n_per_tip))

        processRandomWalki(URD, walks, iclut, transition_matrix.shape[0], aggregate_fun='np.mean',
                            n_subsample=10, save_tipswalk=save_tipswalk)
        print(f"add tip_{iclut}*log10 in adata.obs")
        URD.adata.obs[f'tip_{iclut}_pseud'] = URD.pseud[f'tip_{iclut}']
        URD.adata.obs[f'tip_{iclut}_walk_log10']  = np.log10(URD.diff[f'tip_{iclut}']+1)
    return URD if copy else None
