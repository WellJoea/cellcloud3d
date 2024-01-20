import numpy as np
def loadTipCells(URD, tips):
    tip_clust = URD.adata.obs[tips].copy()
    all_tips = tip_clust[~tip_clust.isna()].unique()
    cells_in_tip = { _i:np.flatnonzero(tip_clust==_i) for _i in all_tips}
    return (cells_in_tip)

def putativeCellsInSegment(URD, segments, minimum_visits, visit_threshold):
    visit_data = np.vstack([ URD.diff[f'tip_{segments}'] for _iseg in segments ]
    np.vstack([URD.diff['tip_3'], URD.diff['tip_2']])
    max_visit = URD.diff['tip_3']
    max_visit <- apply(visit_data, 1, max)

def buildTree(URD, pseudotime='URD_pseudotime', tips_use=None, divergence_method=["ks", "preference"][0],
              weighted_fusion=True, use_only_original_tips=True, cells_per_pseudotime_bin=80, 
              bins_per_pseudotime_window=5, minimum_visits=10, visit_threshold=0.7,
              save_breakpoint_plots=None, save_all_breakpoint_info=False, p_thresh=0.01,
              min_cells_per_segment=1, min_pseudotime_per_segment=0.01, dendro_node_size=100, 
              dendro_cell_jitter=0.15, dendro_cell_dist_to_tree=0.05):

    tips_use = [_k for _k in URD.diff.keys() if _k.startswith('tip_') ]
    # If you're going to do balanced fusion, calculate number of cells per tip.
    # If you're going to do balanced fusion, calculate number of cells per tip.
    if (weighted_fusion):
        tip_size = { _k:len(_v) for _k, _v in cells_in_tip.items()}

    URD.tree['pseudotime'] = URD.pseud[pseudotime]

