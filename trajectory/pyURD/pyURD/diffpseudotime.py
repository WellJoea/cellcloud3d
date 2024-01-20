import numpy as np
from .utils import checkcsr, sigmoid, Time

def pseudotimeDetermineLogistic(URD, pseudotime=None, optimal_cells_forward=20, max_cells_back=40,
                                sort_dec=True, do_plot=True, print_values=True, asymptote=0.01):
    pseudotime = URD.pseud['floods_pseudotime'].copy() if pseudotime is None else pseudotime
    pseudotime_vec = np.sort(pseudotime)[::-1] if sort_dec else np.sort(pseudotime)

    mean_pseudotime_back = np.mean(pseudotime_vec[:-max_cells_back] - pseudotime_vec[max_cells_back:])
    mean_pseudotime_forward = np.mean(pseudotime_vec[optimal_cells_forward:] - pseudotime_vec[:-optimal_cells_forward])

    x0 =  np.mean([mean_pseudotime_back, mean_pseudotime_forward])
    k  =  np.log(asymptote) / (x0 - mean_pseudotime_forward)

    if do_plot:
        import matplotlib.pyplot as plt
        x = np.linspace(2*mean_pseudotime_back, 2*mean_pseudotime_forward, 100)
        plt.figure(figsize=(9,7))
        plt.scatter(x, sigmoid(x, x0, k), s=15, c='black')
        plt.axvline(x=0, color='red')
        plt.axvline(x=mean_pseudotime_back, color='blue')
        plt.axvline(x=mean_pseudotime_forward, color='blue')
        plt.xticks(rotation=45)
        plt.xlabel("Delta pseudotime")
        plt.ylabel("Delta pseudotime")

    if print_values:
        print("Mean pseudotime back (~", max_cells_back, " cells) ", mean_pseudotime_back)
        print("Chance of accepted move to equal pseudotime is ", sigmoid(0, x0, k))
        print("Mean pseudotime forward (~", optimal_cells_forward, " cells) ", mean_pseudotime_forward)

    URD.attr['sigm_intpt'] = x0
    URD.attr['sigm_coef']  = k

def pseudotimeWeightTransitionMatrix(URD, transitions=None, pseudotime=None, x0=None, k=None, max_records=225e6, nwidth=2000):
    x0 = URD.attr['sigm_intpt'] if x0 is None  else x0
    k  = URD.attr['sigm_coef']  if k is None else k
    pseudotime = URD.pseud['floods_pseudotime'].copy() if pseudotime is None else pseudotime
    transitions = URD.adata.obsp['diffmap_trans'].copy() if transitions is None else transitions
    transitions = checkcsr(transitions)
    pseudotime_vec = pseudotime.copy()

    cells_at_a_time = np.floor(max_records / len(pseudotime_vec))
    if cells_at_a_time>= len(pseudotime_vec):
        cells_at_a_time = nwidth
    n_step = np.arange(0, len(pseudotime_vec), cells_at_a_time).astype(int)
    if n_step[-1] < len(pseudotime_vec):
        n_step = np.append(n_step, len(pseudotime_vec)).astype(int)
    print(Time(), "Processing in ", len(n_step)-1, " groups.")

    time_transition_matrix = []
    for _i in range(len(n_step)-1):
        print(Time(), "Do Processing ", _i, " groups.")
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
            nsigm = sigmoid((npseudo - pseudotime_vec[_n]), x0, k)
            ipdata.append(nsigm)
        ipdata = np.concatenate(ipdata)
        ippseu.data = ipdata
        ipseu_trans = ippseu.multiply(itrans)
        time_transition_matrix.append(ipseu_trans)

    import scipy.sparse as sp
    time_transition_matrix = sp.hstack(time_transition_matrix)
    if time_transition_matrix.shape[0] != time_transition_matrix.shape[1]:
        raise('Wrong dimension in final time transition matrix.')

    print("add time_transition_matrix in adata.obsp['time_transition_matrix']")
    URD.adata.obsp['time_transition_matrix'] = checkcsr(time_transition_matrix)

