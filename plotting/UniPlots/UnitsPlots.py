import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42


def _setup_violin_axes_ticks(row_ax, _i, ncells, _g, ylim=[0,1], drop_spine=False, label_size=10):
    row_ax.set_ylim(ylim)
    if _i ==0:
        spines = ['bottom']
    elif _i ==ncells-1:
        spines = ['top']
    else:
        spines= ['top', 'bottom']
    if drop_spine:
        for spine in spines:
            row_ax.spines[spine].set_visible(False)
        
    if  _i !=ncells-1:
        row_ax.set_xticklabels([])
        row_ax.set_xlabel(None)
        row_ax.set_xticks([])
    else:
        row_ax.set_xticklabels(
            row_ax.get_xticklabels(), 
            rotation=90, 
            ha='center',
            va='center_baseline',
            fontsize=label_size,
        )
    row_ax.set_ylabel(_g, rotation=0, fontsize=label_size,
                      verticalalignment='baseline',
                      horizontalalignment='right')
    row_ax.tick_params(
        axis='y',
        left=False,
        right=True,
        labelright=True,
        labelleft=False,
        labelsize='x-small',
    )


def cross_stack_violin(_df, x_name, y_name, value_name, scale='count',
                       y_order = None, x_order=None,
                       figsize=(6,15),hspace=0, palette="Set3",
                       inner='box',saturation=0.6, alpha=0.9,
                       cut=0, width=0.6, linewidth=0.5,
                       point_size=1, jitter=0.15, drop_spine=False,
                       label_size=10,
                       show_violin=True, show_point=True,
                       orient='vertical', show=True, save=None, **kargs):
    import matplotlib.pyplot as plt
    if y_order is None:
        try:
            y_order = _df[y_name].cat.remove_unused_categories().cat.categories
        except:
            y_order = _df[y_name].unique()
            y_order = y_order[~(pd.isnull(y_order) | pd.isna(y_order))]   
    if x_order is None:
        try:
            x_order = _df[x_name].cat.remove_unused_categories().cat.categories
        except:
            x_order = _df[x_name].unique()
            x_order = x_order[~(pd.isnull(x_order) | pd.isna(x_order))]   

    _df = _df[((_df[x_name].isin(x_order)) & (_df[y_name].isin(y_order)))]
    _df[x_name] = pd.Categorical(_df[x_name], categories=x_order)
    _df[y_name] = pd.Categorical(_df[y_name], categories=y_order)

    ncols = 1
    nrows = len(y_order)
    ncells = nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for _i, _g in enumerate(y_order):
        _idf = _df[(_df[y_name]==_g)]

        if min(nrows, ncols) == 1:
            AX = axes[_i]
        else:
            AX = axes[_i//ncols,_i%ncols]
        if show_violin:
            row_ax = sns.violinplot(x=x_name, y=value_name,
                        data=_idf,
                        orient="vertical", 
                        scale=scale,
                        #inner=inner,
                        linewidth=linewidth,
                        palette=palette,
                        saturation=1,
                        #sharex=True, sharey=False,
                        cut=cut, width=width,
                        ax = AX, **kargs)
        if show_point:
            row_ax = sns.stripplot(x=x_name, y=value_name, data=_idf, 
                                   size=point_size, edgecolor=None, 
                                   palette=palette,
                                   #color='grey',
                                   linewidth=0, jitter=jitter, zorder=1,
                                   alpha=alpha, 
                                   ax=AX, 
                                   **kargs)

        _setup_violin_axes_ticks(row_ax, _i, ncells, _g, drop_spine=drop_spine, label_size=label_size)
    plt.subplots_adjust(hspace=0)

    if not save is None:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return fig, axess