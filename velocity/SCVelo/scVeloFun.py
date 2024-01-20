from scvelo.plotting.docs import doc_scatter, doc_params
from scvelo.plotting.utils import *

from inspect import signature
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from pandas import unique, Index

def velocity(
    adata,
    var_names=None,
    basis=None,
    vkey="velocity",
    mode=None,
    fits=None,
    layers="all",
    color=None,
    color_map=None,
    colorbar=True,
    perc=[2, 98],
    alpha=0.5,
    size=None,
    groupby=None,
    groups=None,
    legend_loc="none",
    legend_fontsize=8,
    use_raw=False,
    fontsize=None,
    figsize=None,
    dpi=None,
    show=None,
    save=None,
    ax=None,
    ncols=None,
    wspace=0.9,
    hspace=0.9,
    **kwargs,
):
    """Phase and velocity plot for set of genes.

    The phase plot shows spliced against unspliced expressions with steady-state fit.
    Further the embedding is shown colored by velocity and expression.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    var_names: `str` or list of `str` (default: `None`)
        Which variables to show.
    basis: `str` (default: `'umap'`)
        Key for embedding coordinates.
    mode: `'stochastic'` or `None` (default: `None`)
        Whether to show show covariability phase portrait.
    fits: `str` or list of `str` (default: `['velocity', 'dynamics']`)
        Which steady-state estimates to show.
    layers: `str` or list of `str` (default: `'all'`)
        Which layers to show.
    color: `str`,  list of `str` or `None` (default: `None`)
        Key for annotations of observations/cells or variables/genes
    color_map: `str` or tuple (default: `['RdYlGn', 'gnuplot_r']`)
        String denoting matplotlib color map. If tuple is given, first and latter
        color map correspond to velocity and expression, respectively.
    perc: tuple, e.g. [2,98] (default: `[2,98]`)
        Specify percentile for continuous coloring.
    groups: `str`, `list` (default: `None`)
        Subset of groups, e.g. [‘g1’, ‘g2’], to which the plot shall be restricted.
    groupby: `str`, `list` or `np.ndarray` (default: `None`)
        Key of observations grouping to consider.
    legend_loc: str (default: 'none')
        Location of legend, either 'on data', 'right margin'
        or valid keywords for matplotlib.legend.
    size: `float` (default: 5)
        Point size.
    alpha: `float` (default: 1)
        Set blending - 0 transparent to 1 opaque.
    fontsize: `float` (default: `None`)
        Label font size.
    figsize: tuple (default: `(7,5)`)
        Figure size.
    dpi: `int` (default: 80)
        Figure dpi.
    show: `bool`, optional (default: `None`)
        Show the plot, do not return axis.
    save: `bool` or `str`, optional (default: `None`)
        If `True` or a `str`, save the figure. A string is appended to the default
        filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}.
    ax: `matplotlib.Axes`, optional (default: `None`)
        A matplotlib axes object. Only works if plotting a single component.
    ncols: `int` or `None` (default: `None`)
        Number of columns to arange multiplots into.

    """
    from scvelo import settings
    from scvelo.preprocessing.moments import second_order_moments
    from scvelo.tools.rank_velocity_genes import rank_velocity_genes
    from scvelo.plotting.scatter import scatter
    from scvelo.plotting.utils import (
        savefig_or_show,
        default_basis,
        default_size,
        get_basis,
        get_figure_params,
    )

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as pl
    from matplotlib import rcParams
    from scipy.sparse import issparse


    basis = default_basis(adata) if basis is None else get_basis(adata, basis)
    color, color_map = kwargs.pop("c", color), kwargs.pop("cmap", color_map)
    if fits is None:
        fits = ["velocity", "dynamics"]
    if color_map is None:
        color_map = ["RdYlGn", "gnuplot_r"]

    if isinstance(groupby, str) and groupby in adata.obs.keys():
        if (
            "rank_velocity_genes" not in adata.uns.keys()
            or adata.uns["rank_velocity_genes"]["params"]["groupby"] != groupby
        ):
            rank_velocity_genes(adata, vkey=vkey, n_genes=10, groupby=groupby)
        names = np.array(adata.uns["rank_velocity_genes"]["names"].tolist())
        if groups is None:
            var_names = names[:, 0]
        else:
            groups = [groups] if isinstance(groups, str) else groups
            categories = adata.obs[groupby].cat.categories
            idx = np.array([any([g in group for g in groups]) for group in categories])
            var_names = np.hstack(names[idx, : int(10 / idx.sum())])
    elif var_names is not None:
        if isinstance(var_names, str):
            var_names = [var_names]
        else:
            var_names = [var for var in var_names if var in adata.var_names]
    else:
        raise ValueError("No var_names or groups specified.")
    var_names = pd.unique(var_names)

    if use_raw or "Ms" not in adata.layers.keys():
        skey, ukey = "spliced", "unspliced"
    else:
        skey, ukey = "Ms", "Mu"
    layers = [vkey, skey] if layers == "all" else layers
    layers = [layer for layer in layers if layer in adata.layers.keys() or layer == "X"]

    fits = list(adata.layers.keys()) if fits == "all" else fits
    fits = [fit for fit in fits if f"{fit}_gamma" in adata.var.keys()] + ["dynamics"]
    stochastic_fits = [fit for fit in fits if f"variance_{fit}" in adata.layers.keys()]

    nplts = 1 + len(layers) + (mode == "stochastic") * 2
    ncols = 1 if ncols is None else ncols
    nrows = int(np.ceil(len(var_names) / ncols))
    ncols = int(ncols * nplts)
    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    figsize, dpi = get_figure_params(figsize, dpi, ncols / 2)
    if ax is None:
        gs_figsize = (figsize[0] * ncols / 2, figsize[1] * nrows / 2)
        ax = pl.figure(figsize=gs_figsize, dpi=dpi)
    gs = pl.GridSpec(nrows, ncols, wspace=wspace, hspace=wspace)

    # half size, since fontsize is halved in width and height
    size = default_size(adata) / 2 if size is None else size
    fontsize = rcParams["font.size"] * 0.8 if fontsize is None else fontsize

    scatter_kwargs = dict(colorbar=colorbar, perc=perc, size=size, use_raw=use_raw)
    scatter_kwargs.update(dict(fontsize=fontsize, legend_fontsize=legend_fontsize))

    for v, var in enumerate(var_names):
        _adata = adata[:, var]
        s, u = _adata.layers[skey], _adata.layers[ukey]
        if issparse(s):
            s, u = s.A, u.A

        # spliced/unspliced phase portrait with steady-state estimate
        ax = pl.subplot(gs[v * nplts])
        cmap = color_map
        if isinstance(color_map, (list, tuple)):
            cmap = color_map[-1] if color in ["X", skey] else color_map[0]
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = "spliced"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "unspliced"
        legend_loc_lines = "none" if v < len(var_names) - 1 else legend_loc

        scatter(
            adata,
            basis=var,
            color=color,
            color_map=cmap,
            frameon=True,
            title=var,
            alpha=alpha,
            vkey=fits,
            show=False,
            ax=ax,
            save=False,
            legend_loc_lines=legend_loc_lines,
            **scatter_kwargs,
            **kwargs,
        )

        # velocity and expression plots
        for l, layer in enumerate(layers):
            ax = pl.subplot(gs[v * nplts + l + 1])
            title = "expression" if layer in ["X", skey] else layer
            # _kwargs = {} if title == 'expression' else kwargs
            cmap = color_map
            if isinstance(color_map, (list, tuple)):
                cmap = color_map[-1] if layer in ["X", skey] else color_map[0]
            scatter(
                adata,
                basis=basis,
                color=var,
                layer=layer,
                title=title,
                color_map=cmap,
                alpha=alpha,
                frameon=False,
                show=False,
                ax=ax,
                save=False,
                **scatter_kwargs,
                **kwargs,
            )

        if mode == "stochastic":
            ss, us = second_order_moments(_adata)
            s, u, ss, us = s.flatten(), u.flatten(), ss.flatten(), us.flatten()
            fit = stochastic_fits[0]

            ax = pl.subplot(gs[v * nplts + len(layers) + 1])
            beta, offset = 1, 0
            if f"{fit}_beta" in adata.var.keys():
                beta = _adata.var[f"{fit}_beta"]
            if f"{fit}_offset" in adata.var.keys():
                offset = _adata.var[f"{fit}_offset"]
            x = np.array(2 * (ss - s ** 2) - s)
            y = np.array(2 * (us - u * s) + u + 2 * s * offset / beta)
            kwargs["xlabel"] = r"2 $\Sigma_s - \langle s \rangle$"
            kwargs["ylabel"] = r"2 $\Sigma_{us} + \langle u \rangle$"
            scatter(
                adata,
                x=x,
                y=y,
                color=color,
                title=var,
                frameon=True,
                ax=ax,
                save=False,
                show=False,
                **scatter_kwargs,
                **kwargs,
            )

            xnew = np.linspace(np.min(x), np.max(x) * 1.02)
            for fit in stochastic_fits:
                gamma, beta, offset2 = 1, 1, 0
                if f"{fit}_gamma" in adata.var.keys():
                    gamma = _adata.var[f"{fit}_gamma"].values
                if f"{fit}_beta" in adata.var.keys():
                    beta = _adata.var[f"{fit}_beta"].values
                if f"{fit}_offset2" in adata.var.keys():
                    offset2 = _adata.var[f"{fit}_offset2"].values
                ynew = gamma / beta * xnew + offset2 / beta
                pl.plot(xnew, ynew, c="k", linestyle="--")

    savefig_or_show(dpi=dpi, save=save, show=show)
    if show is False:
        return ax

@doc_params(scatter=doc_scatter)
def scatter(
    adata=None,
    basis=None,
    x=None,
    y=None,
    vkey=None,
    color=None,
    use_raw=None,
    layer=None,
    color_map=None,
    colorbar=None,
    palette=None,
    size=None,
    alpha=None,
    linewidth=None,
    linecolor=None,
    perc=None,
    groups=None,
    sort_order=True,
    components=None,
    projection=None,
    legend_loc=None,
    legend_loc_lines=None,
    legend_fontsize=None,
    legend_fontweight=None,
    legend_fontoutline=None,
    xlabel=None,
    ylabel=None,
    title=None,
    fontsize=None,
    figsize=None,
    figscale=None,
    xlim=None,
    ylim=None,
    add_density=None,
    add_assignments=None,
    add_linfit=None,
    add_polyfit=None,
    add_rug=None,
    add_text=None,
    add_text_pos=None,
    add_outline=None,
    outline_width=None,
    outline_color=None,
    n_convolve=None,
    smooth=None,
    rescale_color=None,
    color_gradients=None,
    dpi=None,
    frameon=None,
    zorder=None,
    ncols=None,
    nrows=None,
    wspace=None,
    hspace=None,
    werror=0.2,
    show=None,
    save=None,
    ax=None,
    multikeynum=100,
    **kwargs,
):
    """\
    Scatter plot along observations or variables axes.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    x: `str`, `np.ndarray` or `None` (default: `None`)
        x coordinate
    y: `str`, `np.ndarray` or `None` (default: `None`)
        y coordinate
    {scatter}

    Returns
    -------
        If `show==False` a `matplotlib.Axis`
    """

    if adata is None and (x is not None and y is not None):
        adata = AnnData(np.stack([x, y]).T)

    # restore old conventions
    add_assignments = kwargs.pop("show_assignments", add_assignments)
    add_linfit = kwargs.pop("show_linear_fit", add_linfit)
    add_polyfit = kwargs.pop("show_polyfit", add_polyfit)
    add_density = kwargs.pop("show_density", add_density)
    add_rug = kwargs.pop("rug", add_rug)
    basis = kwargs.pop("var_names", basis)

    # keys for figures (fkeys) and multiple plots (mkeys)
    fkeys = ["adata", "show", "save", "groups", "ncols", "nrows", "wspace", "hspace"]
    fkeys += ["ax", "kwargs"]
    mkeys = ["color", "layer", "basis", "components", "x", "y", "xlabel", "ylabel"]
    mkeys += ["title", "color_map", "add_text"]
    scatter_kwargs = {"show": False, "save": False}
    for key in signature(scatter).parameters:
        if key not in mkeys + fkeys:
            scatter_kwargs[key] = eval(key)
    mkwargs = {}
    for key in mkeys:  # mkwargs[key] = key for key in mkeys
        mkwargs[key] = eval("{0}[0] if is_list({0}) else {0}".format(key))

    # use c & color and cmap & color_map interchangeably,
    # and plot each group separately if groups is 'all'
    if "c" in kwargs:
        color = kwargs.pop("c")
    if "cmap" in kwargs:
        color_map = kwargs.pop("cmap")
    if "rasterized" not in kwargs:
        kwargs["rasterized"] = settings._vector_friendly
    if isinstance(color_map, (list, tuple)) and all(
        [is_color_like(c) or c == "transparent" for c in color_map]
    ):
        color_map = rgb_custom_colormap(colors=color_map)
    if isinstance(groups, str) and groups == "all":
        if color is None:
            color = default_color(adata)
        if is_categorical(adata, color):
            vc = adata.obs[color].value_counts()
            groups = [[c] for c in vc[vc > 0].index]
    if isinstance(add_text, (list, tuple, np.ndarray, np.record)):
        add_text = list(np.array(add_text, dtype=str))

    # create list of each mkey and check if all bases are valid.
    color = to_list(color, max_len=None)
    layer, components = to_list(layer), to_list(components)
    x, y, basis = to_list(x), to_list(y), to_valid_bases_list(adata, basis)

    # get multikey (with more than one element)
    multikeys = eval(f"[{','.join(mkeys)}]")
    if is_list_of_list(groups):
        multikeys.append(groups)
    key_lengths = np.array([len(key) if is_list(key) else 1 for key in multikeys])
    multikey = (
        multikeys[np.where(key_lengths > 1)[0][0]] if np.max(key_lengths) > 1 else None
    )

    # gridspec frame for plotting multiple keys (mkeys: list or tuple)
    if multikey is not None:
        if np.sum(key_lengths > 1) == 1 and is_list_of_str(multikey):
            multikey = unique(multikey)  # take unique set if no more than one multikey
        if len(multikey) > multikeynum:
            raise ValueError(f"Please restrict the passed list to max {multikeynum} elements.")
        if ax is not None:
            logg.warn("Cannot specify `ax` when plotting multiple panels.")
        if is_list(title):
            title *= int(np.ceil(len(multikey) / len(title)))
        if nrows is None:
            ncols = len(multikey) if ncols is None else min(len(multikey), ncols)
            nrows = int(np.ceil(len(multikey) / ncols))
        else:
            ncols = int(np.ceil(len(multikey) / nrows))
        if not frameon:
            lloc, llines = "legend_loc", "legend_loc_lines"
            if lloc in scatter_kwargs and scatter_kwargs[lloc] is None:
                scatter_kwargs[lloc] = "none"
            if llines in scatter_kwargs and scatter_kwargs[llines] is None:
                scatter_kwargs[llines] = "none"
        print(nrows, ncols,1111111)
        grid_figsize, dpi = get_figure_params(figsize, dpi, ncols)
        grid_figsize = (grid_figsize[0] * ncols, grid_figsize[1] * nrows)
        grid_figsize = ((figscale+werror)* ncols, figscale * nrows) if figscale  else grid_figsize
        fig = pl.figure(None, grid_figsize, dpi=dpi)
        hspace = 0.3 if hspace is None else hspace
        gspec = pl.GridSpec(nrows, ncols, fig, hspace=hspace, wspace=wspace)

        ax = []
        for i, gs in enumerate(gspec):
            if i < len(multikey):
                g = groups[i * (len(groups) > i)] if is_list_of_list(groups) else groups
                multi_kwargs = {"groups": g}
                for key in mkeys:  # multi_kwargs[key] = key[i] if is multikey else key
                    multi_kwargs[key] = eval(
                        "{0}[i * (len({0}) > i)] if is_list({0}) else {0}".format(key)
                    )
                print(multi_kwargs)
                ax.append(
                    scatter(
                        adata,
                        ax=pl.subplot(gs),
                        **multi_kwargs,
                        **scatter_kwargs,
                        **kwargs,
                    )
                )

        if not frameon and isinstance(ylabel, str):
            set_label(xlabel, ylabel, fontsize, ax=ax[0], fontweight="bold")
        savefig_or_show(dpi=dpi, save=save, show=show)
        if show is False:
            return ax

    else:
        # make sure that there are no more lists, e.g. ['clusters'] becomes 'clusters'
        color_map = to_val(color_map)
        color, layer, basis = to_val(color), to_val(layer), to_val(basis)
        x, y, components = to_val(x), to_val(y), to_val(components)
        xlabel, ylabel, title = to_val(xlabel), to_val(ylabel), to_val(title)

        # multiple plots within one ax for comma-separated y or layers (string).

        if any([isinstance(key, str) and "," in key for key in [y, layer]]):
            # comma split
            y, layer, color = [
                [k.strip() for k in key.split(",")]
                if isinstance(key, str) and "," in key
                else to_list(key)
                for key in [y, layer, color]
            ]
            multikey = y if len(y) > 1 else layer if len(layer) > 1 else None

            if multikey is not None:
                for i, mi in enumerate(multikey):
                    ax = scatter(
                        adata,
                        x=x,
                        y=y[i * (len(y) > i)],
                        color=color[i * (len(color) > i)],
                        layer=layer[i * (len(layer) > i)],
                        basis=basis,
                        components=components,
                        groups=groups,
                        xlabel=xlabel,
                        ylabel="expression" if ylabel is None else ylabel,
                        color_map=color_map,
                        title=y[i * (len(y) > i)] if title is None else title,
                        ax=ax,
                        **scatter_kwargs,
                    )
                if legend_loc is None:
                    legend_loc = "best"
                if legend_loc and legend_loc != "none":
                    multikey = [key.replace("Ms", "spliced") for key in multikey]
                    multikey = [key.replace("Mu", "unspliced") for key in multikey]
                    ax.legend(multikey, fontsize=legend_fontsize, loc=legend_loc)

                savefig_or_show(dpi=dpi, save=save, show=show)
                if show is False:
                    return ax

        elif color_gradients is not None and color_gradients is not False:
            vals, names, color, scatter_kwargs = gets_vals_from_color_gradients(
                adata, color, **scatter_kwargs
            )
            cols = zip(adata.obs[color].cat.categories, adata.uns[f"{color}_colors"])
            c_colors = {cat: col for (cat, col) in cols}
            mkwargs.pop("color")
            ax = scatter(
                adata,
                color="grey",
                ax=ax,
                **mkwargs,
                **get_kwargs(scatter_kwargs, {"alpha": 0.05}),
            )  # background
            ax = scatter(
                adata,
                color=color,
                ax=ax,
                **mkwargs,
                **get_kwargs(scatter_kwargs, {"s": 0}),
            )  # set legend
            sorted_idx = np.argsort(vals, 1)[:, ::-1][:, :2]
            for id0 in range(len(names)):
                for id1 in range(id0 + 1, len(names)):
                    cmap = rgb_custom_colormap(
                        [c_colors[names[id0]], "white", c_colors[names[id1]]],
                        alpha=[1, 0, 1],
                    )
                    mkwargs.update({"color_map": cmap})
                    c_vals = np.array(vals[:, id1] - vals[:, id0]).flatten()
                    c_bool = np.array([id0 in c and id1 in c for c in sorted_idx])
                    if np.sum(c_bool) > 1:
                        _adata = adata[c_bool] if np.sum(~c_bool) > 0 else adata
                        mkwargs["color"] = c_vals[c_bool]
                        ax = scatter(
                            _adata, ax=ax, **mkwargs, **scatter_kwargs, **kwargs
                        )
            savefig_or_show(dpi=dpi, save=save, show=show)
            if show is False:
                return ax

        # actual scatter plot
        else:
            # set color, color_map, edgecolor, basis, linewidth, frameon, use_raw
            if color is None:
                color = default_color(adata, add_outline)
            if "cmap" not in kwargs:
                kwargs["cmap"] = (
                    default_color_map(adata, color) if color_map is None else color_map
                )
            if "s" not in kwargs:
                kwargs["s"] = default_size(adata) if size is None else size
            if "edgecolor" not in kwargs:
                kwargs["edgecolor"] = "none"
            is_embedding = ((x is None) | (y is None)) and basis not in adata.var_names
            if basis is None and is_embedding:
                basis = default_basis(adata)
            if linewidth is None:
                linewidth = 1
            if linecolor is None:
                linecolor = "k"
            if frameon is None:
                frameon = True if not is_embedding else settings._frameon
            if isinstance(groups, str):
                groups = [groups]
            if use_raw is None and basis not in adata.var_names:
                use_raw = layer is None and adata.raw is not None
            if projection == "3d":
                from mpl_toolkits.mplot3d import Axes3D

            ax, show = get_ax(ax, show, figsize, dpi, projection)

            # phase portrait: get x and y from .layers (e.g. spliced vs. unspliced)
            if basis in adata.var_names:
                if title is None:
                    title = basis
                if x is None and y is None:
                    x = default_xkey(adata, use_raw=use_raw)
                    y = default_ykey(adata, use_raw=use_raw)
                elif x is None or y is None:
                    raise ValueError("Both x and y have to specified.")
                if isinstance(x, str) and isinstance(y, str):
                    layers_keys = list(adata.layers.keys()) + ["X"]
                    if any([key not in layers_keys for key in [x, y]]):
                        raise ValueError("Could not find x or y in layers.")

                    if xlabel is None:
                        xlabel = x
                    if ylabel is None:
                        ylabel = y

                    x = get_obs_vector(adata, basis, layer=x, use_raw=use_raw)
                    y = get_obs_vector(adata, basis, layer=y, use_raw=use_raw)

                if legend_loc is None:
                    legend_loc = "none"

                if use_raw and perc is not None:
                    ub = np.percentile(x, 99.9 if not isinstance(perc, int) else perc)
                    ax.set_xlim(right=ub * 1.05)
                    ub = np.percentile(y, 99.9 if not isinstance(perc, int) else perc)
                    ax.set_ylim(top=ub * 1.05)

                # velocity model fits (full dynamics and steady-state ratios)
                if any(["gamma" in key or "alpha" in key for key in adata.var.keys()]):
                    plot_velocity_fits(
                        adata,
                        basis,
                        vkey,
                        use_raw,
                        linewidth,
                        linecolor,
                        legend_loc_lines,
                        legend_fontsize,
                        add_assignments,
                        ax=ax,
                    )

            # embedding: set x and y to embedding coordinates
            elif is_embedding:
                X_emb = adata.obsm[f"X_{basis}"][:, get_components(components, basis)]
                x, y = X_emb[:, 0], X_emb[:, 1]
                # todo: 3d plotting
                # z = X_emb[:, 2] if projection == "3d" and X_emb.shape[1] > 2 else None

            elif isinstance(x, str) and isinstance(y, str):
                var_names = (
                    adata.raw.var_names
                    if use_raw and adata.raw is not None
                    else adata.var_names
                )
                if layer is None:
                    layer = default_xkey(adata, use_raw=use_raw)
                x_keys = list(adata.obs.keys()) + list(adata.layers.keys())
                is_timeseries = y in var_names and x in x_keys
                if xlabel is None:
                    xlabel = x
                if ylabel is None:
                    ylabel = layer if is_timeseries else y
                if title is None:
                    title = y if is_timeseries else color
                if legend_loc is None:
                    legend_loc = "none"

                # gene trend: x and y as gene along obs/layers (e.g. pseudotime)
                if is_timeseries:
                    x = (
                        adata.obs[x]
                        if x in adata.obs.keys()
                        else adata.obs_vector(y, layer=x)
                    )
                    y = get_obs_vector(adata, basis=y, layer=layer, use_raw=use_raw)
                # get x and y from var_names, var or obs
                else:
                    if x in var_names and y in var_names:
                        if layer in adata.layers.keys():
                            x = adata.obs_vector(x, layer=layer)
                            y = adata.obs_vector(y, layer=layer)
                        else:
                            data = adata.raw if use_raw else adata
                            x, y = data.obs_vector(x), data.obs_vector(y)
                    elif x in adata.var.keys() and y in adata.var.keys():
                        x, y = adata.var[x], adata.var[y]
                    elif x in adata.obs.keys() and y in adata.obs.keys():
                        x, y = adata.obs[x], adata.obs[y]
                    elif np.any(
                        [var_key in x or var_key in y for var_key in adata.var.keys()]
                    ):
                        var_keys = [
                            k
                            for k in adata.var.keys()
                            if not isinstance(adata.var[k][0], str)
                        ]
                        var = adata.var[var_keys]
                        x = var.astype(np.float32).eval(x)
                        y = var.astype(np.float32).eval(y)
                    elif np.any(
                        [obs_key in x or obs_key in y for obs_key in adata.obs.keys()]
                    ):
                        obs_keys = [
                            k
                            for k in adata.obs.keys()
                            if not isinstance(adata.obs[k][0], str)
                        ]
                        obs = adata.obs[obs_keys]
                        x = obs.astype(np.float32).eval(x)
                        y = obs.astype(np.float32).eval(y)
                    else:
                        raise ValueError(
                            "x or y is invalid! pass valid observation or a gene name"
                        )

            x, y = make_dense(x).flatten(), make_dense(y).flatten()

            # convolve along x axes (e.g. pseudotime)
            if n_convolve is not None:
                vec_conv = np.ones(n_convolve) / n_convolve
                y[np.argsort(x)] = np.convolve(y[np.argsort(x)], vec_conv, mode="same")

            # if color is set to a cell index, plot that cell on top
            if is_int(color) or is_list_of_int(color) and len(color) != len(x):
                color = np.array(np.isin(np.arange(len(x)), color), dtype=bool)
                size = kwargs["s"] * 2 if np.sum(color) == 1 else kwargs["s"]
                if zorder is None:
                    zorder = 10
                ax.scatter(
                    np.ravel(x[color]),
                    np.ravel(y[color]),
                    s=size,
                    zorder=zorder,
                    color=palette[-1] if palette is not None else "darkblue",
                )
                color = (
                    palette[0] if palette is not None and len(palette) > 1 else "gold"
                )
                zorder -= 1

            # if color is in {'ascending', 'descending'}
            elif isinstance(color, str):
                if color == "ascending":
                    color = np.linspace(0, 1, len(x))
                elif color == "descending":
                    color = np.linspace(1, 0, len(x))

            # set palette if categorical color vals
            if is_categorical(adata, color):
                set_colors_for_categorical_obs(adata, color, palette)

            # set color
            if (
                basis in adata.var_names
                and isinstance(color, str)
                and color in adata.layers.keys()
            ):
                # phase portrait: color=basis, layer=color
                c = interpret_colorkey(adata, basis, color, perc, use_raw)
            else:
                # embedding, gene trend etc.
                c = interpret_colorkey(adata, color, layer, perc, use_raw)

            if c is not None and not isinstance(c, str) and not isinstance(c[0], str):
                # smooth color values across neighbors and rescale
                if smooth and len(c) == adata.n_obs:
                    n_neighbors = None if isinstance(smooth, bool) else smooth
                    c = get_connectivities(adata, n_neighbors=n_neighbors).dot(c)
                # rescale color values to min and max acc. to rescale_color tuple
                if rescale_color is not None:
                    try:
                        c += rescale_color[0] - np.nanmin(c)
                        c *= rescale_color[1] / np.nanmax(c)
                    except:
                        logg.warn("Could not rescale colors. Pass a tuple, e.g. [0,1].")

            # set vmid to 0 if color values obtained from velocity expression
            if not np.any([v in kwargs for v in ["vmin", "vmid", "vmax"]]) and np.any(
                [
                    isinstance(v, str)
                    and "time" not in v
                    and (v.endswith("velocity") or v.endswith("transition"))
                    for v in [color, layer]
                ]
            ):
                kwargs["vmid"] = 0

            # introduce vmid by setting vmin and vmax accordingly
            if "vmid" in kwargs:
                vmid = kwargs.pop("vmid")
                if vmid is not None:
                    if not (isinstance(c, str) or isinstance(c[0], str)):
                        lb, ub = np.min(c), np.max(c)
                        crange = max(np.abs(vmid - lb), np.abs(ub - vmid))
                        kwargs.update({"vmin": vmid - crange, "vmax": vmid + crange})

            x, y = np.ravel(x), np.ravel(y)
            if len(x) != len(y):
                raise ValueError("x or y do not share the same dimension.")

            if not isinstance(c, str):
                c = np.ravel(c) if len(np.ravel(c)) == len(x) else c
                if len(c) != len(x):
                    c = "grey"
                    if not isinstance(color, str) or color != default_color(adata):
                        logg.warn("Invalid color key. Using grey instead.")

            # store original order of color values
            color_array, scatter_array = c, np.stack([x, y]).T

            # set color to grey for NAN values and for cells that are not in groups
            if (
                groups is not None
                or is_categorical(adata, color)
                and np.any(pd.isnull(adata.obs[color]))
            ):
                if isinstance(groups, (list, tuple, np.record)):
                    groups = unique(groups)
                zorder = 0 if zorder is None else zorder
                pop_keys = ["groups", "add_linfit", "add_polyfit", "add_density"]
                _ = [scatter_kwargs.pop(key, None) for key in pop_keys]
                ax = scatter(
                    adata,
                    x=x,
                    y=y,
                    basis=basis,
                    layer=layer,
                    color="lightgrey",
                    ax=ax,
                    **scatter_kwargs,
                )
                if groups is not None and len(groups) == 1:
                    if (
                        isinstance(groups[0], str)
                        and groups[0] in adata.var.keys()
                        and basis in adata.var_names
                    ):
                        groups = f"{adata[:, basis].var[groups[0]][0]}"
                idx = groups_to_bool(adata, groups, color)
                if idx is not None:
                    if np.sum(idx) > 0:  # if any group to be highlighted
                        x, y = x[idx], y[idx]
                        if not isinstance(c, str) and len(c) == adata.n_obs:
                            c = c[idx]
                        if isinstance(kwargs["s"], np.ndarray):
                            kwargs["s"] = np.array(kwargs["s"])[idx]
                        if (
                            title is None
                            and groups is not None
                            and len(groups) == 1
                            and isinstance(groups[0], str)
                        ):
                            title = groups[0]
                    else:  # if nothing to be highlighted
                        add_linfit, add_polyfit, add_density = None, None, None

            # check if higher value points should be plotted on top
            if not isinstance(c, str) and len(c) == len(x):
                order = None
                if sort_order and not is_categorical(adata, color):
                    order = np.argsort(c)
                elif not sort_order and is_categorical(adata, color):
                    counts = get_value_counts(adata, color)
                    np.random.seed(0)
                    nums, p = np.arange(0, len(x)), counts / np.sum(counts)
                    order = np.random.choice(nums, len(x), replace=False, p=p)
                if order is not None:
                    x, y, c = x[order], y[order], c[order]
                    if isinstance(kwargs["s"], np.ndarray):  # sort sizes if array-type
                        kwargs["s"] = np.array(kwargs["s"])[order]

            smp = ax.scatter(
                x, y, c=c, alpha=alpha, marker=".", zorder=zorder, **kwargs
            )

            outline_dtypes = (list, tuple, np.ndarray, int, np.int_, str)
            if isinstance(add_outline, outline_dtypes) or add_outline:
                if isinstance(add_outline, (list, tuple, np.record)):
                    add_outline = unique(add_outline)
                if (
                    add_outline is not True
                    and isinstance(add_outline, (int, np.int_))
                    or is_list_of_int(add_outline)
                    and len(add_outline) != len(x)
                ):
                    add_outline = np.isin(np.arange(len(x)), add_outline)
                    add_outline = np.array(add_outline, dtype=bool)
                    if outline_width is None:
                        outline_width = (0.6, 0.3)
                if isinstance(add_outline, str):
                    if add_outline in adata.var.keys() and basis in adata.var_names:
                        add_outline = f"{adata[:, basis].var[add_outline][0]}"
                idx = groups_to_bool(adata, add_outline, color)
                if idx is not None and np.sum(idx) > 0:  # if anything to be outlined
                    zorder = 2 if zorder is None else zorder + 2
                    if kwargs["s"] is not None:
                        kwargs["s"] *= 1.2
                    # restore order of values
                    x, y = scatter_array[:, 0][idx], scatter_array[:, 1][idx]
                    c = color_array
                    if not isinstance(c, str) and len(c) == adata.n_obs:
                        c = c[idx]
                    if isinstance(kwargs["s"], np.ndarray):
                        kwargs["s"] = np.array(kwargs["s"])[idx]
                    if isinstance(c, np.ndarray) and not isinstance(c[0], str):
                        if "vmid" not in kwargs and "vmin" not in kwargs:
                            kwargs["vmin"] = np.min(color_array)
                        if "vmid" not in kwargs and "vmax" not in kwargs:
                            kwargs["vmax"] = np.max(color_array)
                    ax.scatter(
                        x, y, c=c, alpha=alpha, marker=".", zorder=zorder, **kwargs
                    )
                if idx is None or np.sum(idx) > 0:  # if all or anything to be outlined
                    plot_outline(
                        x, y, kwargs, outline_width, outline_color, zorder, ax=ax
                    )
                if idx is not None and np.sum(idx) == 0:  # if nothing to be outlined
                    add_linfit, add_polyfit, add_density = None, None, None

            # set legend if categorical categorical color vals
            if is_categorical(adata, color) and len(scatter_array) == adata.n_obs:
                legend_loc = default_legend_loc(adata, color, legend_loc)
                g_bool = groups_to_bool(adata, add_outline, color)
                if not (add_outline is None or g_bool is None):
                    groups = add_outline
                set_legend(
                    adata,
                    ax,
                    color,
                    legend_loc,
                    scatter_array,
                    legend_fontweight,
                    legend_fontsize,
                    legend_fontoutline,
                    groups,
                )
            if add_density:
                plot_density(x, y, add_density, ax=ax)

            if add_linfit:
                if add_linfit is True and basis in adata.var_names:
                    add_linfit = "no_intercept"  # without intercept
                plot_linfit(
                    x,
                    y,
                    add_linfit,
                    legend_loc != "none",
                    linecolor,
                    linewidth,
                    fontsize,
                    ax=ax,
                )

            if add_polyfit:
                if add_polyfit is True and basis in adata.var_names:
                    add_polyfit = "no_intercept"  # without intercept
                plot_polyfit(
                    x,
                    y,
                    add_polyfit,
                    legend_loc != "none",
                    linecolor,
                    linewidth,
                    fontsize,
                    ax=ax,
                )

            if add_rug:
                rug_color = add_rug if isinstance(add_rug, str) else color
                rug_color = np.ravel(interpret_colorkey(adata, rug_color))
                plot_rug(np.ravel(x), color=rug_color, ax=ax)

            if add_text:
                if add_text_pos is None:
                    add_text_pos = [0.05, 0.95]
                ax.text(
                    add_text_pos[0],
                    add_text_pos[1],
                    f"{add_text}",
                    ha="left",
                    va="top",
                    fontsize=fontsize,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
                )

            set_label(xlabel, ylabel, fontsize, basis, ax=ax)
            set_title(title, layer, color, fontsize, ax=ax)
            update_axes(ax, xlim, ylim, fontsize, is_embedding, frameon, figsize)

            if colorbar is not False:
                if not isinstance(c, str) and not is_categorical(adata, color):
                    labelsize = fontsize * 0.75 if fontsize is not None else None
                    set_colorbar(smp, ax=ax, labelsize=labelsize)

            savefig_or_show(dpi=dpi, save=save, show=show)
            if show is False:
                return ax

def heatmap(
    adata,
    var_names,
    sortby="latent_time",
    layer="Ms",
    color_map="viridis",
    col_color=None,
    palette="viridis",
    n_convolve=30,
    standard_scale=0,
    sort=True,
    colorbar=None,
    col_cluster=False,
    row_cluster=False,
    figsize=(8, 4),
    font_scale=None,
    show=None,
    save=None,
    cbar_pos=(1.01, 0.035, 0.015, 0.3),
    **kwargs,
):
    """\
    Plot time series for genes as heatmap.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    var_names: `str`,  list of `str`
        Names of variables to use for the plot.
    sortby: `str` (default: `'latent_time'`)
        Observation key to extract time data from.
    layer: `str` (default: `'Ms'`)
        Layer key to extract count data from.
    color_map: `str` (default: `'viridis'`)
        String denoting matplotlib color map.
    col_color: `str` or list of `str` (default: `None`)
        String denoting matplotlib color map to use along the columns.
    n_convolve: `int` or `None` (default: `30`)
        If `int` is given, data is smoothed by convolution
        along the x-axis with kernel size n_convolve.
    standard_scale : `int` or `None` (default: `0`)
        Either 0 (rows) or 1 (columns). Whether or not to standardize that dimension
        (each row or column), subtract minimum and divide each by its maximum.
    sort: `bool` (default: `True`)
        Wether to sort the expression values given by xkey.
    colorbar: `bool` or `None` (default: `None`)
        Whether to show colorbar.
    {row,col}_cluster : bool, optional
        If True, cluster the {rows, columns}.
    figsize: tuple (default: `(7,5)`)
        Figure size.
    show: `bool`, optional (default: `None`)
        Show the plot, do not return axis.
    save: `bool` or `str`, optional (default: `None`)
        If `True` or a `str`, save the figure. A string is appended to the default
        filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}.
    kwargs:
        Arguments passed to seaborns clustermap,
        e.g., set `yticklabels=True` to display all gene names in all rows.

    Returns
    -------
        If `show==False` a `matplotlib.Axis`
    """

    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as pl
    from matplotlib import rcParams
    from matplotlib.colors import ColorConverter
    import pandas as pd
    from pandas import unique, isnull
    from scipy.sparse import issparse

    from scvelo import logging as logg
    from scvelo.plotting.utils import is_categorical, interpret_colorkey, savefig_or_show, to_list
    from scvelo.plotting.utils import set_colors_for_categorical_obs, strings_to_categoricals

    var_names = [name for name in var_names if name in adata.var_names]

    tkey, xkey = kwargs.pop("tkey", sortby), kwargs.pop("xkey", layer)
    time = adata.obs[tkey].values
    time = time[np.isfinite(time)]

    X = (
        adata[:, var_names].layers[xkey]
        if xkey in adata.layers.keys()
        else adata[:, var_names].X
    )
    if issparse(X):
        X = X.A
    df = pd.DataFrame(X[np.argsort(time)], columns=var_names)

    if n_convolve is not None:
        weights = np.ones(n_convolve) / n_convolve
        for gene in var_names:
            try:
                df[gene] = np.convolve(df[gene].values, weights, mode="same")
            except:
                pass  # e.g. all-zero counts or nans cannot be convolved

    if sort:
        max_sort = np.argsort(np.argmax(df.values, axis=0))
        df = pd.DataFrame(df.values[:, max_sort], columns=df.columns[max_sort])
    strings_to_categoricals(adata)

    if col_color is not None:
        col_colors = to_list(col_color)
        columns = col_colors
        col_color = []
        for _, col in enumerate(col_colors):
            if not is_categorical(adata, col):
                obs_col = adata.obs[col]
                cat_col = np.round(obs_col / np.max(obs_col), 2) * np.max(obs_col)
                adata.obs[f"{col}_categorical"] = pd.Categorical(cat_col)
                col += "_categorical"
                set_colors_for_categorical_obs(adata, col, palette)
            
            col_color.append(interpret_colorkey(adata, col)[np.argsort(time)])
        col_color = pd.DataFrame(np.c_[col_color].T, columns=columns)
        
    if font_scale is not None:
        sns.set(font_scale=font_scale)
    if "dendrogram_ratio" not in kwargs:
        kwargs["dendrogram_ratio"] = (
            0.1 if row_cluster else 0,
            0.2 if col_cluster else 0,
        )
    #if "cbar_pos" is None or not colorbar:
    kwargs["cbar_pos"] = cbar_pos
        
    kwargs.update(
        dict(
            col_colors=col_color,
            col_cluster=col_cluster,
            row_cluster=row_cluster,
            cmap=color_map,
            xticklabels=False,
            standard_scale=standard_scale,
            figsize=figsize,
        )
    )
    #try:
    #    cm = sns.clustermap(df.T, **kwargs)
    #except:
    #    logg.warn("Please upgrade seaborn with `pip install -U seaborn`.")
    kwargs.pop("dendrogram_ratio")
    cm = sns.clustermap(df.T, **kwargs)

    savefig_or_show("heatmap", save=save, show=show)
    if show is False:
        return cm

@doc_params(scatter=doc_scatter)
def velocity_embedding(
    adata,
    basis=None,
    vkey="velocity",
    density=None,
    arrow_size=None,
    arrow_length=None,
    scale=None,
    X=None,
    V=None,
    recompute=None,
    color=None,
    use_raw=None,
    layer=None,
    color_map=None,
    colorbar=True,
    palette=None,
    size=None,
    alpha=0.2,
    perc=None,
    sort_order=True,
    groups=None,
    components=None,
    projection="2d",
    legend_loc="none",
    legend_fontsize=None,
    legend_fontweight=None,
    quiver_lw=0.1,
    xlabel=None,
    ylabel=None,
    title=None,
    fontsize=None,
    figsize=None,
    dpi=None,
    frameon=None,
    show=None,
    save=None,
    ax=None,
    ncols=None,
    **kwargs,
):
    """\
    Scatter plot of velocities on the embedding.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    density: `float` (default: 1)
        Amount of velocities to show - 0 none to 1 all
    arrow_size: `float` or triple `headlength, headwidth, headaxislength` (default: 1)
        Size of arrows.
    arrow_length: `float` (default: 1)
        Length of arrows.
    scale: `float` (default: 1)
        Length of velocities in the embedding.
    {scatter}

    Returns
    -------
        `matplotlib.Axis` if `show==False`
    """
    
    from scvelo.tools.velocity_embedding import velocity_embedding as compute_velocity_embedding
    from scvelo.tools.utils import groups_to_bool
    from scvelo.plotting.scatter import scatter
    from matplotlib import rcParams
    from matplotlib.colors import is_color_like
    import matplotlib.pyplot as pl
    import numpy as np

    if vkey == "all":
        lkeys = list(adata.layers.keys())
        vkey = [key for key in lkeys if "velocity" in key and "_u" not in key]
    color, color_map = kwargs.pop("c", color), kwargs.pop("cmap", color_map)
    layers, vkeys = make_unique_list(layer), make_unique_list(vkey)
    colors = make_unique_list(color, allow_array=True)
    bases = make_unique_valid_list(adata, basis)
    bases = [default_basis(adata, **kwargs) if b is None else b for b in bases]

    if V is None:
        for key in vkeys:
            for bas in bases:
                if recompute or velocity_embedding_changed(adata, basis=bas, vkey=key):
                    compute_velocity_embedding(adata, basis=bas, vkey=key)

    scatter_kwargs = {
        "perc": perc,
        "use_raw": use_raw,
        "sort_order": sort_order,
        "alpha": alpha,
        "components": components,
        "projection": projection,
        "legend_loc": legend_loc,
        "groups": groups,
        "legend_fontsize": legend_fontsize,
        "legend_fontweight": legend_fontweight,
        "palette": palette,
        "color_map": color_map,
        "frameon": frameon,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "colorbar": colorbar,
        "dpi": dpi,
        "fontsize": fontsize,
        "show": False,
        "save": False,
    }

    multikey = (
        colors
        if len(colors) > 1
        else layers
        if len(layers) > 1
        else vkeys
        if len(vkeys) > 1
        else bases
        if len(bases) > 1
        else None
    )
    if multikey is not None:
        if title is None:
            title = list(multikey)
        elif isinstance(title, (list, tuple)):
            title *= int(np.ceil(len(multikey) / len(title)))
        ncols = len(multikey) if ncols is None else min(len(multikey), ncols)
        nrows = int(np.ceil(len(multikey) / ncols))
        figsize = rcParams["figure.figsize"] if figsize is None else figsize
        figsize, dpi = get_figure_params(figsize, dpi, ncols)
        gs_figsize = (figsize[0] * ncols, figsize[1] * nrows)
        ax = []
        for i, gs in enumerate(
            pl.GridSpec(nrows, ncols, pl.figure(None, gs_figsize, dpi=dpi))
        ):
            if i < len(multikey):
                ax.append(
                    velocity_embedding(
                        adata,
                        density=density,
                        scale=scale,
                        size=size,
                        ax=pl.subplot(gs),
                        arrow_size=arrow_size,
                        arrow_length=arrow_length,
                        basis=bases[i] if len(bases) > 1 else basis,
                        color=colors[i] if len(colors) > 1 else color,
                        layer=layers[i] if len(layers) > 1 else layer,
                        vkey=vkeys[i] if len(vkeys) > 1 else vkey,
                        title=title[i] if isinstance(title, (list, tuple)) else title,
                        **scatter_kwargs,
                        **kwargs,
                    )
                )
        savefig_or_show(dpi=dpi, save=save, show=show)
        if show is False:
            return ax

    else:
        if projection == "3d":
            from mpl_toolkits.mplot3d import Axes3D
        ax, show = get_ax(ax, show, figsize, dpi, projection)

        color, layer, vkey, basis = colors[0], layers[0], vkeys[0], bases[0]
        color = default_color(adata) if color is None else color
        color_map = default_color_map(adata, color) if color_map is None else color_map
        size = default_size(adata) / 2 if size is None else size
        if use_raw is None and "Ms" not in adata.layers.keys():
            use_raw = True
        _adata = (
            adata[groups_to_bool(adata, groups, groupby=color)]
            if groups is not None and color in adata.obs.keys()
            else adata
        )

        quiver_kwargs = {
            "scale": scale,
            "cmap": color_map,
            "angles": "xy",
            "scale_units": "xy",
            "edgecolors": "k",
            "linewidth": quiver_lw,
            "width": None,
        }
        if basis in adata.var_names:
            if use_raw:
                x = adata[:, basis].layers["spliced"]
                y = adata[:, basis].layers["unspliced"]
            else:
                x = adata[:, basis].layers["Ms"]
                y = adata[:, basis].layers["Mu"]
            dx = adata[:, basis].layers[vkey]
            dy = np.zeros(adata.n_obs)
            if f"{vkey}_u" in adata.layers.keys():
                dy = adata[:, basis].layers[f"{vkey}_u"]
            X = np.stack([np.ravel(x), np.ravel(y)]).T
            V = np.stack([np.ravel(dx), np.ravel(dy)]).T
        else:
            x = None if X is None else X[:, 0]
            y = None if X is None else X[:, 1]
            comps = get_components(components, basis, projection)
            X = _adata.obsm[f"X_{basis}"][:, comps] if X is None else X[:, :2]
            V = _adata.obsm[f"{vkey}_{basis}"][:, comps] if V is None else V[:, :2]

            hl, hw, hal = default_arrow(arrow_size)
            if arrow_length is not None:
                scale = 1 / arrow_length
            if scale is None:
                scale = 1
            quiver_kwargs.update({"scale": scale, "width": 0.0005, "headlength": hl})
            quiver_kwargs.update({"headwidth": hw, "headaxislength": hal})

        for arg in list(kwargs):
            if arg in quiver_kwargs:
                quiver_kwargs.update({arg: kwargs[arg]})
            else:
                scatter_kwargs.update({arg: kwargs[arg]})

        if (
            basis in adata.var_names
            and isinstance(color, str)
            and color in adata.layers.keys()
        ):
            c = interpret_colorkey(_adata, basis, color, perc)
        else:
            c = interpret_colorkey(_adata, color, layer, perc)

        if density is not None and 0 < density < 1:
            s = int(density * _adata.n_obs)
            ix_choice = np.random.choice(_adata.n_obs, size=s, replace=False)
            c = c[ix_choice] if len(c) == _adata.n_obs else c
            X = X[ix_choice]
            V = V[ix_choice]

        if projection == "3d" and X.shape[1] > 2 and V.shape[1] > 2:
            V, size = V / scale / 5, size / 10
            x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
            v0, v1, v2 = V[:, 0], V[:, 1], V[:, 2]
            quiver3d_kwargs = {"zorder": 3, "linewidth": 0.5, "arrow_length_ratio": 0.3}
            c = list(c) + [element for element in list(c) for _ in range(2)]
            if is_color_like(c[0]):
                ax.quiver(x0, x1, x2, v0, v1, v2, color=c, **quiver3d_kwargs)
            else:
                ax.quiver(x0, x1, x2, v0, v1, v2, c, **quiver3d_kwargs)
        else:
            quiver_kwargs.update({"zorder": 3})
            if is_color_like(c[0]):
                ax.quiver(X[:, 0], X[:, 1], V[:, 0], V[:, 1], color=c, **quiver_kwargs)
            else:
                ax.quiver(X[:, 0], X[:, 1], V[:, 0], V[:, 1], c, **quiver_kwargs)

        scatter_kwargs.update({"basis": basis, "x": x, "y": y, "color": color})
        scatter_kwargs.update({"vkey": vkey, "layer": layer})
        ax = scatter(adata, size=size, title=title, ax=ax, zorder=0, **scatter_kwargs)

        savefig_or_show(dpi=dpi, save=save, show=show)
        if show is False:
            return ax
