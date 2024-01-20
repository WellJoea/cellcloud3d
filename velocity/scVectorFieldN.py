import matplotlib
import numpy as np
import pandas as pd

# from scipy.sparse import issparse
from matplotlib import cm
from matplotlib.axes import Axes
from anndata import AnnData
from typing import Union, Optional, List
from matplotlib.figure import Figure
from scipy.sparse import spmatrix
from sklearn.preprocessing import normalize
import scanpy as sc
import plotly.express as px #5.3.1
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import dynamo as dyn
from dynamo.plot.utils import (
    quiver_autoscaler,
    default_quiver_args,
    save_fig,
    set_arrow_alpha,
    set_stream_line_alpha,
    _get_adata_color_vec,
)

from dynamo.dynamo_logger import main_debug, main_info

def _sccolor(adata, groupby):
    return sc.pl._tools.scatterplots._get_palette(adata, groupby)
    
def cell_wise_vectors_3dN(
    adata: AnnData,
    basis: str = "umap",
    x: int = 0,
    y: int = 1,
    z: int = 2,
    ekey: str = None,
    vkey: str = "velocity_S",
    X: Union[np.array, spmatrix] = None,
    V: Union[np.array, spmatrix] = None,
    color: Union[str, List[str]] = None,
    layer: str = "X",
    background: Optional[str] = "white",
    ncols: int = 4,
    figsize: tuple = (6, 4),
    ax: Optional[Axes] = None,
    inverse: True = False,
    cell_inds: str = "all",
    vector: str = "velocity",
    save_show_or_return: str = "show",
    save_kwargs: dict = {},
    quiver_3d_kwargs: dict = {
        "zorder": 3,
        "length": 2,
        "linewidth": 5,
        "arrow_length_ratio": 5,
        "norm": cm.colors.Normalize(),
        "cmap": cm.PRGn,
    },
    grid_color: Optional[str] = None,
    axis_label_prefix: Optional[str] = None,
    axis_labels: Optional[list] = None,
    elev: float = None,
    azim: float = None,
    alpha: Optional[float] = None,
    show_magnitude=False,
    show=False,
    titles: list = None,
    **cell_wise_kwargs,
):
    """Plot the velocity or acceleration vector of each cell.

    Parameters
    ----------
        %(scatters.parameters.no_show_legend|kwargs|save_kwargs)s
        ekey: `str` (default: "M_s")
            The expression key
        vkey: `str` (default: "velocity_S")
            The velocity key
        inverse: `bool` (default: False)
            Whether to inverse the direction of the velocity vectors.
        cell_inds: `str` or `list` (default: all)
            the cell index that will be chosen to draw velocity vectors. Can be a list of integers (cell indices) or str
            (Cell names).
        quiver_size: `float` or None (default: None)
            The size of quiver. If None, we will use set quiver_size to be 1. Note that quiver quiver_size is used to
            calculate the head_width (10 x quiver_size), head_length (12 x quiver_size) and headaxislength (8 x
            quiver_size) of the quiver. This is done via the `default_quiver_args` function which also calculate the
            scale of the quiver (1 / quiver_length).
        quiver_length: `float` or None (default: None)
            The length of quiver. The quiver length which will be used to calculate scale of quiver. Note that befoe
            applying `default_quiver_args` velocity values are first rescaled via the quiver_autoscaler function. Scale
            of quiver indicates the nuumber of data units per arrow length unit, e.g., m/s per plot width; a smaller
            scale parameter makes the arrow longer.
        vector: `str` (default: `velocity`)
            Which vector type will be used for plotting, one of {'velocity', 'acceleration'} or either velocity field or
            acceleration field will be plotted.
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show
            area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips &
            tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq
            paper: https://science.sciencemag.org/content/367/6482/1151.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'cell_wise_velocity', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a
            dictionary that properly modify those keys according to your needs.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        cell_wise_kwargs:
            Additional parameters that will be passed to plt.quiver function
    Returns
    -------
        Nothing but a cell wise quiver plot.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex
    from matplotlib import cm

    def add_axis_label(ax, labels):
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

    projection_dim_indexer = [x, y, z]

    # ensure axis_label prefix is not None
    if ekey is not None and axis_label_prefix is None:
        axis_label_prefix = ekey
    elif axis_label_prefix is None:
        axis_label_prefix = "dim"

    # ensure axis_labels is not None
    if axis_labels is None:
        axis_labels = [axis_label_prefix + "_" + str(index) for index in projection_dim_indexer]

    if type(color) is str:
        color = [color]

    if titles is None:
        titles = color

    assert len(color) == len(titles), "#titles does not match #color."

    if grid_color:
        plt.rcParams["grid.color"] = grid_color

    if alpha:
        quiver_3d_kwargs = dict(quiver_3d_kwargs)
        quiver_3d_kwargs["alpha"] = alpha

    if X is not None and V is not None:
        X = X[:, [x, y, z]]
        V = V[:, [x, y, z]]

    elif type(x) == str and type(y) == str and type(z) == str:
        if len(adata.var_names[adata.var.use_for_dynamics].intersection([x, y, z])) != 3:
            raise ValueError(
                "If you want to plot the vector flow of three genes, please make sure those three genes "
                "belongs to dynamics genes or .var.use_for_dynamics is True."
            )
        X = adata[:, projection_dim_indexer].layers[ekey].A
        V = adata[:, projection_dim_indexer].layers[vkey].A
        layer = ekey
    else:
        if ("X_" + basis in adata.obsm.keys()) and (vector + "_" + basis in adata.obsm.keys()):
            X = adata.obsm["X_" + basis][:, projection_dim_indexer]
            V = adata.obsm[vector + "_" + basis][:, projection_dim_indexer]
        else:
            if "X_" + basis not in adata.obsm.keys():
                layer, basis = basis.split("_")
                reduceDimension(adata, layer=layer, reduction_method=basis)
            if "kmc" not in adata.uns_keys():
                cell_velocities(adata, vkey="velocity_S", basis=basis)
                X = adata.obsm["X_" + basis][:, projection_dim_indexer]
                V = adata.obsm[vector + "_" + basis][:, projection_dim_indexer]
            else:
                kmc = adata.uns["kmc"]
                X = adata.obsm["X_" + basis][:, projection_dim_indexer]
                V = kmc.compute_density_corrected_drift(X, kmc.Idx, normalize_vector=True)
                adata.obsm[vector + "_" + basis] = V

    X, V = X.copy(), V.copy()
    if not show_magnitude:
        X = normalize(X, axis=0, norm="max")
        V = normalize(V, axis=0, norm="l2")
        V = normalize(V, axis=1, norm="l2")

    V /= 3 * quiver_autoscaler(X, V)
    if inverse:
        V = -V

    main_info("X shape: " + str(X.shape) + " V shape: " + str(V.shape))
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "z": X[:, 2], "u": V[:, 0], "v": V[:, 1], "w": V[:, 2]})

    if cell_inds == "all":
        ix_choice = np.arange(adata.shape[0])
    elif cell_inds == "random":
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=1000, replace=False)
    elif type(cell_inds) is int:
        ix_choice = np.random.choice(np.range(adata.shape[0]), size=cell_inds, replace=False)
    elif type(cell_inds) is list:
        if type(cell_inds[0]) is str:
            cell_inds = [adata.obs_names.to_list().index(i) for i in cell_inds]
        ix_choice = cell_inds
    else:
        ix_choice = np.arange(adata.shape[0])

    df = df.iloc[ix_choice, :]

    if background is None:
        _background = rcParams.get("figure.facecolor")
        background = to_hex(_background) if type(_background) is tuple else _background

    # single axis output
    x0, x1, x2 = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    v0, v1, v2 = df.iloc[:, 3], df.iloc[:, 4], df.iloc[:, 5]
    nrows = len(color) // ncols
    if nrows * ncols < len(color):
        nrows += 1
    ncols = min(ncols, len(color))
    print(df)

    for i in color:
        colormap  = _sccolor(adata, i)
        colorlist = adata.obs[i].map(colormap)
        gocones = []
        for n,ig in enumerate(adata.obs[i].cat.categories.tolist()):
            idx = (adata.obs[i]==ig).to_list()
            icone = go.Cone(  x=x0[idx], y=x1[idx], z=x2[idx], 
                              u=v0[idx], v=v1[idx], w=v2[idx],
                              sizemode='scaled',
                              sizeref=6,
                              showlegend=True,
                              showscale=False,
                              name = ig,
                              colorscale=[(0, colormap[ig]), (1,colormap[ig])] )
            gocones.append(icone)

        camera = dict(up=dict(x=0, y=0, z=1),
                      center=dict(x=0, y=0, z=0),
                      eye=dict(x=-.75, y=-1.35, z=0.85))

        layout = go.Layout(scene=dict(xaxis=dict(title='Longitude'),
                                      yaxis=dict(title='Latitude'),
                                      zaxis=dict(title='Elevation'),
                                      camera=camera))

        fig1 = go.Figure(data=gocones)
        fig1.update_layout(legend_font_size=14,
                           width =1300,
                           height=1000,
                           legend=dict(itemsizing = 'constant'),
                           template='none',
                        scene=dict(
                            xaxis=dict(visible=True, showticklabels=True),
                            yaxis=dict(visible=True, showticklabels=True),
                            zaxis=dict(visible=True, showticklabels=True),
                        ),
                      plot_bgcolor='#FFFFFF',) #
                      #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                      #paper_bgcolor='#000000',
                      #plot_bgcolor='#000000'
                      #fig.update_xaxes(visible=False, showticklabels=False)

        if show:
            fig1.show()
        fig1.write_html(f'dynamo_{i}.cones.3d.html')
        return fig1