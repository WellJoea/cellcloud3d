from typing import (
    Optional,
    Generator,
    FrozenSet,
    Sequence,
    List,
    Tuple,
    Dict,
    Any,
    Mapping,
    Literal
)
_Format = Literal[
    'png', 'jpg', 'tif', 'tiff',
    'pdf', 'ps', 'eps', 'svg', 'svgz', 'pgf',
    'raw', 'rgba',
]

def set_figure_params(
    scanpy: bool = True,
    dpi: int = 80,
    dpi_save: int = 150,
    frameon: bool = True,
    vector_friendly: bool = True,
    fontsize: int = 14,
    figsize: Optional[int] = None,
    color_map: Optional[str] = None,
    format: _Format = "pdf",
    facecolor: Optional[str] = None,
    transparent: bool = False,
    ipython_format: str = "png2x",
):
    """
    Coding from scanpy
    """
    from matplotlib import rcParams

    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if facecolor is not None:
        rcParams['figure.facecolor'] = facecolor
        rcParams['axes.facecolor'] = facecolor
    if figsize is not None:
        rcParams['figure.figsize'] = figsize
    rcParams["figure.frameon"] = frameon

