import pandas as pd
import numpy as np
import scanpy as sc
import PIL
import io
import base64

def colrows(ncell, nrows=None, ncols=None, soft=True):
    import math
    if (ncols is None) and (nrows is None):
        ncols = int(np.ceil(ncell**0.5))
    if not ncols is None:
        nrows = math.ceil(ncell/ncols)
        ncols = min(ncell, ncols)
    elif not nrows is None:
        ncols = math.ceil(ncell/nrows)
        nrows = min(ncell, nrows)
    if soft and ncell> 1 and (ncell - ncols*(nrows-1)<=1):
        ncols += 1
        nrows -= 1
    return (nrows, ncols)

def pdtype(pdseries):
    if pdseries.dtype in ['category', 'object', 'bool']:
        return 'discrete'
    elif pdseries.dtype in ['float32', 'float64', 'float', 'int32', 'int64', 'int']:
        return 'continuous'
    else:
        return 'continuous'

def image2batyes(image, scale=True):
    if scale and (image.max()>255):
        amin = image.min()
        amax = image.max()
        image = np.clip(255.0 * (image-amin)/(amax-amin), 0, 255).astype(np.uint8)
    img_obj = PIL.Image.fromarray(image)
    prefix = "data:image/png;base64,"
    with io.BytesIO() as stream:
        img_obj.save(stream, format='png')
        b64_str = prefix + base64.b64encode(stream.getvalue()).decode('unicode_escape')
    return b64_str