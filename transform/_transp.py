import numpy as np
def points_pad(locs, tl=[30,30]):
    tp, lf = tl
    ipos = locs.copy()
    ipos[:,0] += lf
    ipos[:,1] += tp
    return ipos

def rescale_point2d(locs, scale=None, inverse=False):
    if (scale is None) or (scale==1):
        return locs

    if type(scale) in [int, float]:
        scale = [scale, scale]

    locs = np.asarray(locs).copy()
    new_locs = locs.copy()[:,:2]
    if inverse:
        new_locs = new_locs / np.array(scale)
    else:
        new_locs = new_locs * np.array(scale)
    locs[:,:2] = new_locs[:,:2]
    return locs

def rescale_points(locs, scales, inverse=False):
    assert len(locs) == len(scales), 'the length between locs and scales must be same.'
    new_locs = [ rescale_point2d(locs[i], scales[i], inverse=inverse)
                    for i in range(len(locs))]
    return new_locs

def homotransform_point(locs, tmat, inverse=False):
    if locs is None:
        return locs
    locs = np.asarray(locs).copy()
    new_locs = locs.copy()[:,:2]
    new_locs = np.c_[new_locs, np.ones(new_locs.shape[0])]

    if inverse:
        new_locs =  new_locs @ tmat.T
    else:
        new_locs =  new_locs @ np.linalg.inv(tmat).T

    # locs[:,:2] = new_locs[:,:2]
    locs[:,:2] = new_locs[:,:2]/new_locs[:,[2]]
    return locs

def homotransform_points(locs, tmats, inverse=False):
    assert len(locs) == len(tmats), 'the length between locs and tmats must be same.'
    new_locs = [ homotransform_point(locs[i], tmats[i], inverse=inverse)
                    for i in range(len(locs))]
    return new_locs