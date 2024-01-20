import numpy as np
from scipy.sparse import issparse, csr_array
from sklearn.utils.extmath import randomized_svd
from typing import List, Optional

from cellcloud3d.preprocessing._normalize import scale_array

def dual_pca(X:np.ndarray, Y:np.ndarray, 
            n_comps:Optional[int]=50,
            scale:Optional[bool]=True,
            copy:Optional[bool]=True,
            seed:Optional[int]=200504,
            axis:Optional[int]=0,
            zero_center:Optional[bool]=True,
            **kargs
    ) -> List:
    assert X.shape[1] == Y.shape[1]
    if scale:
        X = scale_array(X, axis=axis, copy=copy, zero_center=zero_center, **kargs)
        Y = scale_array(Y, axis=axis, copy=copy, zero_center=zero_center, **kargs)
    cor_var = X @ Y.T
    U, S, V = randomized_svd(cor_var, n_components=n_comps, random_state=seed)
    S = np.diag(S)
    Xh = U @ np.sqrt(S)
    Yh = V.T @ np.sqrt(S)
    return Xh, Yh