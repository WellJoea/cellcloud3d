import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, bsr_matrix,csc_matrix, csgraph
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc,isspmatrix_coo
import matplotlib.pyplot as plt
from sys import getsizeof

class createURD:
    def __init__(self, adata, root_cells=[0], tip_cells=[-1], metadata=None, 
                pseud=None, diff=None, tree={}, stable={}, attr={}):
        self._adata = adata.copy()
        self.root_cells=root_cells.copy()
        self.tip_cells =tip_cells.copy()
        self._cells = adata.obs_names
        
        self._metadata = pd.DataFrame(index=self._cells) if metadata is None else metadata.loc[self._cells,:]
        self._diff = pd.DataFrame(index=self._cells) if diff is None else diff.loc[self._cells,:]
        self._pseud = pd.DataFrame(index=self._cells) if pseud is None else pseud.loc[self._cells,:]
        
        self._tree = tree
        self._stable = stable
        self._attr = attr

    @property
    def adata(self):
        return self._adata
    @adata.setter
    def adata(self, values):
        self._adata = values
    @adata.getter
    def adata(self):
        return self._adata

    @property
    def cells(self):
        self._cells = self.adata.obs_names.copy()
        return self._cells
    @cells.setter
    def cells(self, values):
        self._cells = values

    def _set_column(self, df, values):
        if isinstance(values,  pd.DataFrame):
            df = values.loc[self.cells,:].copy()
            return df
        elif(isinstance(values, list) or isinstance(values, tuple)):
            if len(values)==2:
                df.__setitem__(*values)
                return df
            else:
                raise("the value format is:('colname','value')!")
        else:
            raise("please input the pd.DataFrame or 2 length list/tuple!")

    @property
    def metadata(self):
        return self._metadata
    @metadata.setter
    def metadata(self, values):
        self._metadata = self._set_column(self._metadata, values)
    @metadata.deleter
    def metadata(self):
        self._metadata = None

    @property
    def diff(self):
        return self._diff
    @diff.setter
    def diff(self, values):
        self._diff = self._set_column(self._diff, values)
    @diff.getter
    def diff(self):
        return self._diff
    @diff.deleter
    def diff(self):
        self._diff = None

    @property
    def pseud(self):
        return self._pseud
    @pseud.setter
    def pseud(self, values):
        self._pseud = self._set_column(self._pseud, values)
    @pseud.getter
    def pseud(self):
        return self._pseud
    @pseud.deleter
    def pseud(self):
        self._pseud = None

    @property
    def stable(self):
        return self._stable
    @stable.setter
    def stable(self, values):
        self._stable.update(values)
    @stable.getter
    def stable(self):
        return self._stable
    @stable.deleter
    def stable(self):
        self._stable.clear()

    @property
    def tree(self):
        return self._tree
    @tree.setter
    def tree(self, values):
        self._tree.update(values)
    @tree.getter
    def tree(self):
        return self._tree
    @tree.deleter
    def tree(self):
        self._tree.clear()

    @property
    def attr(self):
        return self._attr
    @attr.setter
    def attr(self, values):
        self._attr.update(values)

    def Time(self):
        import time
        return(time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime()))

    def corstate(self, x,y, methd='pearsonr'):
        from minepy import MINE
        import scipy.stats

        if methd =='pearsonr':
            return(scipy.stats.pearsonr(x, y)[0])
        elif methd =='spearmanr':
            return(scipy.stats.spearmanr(x, y)['correlation'])
        elif methd =='kendalltau':
            return(scipy.stats.kendalltau(x, y)['correlation'])
        elif methd=='mic':
            m = MINE()
            m.compute_score(x, y)
            return(m.mic())

    def sigmoid(self, x, x0, k, c=1):
        return(c/(1 + np.exp(-1 * k * (x - x0))))

    def subcsxm(self, csxm):
        rowidx, colidx = csxm.nonzero()
        cmdata = csxm.data

class sp():
    def __init(self, data, index=None, columns=None):
        self._data = data
        self._shape = self._data.shape
        self._index = index
        self._columns = columns
        self._rcidx  = self._data.nonzero()
        self._data1d = self._data.data

    @property
    def columns(self):
        return self._columns
    @columns.setter
    def columns(self, values):
        self._columns = values
    @columns.deleter
    def columns(self):
        del self._columns

    @property
    def index(self):
        return self._index
    @index.setter
    def index(self, values):
        self._index = values
    @index.deleter
    def index(self):
        del self._index

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, values):
        self._data = values
    @data.deleter
    def data(self):
        del self._data
 
    @property
    def shape(self):
        return self.data.shape

    @property
    def rcidx(self):
        self._rcidx = self.data.nonzero()
        return self._rcidx

    @property
    def data1d(self):
        self._data1d = self.data.data
        return self._data1d

    def checkcsr(self):
        if not issparse(self.data):
            csr_cm = csr_matrix(self.data)
        if isspmatrix_csc(self.data):
            csr_cm = self.data.transpose()
        if isspmatrix_coo(self.data):
            csr_cm = self.data.tocsr()
        if not isspmatrix_csr(self.data):
            print(type(self.data))
            raise('please input scipy.sparse data!')
        else:
            return(csr_cm)
    
    def toarray(self):
        return self.data.toarray()
    def todense(self):
        return self.data.todense()
    def toframe(self, savena=np.nan):
        return pd.DataFrame.sparse.from_spmatrix(self.data, index=self.index, columns=self.columns)