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
        self.adata = adata.copy()
        self.root_cells=root_cells.copy()
        self.tip_cells =tip_cells.copy()
        self.cells = adata.obs_names
        
        self.metadata = pd.DataFrame(index=self.cells) if metadata is None else metadata.loc[self.cells,:]
        self.diff = pd.DataFrame(index=self.cells) if diff is None else diff.loc[self.cells,:]
        self.pseud = pd.DataFrame(index=self.cells) if pseud is None else pseud.loc[self.cells,:]
        
        self.tree = tree
        self.stable = stable
        self.attr = attr

    @property
    def adata(self):
        return self.adata
    @adata.setter
    def adata(self, values):
        self.adata = values

    @property
    def cells(self):
        self.cells = self.adata.obs_names.copy()
        return self.cells
    @cells.setter
    def cells(self, values):
        self.cells = values

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
        return self.metadata
    @metadata.setter
    def metadata(self, values):
        self.metadata = self._set_column(self.metadata, values)
    @metadata.deleter
    def metadata(self):
        del self.metadata

    @property
    def diff(self):
        return self.diff
    @diff.setter
    def diff(self, values):
        self.diff = self._set_column(self.diff, values)
    @diff.deleter
    def diff(self):
        del self.diff

    @property
    def pseud(self):
        return self.pseud
    @pseud.setter
    def pseud(self, values):
        self.pseud = self._set_column(self.pseud, values)
    @pseud.deleter
    def pseud(self):
        del self.pseud

    @property
    def stable(self):
        return self.stable
    @stable.setter
    def stable(self, values):
        self.stable.update(values)
    @stable.deleter
    def stable(self):
        del self.stable

    @property
    def tree(self):
        return self.tree
    @tree.setter
    def tree(self, values):
        self.tree.update(values)
    @tree.deleter
    def tree(self):
        del self.tree

    @property
    def attr(self):
        return self.attr
    @attr.setter
    def attr(self, values):
        self.attr.update(values)

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
        self.data = data
        self.shape = self.data.shape
        self.index = index
        self.columns = columns
        self.rcidx  = self.data.nonzero()
        self.data1d = self.data.data

    @property
    def columns(self):
        return self.columns
    @columns.setter
    def columns(self, values):
        self.columns = values
    @columns.deleter
    def columns(self):
        del self.columns

    @property
    def index(self):
        return self.index
    @index.setter
    def index(self, values):
        self.index = values
    @index.deleter
    def index(self):
        del self.index

    @property
    def data(self):
        return self.data
    @data.setter
    def data(self, values):
        self.data = values
    @data.deleter
    def data(self):
        del self.data
 
    @property
    def shape(self):
        return self.data.shape

    @property
    def rcidx(self):
        self.rcidx = self.data.nonzero()
        return self.rcidx

    @property
    def data1d(self):
        self.data1d = self.data.data
        return self.data1d

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