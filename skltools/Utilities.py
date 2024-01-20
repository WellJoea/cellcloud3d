import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import (StratifiedShuffleSplit, LeavePOut, 
                                     LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold)
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.validation import column_or_1d
import warnings

def _default_kargs(_func):
    import inspect
    signature = inspect.signature(_func)
    return { k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty}

def _predict_proba_lr(prob):
    """
    Probability estimation for OvR logistic regression.
    Positive class probabilities are computed as
    1. / (1. + np.exp(-self.decision_function(X)));
    multiclass is handled by normalizing that over all classes.
    """
    prob = expit(prob)
    if prob.ndim == 1:
        return np.vstack([1 - prob, prob]).T
    else:
        # OvR normalization, like LibLinear's predict_probability
        prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
    return prob

def to_n_array(Ys):
    '''
    change the type to np.ndarray.
    '''
    if isinstance(Ys, list):
        return np.array(Ys)
    elif isinstance(Ys, np.ndarray):
        if len(Ys.shape)>=2 and min(Ys.shape)==1:
            return np.concatenate(Ys)
        else:
            return Ys
    elif isinstance(Ys, pd.Series):
        return Ys.to_numpy()
    elif isinstance(Ys, pd.DataFrame):
        if len(Ys.shape)>=2 and min(Ys.shape)==1:
            return np.concatenate(Ys.to_numpy())
        else:
            return Ys.to_numpy()
    else:
        raise ValueError('Invalid type: must be list, np.ndarray, pd.Series or pd.DataFrame.')

def get_n_classes(Ys):
    #sklearn.utils.multiclass.type_of_target
    Yc = to_n_array(Ys)
    if len(Yc.shape)<2:
        return len(set(Yc))
    else:
        return Yc.shape[1]

def Check_proba(prob):
    if isinstance(prob, list) or isinstance(prob, tuple):
        prob = np.array(prob)
    if prob.ndim==1:
        prob = np.array([1-prob, prob]).T
        #if isinstance(prob, pd.Series):
    if prob.ndim==2:
        if prob.shape[1] ==1:
            prob = np.c_[1-prob, prob]
        return np.array(prob)
    else:
        raise ValueError('The data type is unkown.')

def Check_Binar(_y):
    y = column_or_1d(_y)
    y = LabelBinarizer().fit_transform(y)
    if y.shape[1] ==1:
        y = np.c_[1-y,y]
    return y

def Check_Label(_y):
    y = column_or_1d(_y)
    if y.min() >0:
        warnings.warn("The taget babel should start with 0,, place check you data format.")
        return LabelEncoder().fit_transform(y)
    else:
        return _y

def CrossvalidationSplit(cvm='SSA', n_splits=10, n_repeats=3, leavP=1, test_size=0.3, random_state = None ):
    n_splits, n_repeats, leavP = int(n_splits), int(n_repeats), int(leavP)
    CV = { 'SSS' : StratifiedShuffleSplit(
                        n_splits = n_splits,
                        test_size= test_size,
                        random_state=random_state),
            'SSA': StratifiedShuffleSplit(
                        n_splits = n_splits,
                        test_size= test_size,
                        random_state=random_state),
            'SFA': StratifiedKFold(
                        n_splits=int(round(1/test_size)),
                        random_state=random_state),
            'SFK': StratifiedKFold(
                        n_splits = n_splits,
                        random_state=random_state),
            'RKF': RepeatedKFold(
                        n_splits = n_splits,
                        n_repeats= n_repeats,
                        random_state=random_state),
            'RSKF': RepeatedStratifiedKFold(
                        n_splits = n_splits,
                        n_repeats= n_repeats,
                        random_state=random_state),
            'LOU': LeaveOneOut(),
            'LPO': LeavePOut(leavP),
    }
    return CV[cvm]

def cvsplit(X, y, cvm=['SSA','SFA'], cvt=7, test=0.3, random_state = None ):
    for _m in cvm:
        random_state = np.random.randint(1,1000000, 1)[0]
        _cvs = CrossvalidationSplit(n_splits=cvt, test_size=test, cvm=_m)
        for train_index, test_index in _cvs.split(X, y):
            yield (train_index, test_index)