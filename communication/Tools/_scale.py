from sklearn.preprocessing import (label_binarize,  OneHotEncoder, FunctionTransformer,
                                    MinMaxScaler, minmax_scale, MaxAbsScaler,
                                    StandardScaler, RobustScaler, Normalizer,
                                    QuantileTransformer, PowerTransformer, OrdinalEncoder)
import numpy as np
from sklearn.pipeline import Pipeline

def Standard(dfa, 
             scale = 'S',
              Sargs={"copy": True, "with_mean": True, "with_std":True},
              Rargs={'with_centering':True, 
                'with_scaling':True,
                'quantile_range':(25.0, 75.0), 
                'copy':True, 
                'unit_variance':False},
             Margs={'feature_range':(0, 1), 'copy':True, 'clip':False},
             OEargs={ 'categories':'auto', 
                        'handle_unknown':'error',
                        'unknown_value':None,
                        'encoded_missing_value':np.nan},
             OHargs={'categories':'auto',
                         'drop':None, 
                         'sparse':True, 
                         'handle_unknown':'error',
                         'min_frequency':None, 
                         'max_categories':None},
             NLargs={'norm':'l2', 'copy':True},
             QTargs={'n_quantiles':1000, 
                        'output_distribution':'uniform', 
                        'ignore_implicit_zeros':False, 
                        'subsample':100000,
                         'random_state':None, 
                         'copy':True},
             PTargs={'method':'yeo-johnson', 'standardize':True, 'copy':True},
            ):
    Scalers = {
        'S' : StandardScaler(**Sargs),
        'R' : RobustScaler(**Rargs),
        'M' : MinMaxScaler(**Margs),
        'MA': MaxAbsScaler(),
        'OE': OrdinalEncoder(**OEargs),
        'OH': OneHotEncoder(**OHargs),
        'NL' : Normalizer(**NLargs),
        'QT': QuantileTransformer(**QTargs),
        'PT': PowerTransformer(**PTargs),
        'N' : FunctionTransformer( validate=False ),
    }
    mapper = Pipeline([ (i, Scalers[i]) for i in scale])
    dfaft  = mapper.fit_transform( dfa )

    return dfaft