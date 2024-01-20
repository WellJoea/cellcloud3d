import joblib
import Modeldata 

def read_pkl(file):
    return joblib.load(file)

def save_pkl(mdata, file):
    joblib.dump(mdata, file)

def save_mdata(mdata, file):
    mdata_dict = {}
    for i in dir(mdata):
        if not i.startswith('_'):
            mdata_dict[i] = eval(f'mdata.{i}')
        
    save_pkl(mdata_dict, file)

def read_mdata(file):
    mdata_dict = read_pkl(file)
    mdata = Modeldata.mdata()
    for _k in [ 'X', 'y', 'model', 'cmodel', 'args', 'cv', 'features', 'predicts',
                'hypers', 'learns', 'progress', 'samples', 'stackmodel']:
        if _k in mdata_dict.keys():
            _v = mdata_dict[_k]
            exec('mdata.%s = _v'%_k)
    return mdata