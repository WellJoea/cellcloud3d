#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : Supervising.py
* @Author  : Zhou Wei                                     *
* @Date    : 2019/10/25 19:28:13                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Zhou Wei.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

# Please start your performance

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             classification_report, make_scorer, balanced_accuracy_score,
                             precision_recall_curve, mean_squared_error, roc_auc_score, 
                             roc_curve, auc, r2_score, mean_absolute_error,
                             average_precision_score, explained_variance_score, 
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import  OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.calibration import CalibratedClassifierCV

from scipy.stats import pearsonr, stats
from scipy.sparse import hstack, vstack
import joblib
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
import re
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100000)

import Utilities as ul
from Estimators import Estimators
from Config import Config as cf
import collections
import copy

from tqdm import tqdm
from termcolor import colored

def splitcv(mdata, Repeatime=1, cvm=['SSA','SFA'], cvt=10):
    cf.log.CI(colored("Split data into training and test data.", 'blue'))
    Xa, ya = mdata.X.copy(), mdata.y.copy()
    cvs =  sum([list(ul.cvsplit(Xa, ya, cvt =cvt,cvm=cvm)) for _k in range(Repeatime)], [])
    mdata.cv = cvs
    return mdata

def hypercv(mdata, searchcv='HGSCV', searchlev='level1', test_size=0.3, n_jobs=-1, score=None, n_iter=2500):
    cf.log.CI(colored("Get estimator and initial parameters.", 'blue'))
    model = mdata.cmodel
    estimat_para = Hypermodel(mdata, searchcv=searchcv, searchlev=searchlev, test_size=test_size)
    estimat_para['cvidx'] = list(range(len(mdata.cv)))
    mdata.hypers[model].update(estimat_para)
    
    cf.log.CI(colored("Hyper-parameter optimize.", 'blue'))
    for _idx in tqdm(estimat_para['cvidx'], colour='green', bar_format='{l_bar}{bar}{r_bar}\n'):
        hyper_para = Hyperparemet(mdata, _idx, n_jobs=n_jobs, score=score, n_iter=n_iter)
        mdata.hypers[model].update({ _idx: {**estimat_para, **hyper_para} })
        #cf.log.CI( ('Modeling has been Completed %2d%%'%((_idx+1)*100/len(mdata.cv))).center(45, '-') )
    return mdata

def learncv(mdata, clfs=['clf', 'clf_c']):
    cf.log.CI(colored("Fit the models and get related attributes.", 'blue'))
    n_classes = mdata.attr['n_classes']
    model = mdata.cmodel
    for _idx in tqdm(mdata.hypers[model]['cvidx'], colour='green', bar_format='{l_bar}{bar}{r_bar}\n'):
        _clfs_attr = {}
        for clf_base in clfs:
            _clf   = mdata.hypers[model][_idx][clf_base]
            _import = Featurescoef(_clf, model, n_classes=n_classes)
            _pred, _prob = Predict_prob(_clf, mdata.X, n_classes=n_classes)
            _clfs_attr[clf_base] = {'importance':_import , 'predict':_pred, 'predict_proba':_prob }
        mdata.learns[model].update({_idx : _clfs_attr})   
        mdata = evaluat_scoring(mdata, _idx, clf_bases=clfs)
    return mdata

def predictcv(mdata, pdata, samples = None, labels= None, clfs=['clf', 'clf_c'], model=None):
    cf.log.CI(colored('predict results from new data.', 'blue'))
    n_classes = mdata.attr['n_classes']
    model =  mdata.cmodel if model is None else model
    
    samples = np.arange(pdata.shape[0]) if samples is None else samples
    if not labels is None:
        y_labels  = labels
        y_blabels = mdata.ym.transform(y_labels)
    else:
        y_labels = None
        y_blabels = None
    mdata.predicts[model].update({'samples':samples, 'y':y_labels, 'yb':y_blabels})   
    
    for _idx in tqdm( mdata.hypers[model]['cvidx'], colour='green', bar_format='{l_bar}{bar}{r_bar}\n'):
        _clfs_attr = {}
        for clf_base in clfs:
            _clf   = mdata.hypers[model][_idx][clf_base]
            _pred, _prob = Predict_prob(_clf, pdata, n_classes=n_classes)
            _clfs_attr[clf_base] = {'predict':_pred, 'predict_proba':_prob }
        mdata.predicts[model].update({_idx : _clfs_attr})   
        mdata = evaluat_scoring(mdata, _idx,  clf_bases=clfs, target='predicts')
    return mdata

def integratecv(mdata, target='learns', select_cv=None, samples= None, model=None):
    cf.log.CI(colored(f'evaluate the performance of {target} results.', 'blue'))
    model =  mdata.cmodel if model is None else model
    mdata_atrr = eval(f'mdata.{target}')[model]
    select_cv =  mdata.hypers[model]['cvidx'] if select_cv is None else select_cv

    if samples is None:
        samples = mdata.predicts[model]['samples'] if target=='predicts' else mdata.samples
    else:
        samples = samples
  
    resulty = [ mdata_atrr[_idx]['predicts'] for _idx in select_cv ]
    resulty = pd.concat(resulty, axis=0)
    
    prob_mean = resulty.filter(regex=("^(?!y_pred).*(_clf|_clf_c)$")).groupby([resulty.index], sort=False,).mean()
    pred_mode = resulty.filter(regex=("^(y_pred).*")).groupby([resulty.index], sort=False).apply(lambda x : x.mode(0).loc[0,:])
    pred_mode.columns += '_mode'
    for i in ['_clf','_clf_c']:
        ires = prob_mean.filter(regex="^(?!y_pred).*%s$"%(i))
        pred_mode[f'y_pred{i}'] = ires.columns.str.replace(f'{i}$', '', regex=True)[ires.values.argmax(1)]
        pred_mode[f'y_pred_maxscore{i}'] = ires.max(1)

    resulty = pd.concat([pred_mode, prob_mean], axis=1).loc[samples,:]
    eval(f'mdata.{target}')[model]['resulty'] = resulty
    return mdata

def _get_train_test(mdata, idx, Xy_mtx=None, cvs_idx=None):
    Xy_mtx  = [mdata.X,  mdata.y] if Xy_mtx is None else Xy_mtx
    cvs_idx = mdata.cv if cvs_idx is None else cvs_idx
    
    train_idx, test_idx=cvs_idx[idx]
    cvs_mtx = []
    for iXy in Xy_mtx:
        iXy = np.squeeze(iXy.copy())
        if len(iXy.shape)==1:
            cvs_mtx.extend([iXy[train_idx], iXy[test_idx]])
        else:
            cvs_mtx.extend([iXy[train_idx,:],iXy[test_idx,:]])
    return cvs_mtx

def Hypermodel(mdata, label_type=None, model=None, searchcv='HGSCV', searchlev='level1', test_size=0.3):
    label_type = mdata.attr['label_type'] if label_type is None else label_type
    model = mdata.cmodel if model is None else model

    if label_type == 'C':
        if searchcv in ['HGSCV', 'HRSCV']:
            CVS = ul.CrossvalidationSplit(n_splits=5, cvm='SFK')
        else:
            CVS = ul.CrossvalidationSplit(n_splits=5, test_size=test_size, cvm='SSS')
    elif label_type =='R':
        CVS = ul.CrossvalidationSplit(n_splits=3, n_repeats= 2, cvm='RSKF')    

    estimator  = Estimators(model, Type=label_type).getparams().estimator
    parameters = Estimators(model, Type=label_type, searchlev=searchlev).getparams().parameters

    estimat_para = {'cvs':CVS, 'searchcv':searchcv, 'searchlev':searchlev, 'estimator':estimator, 'parameters':parameters}
    return(estimat_para)

def Hyperparemet(mdata, idx, n_jobs=-1, score=None, n_iter=2500, calibme='isotonic'):
    n_classes = mdata.attr['n_classes']
    model = mdata.cmodel
    estimator = mdata.hypers[model]['estimator']
    parameters= mdata.hypers[model]['parameters']
    cvs = mdata.hypers[model]['cvs']
    searchcv = mdata.hypers[model]['searchcv']

    X_train, X_test, y_train, y_test = _get_train_test(mdata, idx)
    cf.log.CI( f'hyperparameter optimization in the {model} model......' )
    if ( n_classes >2 ) & (model in ['XGB']):
        _kargs = {'objective':'multi:softprob', 'use_label_encoder':False, 'verbosity':0}
        estimator  = estimator.set_params(**_kargs)
    if model in ['XGB']:
        n_jobs = 20
        if 'scale_pos_weight' in parameters.keys():
            pos_weight = y_train.value_counts(normalize=True)
            parameters['scale_pos_weight'] += ( (1-pos_weight)/pos_weight ).to_list()
            parameters['scale_pos_weight']  = list(set( parameters['scale_pos_weight'] ))

    ## add hyperopt Spearmint hyperparameter
    if (not searchcv) or (not parameters):
        clf = estimator
        cf.log.CI( 'decrepate hyperparameter optimization......' )
    elif searchcv == 'GSCV':
        clf = GridSearchCV(estimator, parameters,
                            n_jobs=n_jobs,
                            cv=cvs,
                            scoring=score,
                            error_score = np.nan,
                            return_train_score=True,
                            refit = True)
        cf.log.CI( 'GSCV hyperparameter optimization......')
    elif searchcv == 'HGSCV':
        clf = HalvingGridSearchCV(estimator, parameters,
                            factor=3,
                            resource='n_samples',
                            max_resources='auto', 
                            min_resources='exhaust', 
                            aggressive_elimination=False, 
                            random_state=None,
                            n_jobs=n_jobs,
                            cv=cvs,
                            scoring=score,
                            error_score = np.nan,
                            return_train_score=True,
                            refit = True,
                            verbose=0)
        cf.log.CI( 'HGSCV hyperparameter optimization......')
    elif searchcv == 'RSCV':
        clf = RandomizedSearchCV(estimator, parameters,
                                    n_jobs=n_jobs,
                                    cv=cvs,
                                    n_iter = n_iter,
                                    scoring=score,
                                    return_train_score=True,
                                    refit = True,
                                    error_score='raise')
        cf.log.CI( 'RSCV hyperparameter optimization......')
    elif searchcv == 'HRSCV':
        clf = HalvingRandomSearchCV(estimator, parameters,
                                    n_candidates='exhaust',
                                    factor=3,
                                    resource='n_samples',
                                    max_resources='auto', 
                                    min_resources='smallest', 
                                    aggressive_elimination=False, 
                                    random_state=None,
                                    n_jobs=n_jobs,
                                    cv=cvs,
                                    scoring=score,
                                    return_train_score=True,
                                    refit = True,
                                    error_score = 'raise', 
                                    verbose=0)
        cf.log.CI( 'HRSCV hyperparameter optimization......')
    
    if model in ['XGB_']:
        #y_train  = mdata.ym.transform(y_train)
        #y_test   = mdata.ym.transform(y_test)
        clf.fit(X_train, y_train,
                #eval_metric=["error", "logloss"],
                #eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_set=[(X_test, y_test)],
                eval_metric= 'auc',
                early_stopping_rounds=15,
                verbose=False)
    else:
        clf.fit(X_train, y_train)

    scores = dict()
    if hasattr(clf, 'best_estimator_'):
        import copy
        base_estimator = copy.deepcopy(estimator).set_params(**clf.best_estimator_.get_params())
        clf_r = clf.best_estimator_
        scores.update({'best_score_': clf.best_score_})
        cf.log.CI( '%s best hyper parameters in %s: %s'%(model, searchcv, clf.best_params_) )
    else:
        base_estimator = copy.deepcopy(estimator)
        clf_r = clf

    clf_c = CalibratedClassifierCV(base_estimator, cv=cvs, method=calibme)
    clf_c.fit(X_train, y_train)
    clf_r.fit(X_train, y_train)
    clfr_score = clf_r.score(X_train, y_train)
    clfc_score = clf_c.score(X_train, y_train)
    scores.update({'refit_score':clfr_score, 'cccv_refit_score':clfc_score})
    cf.log.CI( '%s scores: %s' %(model, scores) )

    hyper_para = { 'scores':scores, 
                    'estimator':estimator,
                    'parameters':parameters,
                    'clf':clf_r,
                    'clf_c':clf_c }
    return hyper_para

def Featurescoef(clf, model, n_classes=None):
    n_classes = len(clf.classes_) if n_classes is None else n_classes
    if model in ['MNB','BNB'] :
        importances= np.exp(clf.coef_)
        cf.log.CW(f'Note: Use coef_ exp values as the feature importance of the {model} estimator.')
    elif model in ['CNB'] :
        if n_classes ==2:
            importances= np.exp(-clf.feature_log_prob_)[1]
        else:
            importances= np.exp(-clf.feature_log_prob_)
        cf.log.CW(f'Note: Use feature_log_prob_ negative exp values as the feature importance of the {model} estimator.')
    elif model in ['GNB'] :
        if n_classes ==2:
            importances= clf.theta_[1]
        else:
            importances= clf.theta_
        cf.log.CW(f'Note: Use theta_ values as the feature importance of the {model} estimator.')
    elif model in ['MLP'] :
        def collapse(coefs):
            Coefs = coefs[0]
            for b in coefs[1:]:
                Coefs = Coefs.dot(b)
            return(Coefs/Coefs.sum(0))
        importances = collapse(clf.coefs_).T
        cf.log.CW(f'Note: Use coefs_ weighted average as the feature importance of the {model} estimator.')
    elif model in ['SVM', 'SVMrbf', 'nuSVMrbf']:
        dot_coef_ = clf.dual_coef_.dot( clf.support_vectors_ )
        import numpy
        numpy.seterr(divide='ignore',invalid='ignore')
        importances = (1-numpy.exp(-dot_coef_)) / (1+numpy.exp(-dot_coef_))
        cf.log.CW(f'Note: Use exp dot of support_vectors_ and dual_coef_ values as the feature importance of the {model} estimator.')
    else:
        for i in ['feature_importances_', 'coef_']:
            try:
                importances= eval(f'clf.{i}')
                cf.log.CI(f'Use {i} as the feature importance of the {model} estimator.')
                break
            except AttributeError:
                importances = []
                cf.log.CW(f'Note: Cannot find the feature importance attributes of {model} estimator.')
    return (importances)
    '''
    importances = np.asarray(importances)
    featues = clf.feature_names_in_
    if not len(featues) in importances.shape:
        cf.log.CW(f'Note: features numer({len(featues)}) is inconsistent with importance shape({importances.shape}) in {model} estimator.')
    else:
        importances = pd.DataFrame(importances, index=featues)
    if not idx in mdata.learns[model]:
        mdata.learns[model][idx] ={'featurescoef':{}}
    mdata.learns[model][idx]['featurescoef'].update({clf_base:importances})
    return mdata
    df_import = pd.DataFrame(np.array(importances).T)
    if not df_import.empty :
        df_import.index=index_
        df_import = df_import[(df_import != 0).any(1)]
    return df_import
    '''

def Predict_prob(clf, X_matrix, y_label=None, n_classes=None):
    if y_label:
        clf.fit(X_matrix, y_label)

    _predict = clf.predict(X_matrix)
    try:
        _proba = clf.predict_proba(X_matrix)
    except AttributeError:
        _proba = clf._predict_proba_lr(X_matrix)
        cf.log.CW('Note: LinearSVC, SGB use _predict_proba_lr based on decision_function as predict_proba in GridsearchCV.')
    except:
        _proba = _predict_proba_lr( clf.decision_function(X_matrix) )
        cf.log.CW('predict_proba use sigmoid transversion based on decision_function!')

    if (n_classes and n_classes != _proba.shape[1]):
        raise Exception('The columns length of predict probability is wrong!') 

    pre_prob = [_predict, _proba]
    return(pre_prob)

def _scoring(y_true, y_trueb, y_predict, y_score, name=None, classes_=None):
    _score = pd.Series({  
                'accuracy'  : accuracy_score(y_true, y_predict),
                'accuracy_b': balanced_accuracy_score(y_true, y_predict),
                'f1_score'  : f1_score(y_true, y_predict, average='macro'),
                'precision' : precision_score(y_true, y_predict, average='macro'),
                'prec_aver' : average_precision_score( y_trueb, y_score),
                'recall'    : recall_score(y_true, y_predict, average='macro'),
                'roc'       : roc_auc_score(y_true, y_score, multi_class='ovr'),
    })
    
    for (_n, _m) in [('roc',roc_auc_score), ('prec',average_precision_score)]:
        for i in range(y_score.shape[1]):
            k = i if classes_ is None else classes_[i]
            _score[f'{_n}_{k}']  = _m(y_trueb[:, i], y_score[:, i])
    #Report = classification_report(y_true=y_true, y_pred=y_predict, output_dict=False)
    #disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))
    _score.name = name
    return _score

def evaluat_scoring0(mdata, idx, clf_bases=['clf','clf_c'], model=None):
    model =  mdata.cmodel if model is None else model
    classes_ = mdata.ym.classes_
    samples  = mdata.samples
    features = mdata.features
    scores= []
    yyy   = []
    for _c in clf_bases:
        y_pred = mdata.learns[model][idx][_c]['predict']
        y_score = mdata.learns[model][idx][_c]['predict_proba']
        yyy_mtx = [mdata.y, mdata.yb, y_pred, y_score]
        
        yyy.append(np.c_[y_pred, y_score])
        train_test = _get_train_test( mdata, idx, Xy_mtx=yyy_mtx)
        scores.append( _scoring(*train_test[0::2], name=f'{_c}_train',classes_=classes_) )
        scores.append( _scoring(*train_test[1::2], name=f'{_c}_test', classes_=classes_) )
    
    scores = pd.concat(scores, axis=1)
    columns = [f'{i}_{_c}' for _c in clf_bases for i in ['y_pred']+classes_.tolist()]
    yyy = pd.DataFrame(np.hstack(yyy), index=samples, columns=columns)
    yyy = yyy.iloc[mdata.cv[idx][1]]
    
    mdata.learns[model][idx]['predicts']=yyy
    mdata.learns[model][idx]['scoring']=scores
    cf.log.CI(f'SCORING:\n{scores}')
    return mdata
    
    
def evaluat_scoring(mdata, idx, samples = None, labels = None, target='learns', clf_bases=['clf','clf_c'], model=None):
    model =  mdata.cmodel if model is None else model
    classes_ = mdata.ym.classes_
    if samples is None:
        samples = mdata.predicts[model]['samples'] if target=='predicts' else mdata.samples
    else:
        samples = samples

    if not labels is None:
        y_labels  = labels
        y_blabels = mdata.ym.transform(y_labels)
    else:
        y_labels = mdata.predicts[model]['y'] if target=='predicts' else mdata.y
        y_blabels = mdata.predicts[model]['yb'] if target=='predicts' else mdata.yb
    features = mdata.features

    yyy   = []
    scores= []
    for _c in clf_bases:
        y_pred  = eval(f'mdata.{target}')[model][idx][_c]['predict']
        y_score = eval(f'mdata.{target}')[model][idx][_c]['predict_proba']
        yyy.append(np.c_[y_pred, y_score])
        
        if not y_labels is None:
            yyy_mtx = [y_labels, y_blabels, y_pred, y_score]
            if target=='learns':
                train_test = _get_train_test( mdata, idx, Xy_mtx=yyy_mtx)
                scores.append( _scoring(*train_test[0::2], name=f'{_c}_train',classes_=classes_) )
                scores.append( _scoring(*train_test[1::2], name=f'{_c}_test', classes_=classes_) )
            elif target=='predicts':
                scores.append( _scoring(*yyy_mtx, name=f'{_c}_predict',classes_=classes_) )

    scores = pd.concat(scores, axis=1) if len(scores)>0 else []
    columns = [f'{i}_{_c}' for _c in clf_bases for i in ['y_pred']+classes_.tolist()]
    yyy = pd.DataFrame(np.hstack(yyy), index=samples, columns=columns)

    if target=='learns':
        yyy = yyy.iloc[mdata.cv[idx][1]]
    
    eval(f'mdata.{target}')[model][idx]['predicts']=yyy
    eval(f'mdata.{target}')[model][idx]['scoring']=scores
    if len(scores)>0:
        cf.log.CI(f'SCORING:\n{scores}')
    return mdata
