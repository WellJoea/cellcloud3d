#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : modeldata.py
* @Author  : Zhou Wei                                     *
* @Date    : 2019/10/25 15:20:42                          *
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
import Utilities as ul
from collections import OrderedDict
import pandas as pd
import numpy as np
class _super:
    def __init(self, model='DT', idx=0, cvs=list(), hypers=dict(), learns=dict(), features_coef=list(),  ):
        self._model = model
        self._idx = idx
        self._hypers = hypers
        self._learns = learns
        self._cvs = cvs

    @property
    def cvlen(self):
        return len(self._cvs)

    @property
    def hypers(self):
        return self._hypers
    @hypers.setter
    def hypers(self, values):
        self._hypers.update(values)

    @property
    def learns(self):
        return self._learns
    @learns.setter
    def learns(self, values):
        self._learns.update(values)
    
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, values):
        self._model=values

class mdata:
    def __init__(self, X=None, y=None, ym=None, yb=None, yt=None, cv=None, 
                 hypers=dict(), learns=dict(), predicts=dict(),
                 progress = OrderedDict(), 
                 model=None, stackmodel=None,cmodel=None, features=None, 
                 samples=None, args=dict(), attr=dict()):
        self._X= X
        self._y= y
        self._ym=ym
        self._yt=yt
        self._yb=yb
        self._features=features
        self._samples=samples
        self._cv=cv
        self._hypers=hypers
        self._learns=learns
        self._predicts=predicts
        self._model=model
        self._stackmodel=stackmodel
        self._cmodel=cmodel
        self._progress=progress 
        self._args = args
        self._attr = attr

    def __repr__(self):
        return "<%s instance at %s>" % (self.__class__.__name__, id(self))
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    @property
    def X(self):
        return self._X
    @X.setter
    def X(self, values):
        self._X = values
    
    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, values):
        from sklearn.preprocessing import LabelBinarizer
        from sklearn.utils.multiclass import type_of_target
        self._y = ul.to_n_array(values)
        self._ym = LabelBinarizer().fit(values)
        self._yb = self.ym.transform(values)
        
        self.attr = {'n_classes': ul.get_n_classes(self._y)}
        if type_of_target(self._y) in ['continuous','continuous-multioutput']:
            self.attr = {'label_type': 'R'}
        elif type_of_target(self._y) in ['binary','multiclass','multiclass-multioutput','multilabel-indicator']:
            self.attr = {'label_type': 'C'}
        else:
            self.attr = {'label_type': 'unknown'}

    @property
    def ym(self):
        return self._ym
    @ym.setter
    def ym(self, values):
        self._ym = values

    @property
    def yb(self):
        return self._yb
    @yb.setter
    def yb(self, values):
        self._yb = values

    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, values):
        self._features = values

    @property
    def samples(self):
        return self._samples
    @samples.setter
    def samples(self, values):
        self.yt = pd.Series(self.y, index=values)
        self._samples = values

    @property
    def cv(self):
        return self._cv
    @cv.setter
    def cv(self, values):
        self._cv=values

    @property
    def hypers(self):
        return self._hypers
    @hypers.setter
    def hypers(self, values):
        self._hypers.update(values)
    @hypers.deleter
    def hypers(self):
        del self._hypers
        
    @property
    def learns(self):
        return self._learns
    @learns.setter
    def learns(self, values):
        self._learns.update(values)
    @learns.deleter
    def learns(self):
        del self._learns

    @property
    def predicts(self):
        return self._predicts
    @predicts.setter
    def predicts(self, values):
        self._predicts.update(values)
    @predicts.deleter
    def predicts(self):
        del self._predicts

    @property
    def progress(self):
        return self._progress
    @progress.setter
    def progress(self, values):
        self._progress.update(values)

    @property
    def args(self):
        return self._args
    @args.setter
    def args(self, values):
        self._args.update(values)

    @property
    def attr(self):
        return self._attr
    @attr.setter
    def attr(self, values):
        self._attr.update(values)

    @property
    def cmodel(self):
        return self._cmodel
    @cmodel.setter
    def cmodel(self, values):
        self._cmodel=values

    @property
    def stackmodel(self):
        return self._stackmodel
    @stackmodel.setter
    def stackmodel(self, values):
        self._stackmodel=values

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, values):
        self._model=values
        self.cmodel = self._model

        if not values in self.hypers:
            self._hypers[self._model] = dict()
            #self._hypers[self._model]['clf'] =  OrderedDict()
            #self._hypers[self._model]['clf_c'] =  OrderedDict()

        if not values in self.learns:
            self._learns[self._model] = dict()
            #self._learns[self._model]['predicts']= OrderedDict()
            #self._learns[self._model]['scoring'] = OrderedDict()
            #self._learns[self._model]\
            #    .update({ ic:{ ia:OrderedDict() 
            #                    for ia in ['featurescoef', 'predict', 'predict_proba']}
            #                for ic in ['clf','clf_c']})

        if not values in self.predicts:
            self._predicts[self._model] = dict()