from sklearn.linear_model import (Lasso, LassoCV, ElasticNetCV, LassoLarsIC, SGDClassifier, LogisticRegressionCV,LogisticRegression)
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC,SVR,LinearSVC,NuSVC,NuSVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from scipy.stats import randint,uniform
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
#from bartpy.sklearnmodel import SklearnModel
#from bartpy.extensions.baseestimator import ResidualBART

class Estimators():
    def __init__(self, modle, Type='C', searchlev='level1', *array, **dicto):
        self.modle = modle
        self.Type  = Type
        self.searchlev = searchlev
        self.array = array
        self.dicto = dicto

    def getparams(self):
        if self.Type == 'C' :
            Parameters = self.Classification()
        elif self.Type == 'R' :
            Parameters = self.Regression()
        self.model      = Parameters[self.modle]
        self.estimator  = self.model['estimator']
        self.parameters = self.model['parameters'][self.searchlev] if self.searchlev in ['level1', 'level2'] else {}
        return self

    def Classification(self):
        return {
        'RF'   : {
            'estimator' : RandomForestClassifier(oob_score=True, n_estimators=100, n_jobs=10),
            'parameters': {
                'level1'  : { 'max_features'   : [0.6, 0.7, 0.8, 0.9, 'sqrt','log2', None],
                            'max_depth'        : [3, 4, 5, None],
                            'n_estimators'     : [100],
                            'class_weight'     : ['balanced', 'balanced_subsample', None], 
                            'min_samples_split': [2, 3, 4, 5 ],
                            'min_samples_leaf' : [2, 3, 4, 5 ],
                            'criterion'        : ['gini', 'entropy'] },
                'level2'  : { 'max_features': [ 0.6, 0.7, 0.8, 0.9, 0.99, 'sqrt','log2', None ],
                            'max_depth'   : [ 3, 4, 5, 6,  None ],
                            'n_estimators': [ 100 ],
                            'criterion'   : ['gini', 'entropy'],
                            'class_weight': ['balanced', 'balanced_subsample', None], 
                            'min_samples_split' : [ 2, 3, 4, 5, 6, 8, 11, 15],
                            'min_samples_leaf'  : [ 1, 2, 3, 4, 5, 6, 7, 9, 11, 15],
                            }
                }},

        'GBDT'          : {
            'estimator' : GradientBoostingClassifier( n_estimators=200, learning_rate=0.1, subsample=1.0, max_depth=5),
            'parameters': {
                'level1'  : { 'max_features': (0.5, 0.75, 0.8, 0.9, 'auto'),
                            'loss' : ['deviance', 'exponential'],
                            'max_depth': [ 3, 4, 5 ],
                            'n_estimators': [100],
                            'learning_rate': [0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
                            'min_samples_split':[2, 3, 4, 5],
                            'subsample': [0.75, 0.85, 0.95 ]},
                'level2'  : { 'max_features': (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'auto', 'log2', None),
                            'loss' : ['deviance', 'exponential'],
                            'max_depth': [ 3, 4, 5, 6, 7, 8, 9, 11, 15],
                            'n_estimators': [100],
                            'learning_rate': [0.005, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                            'min_samples_split':[2, 3, 4, 5, 6, 8],
                            'subsample': [0.7, 0.8, 0.9, 1]},
                }},
        'CATB'          : {
            'estimator' : CatBoostClassifier(iterations=500,
                         learning_rate=None, #0.03,
                         depth=6,
                         l2_leaf_reg=3,
                         model_size_reg=None,
                         rsm=None,
                         loss_function='Logloss',
                         target_border=None,
                         #custom_loss=['AUC', 'Accuracy'],
                         #eval_metric='AUC,
                         border_count =None, #CPU 254 GPU 128
                         task_type=None, #CPU GPU
                         feature_border_type='GreedyLogSum',
                         per_float_feature_quantization=None,                         
                         input_borders=None,
                         output_borders=None,
                         fold_permutation_block=1,
                         od_pval=None,
                         od_wait=None,
                         od_type=None,
                         verbose=True,
                         #logging_level='verbose',
                         metric_period=10,
                         early_stopping_rounds=None, 
                         cat_features=None,
                         max_ctr_complexity=4,
                         custom_metric=None,
                         random_state=None,
                         ),
            'parameters': {
                'level1'  : {'iterations': [10,50,200,500,700,5000,10000,50000],
                             'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
                             'depth': [5, 6, 8, 10, 12],
                             'loss_function': ['Logloss', 'MAE', 'CrossEntropy','MultiRMSE','RMSE', 'MultiClass', 'MultiClassOneVsAll'],
                             'custom_loss': [None, 'AUC', 'Accuracy'], 
                             'eval_metric': [None, 'AUC'],
                             'od_type': ['IncToDec', "Iter"],
                             'early_stopping_rounds': [None, 500],
                             },
                'level2'  : {'iterations': [10,50,200,500,700,5000,10000,50000],
                             'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
                             'depth': [5, 6, 8, 10, 12],
                             'loss_function': ['Logloss', 'MAE', 'CrossEntropy','MultiRMSE','RMSE', 'MultiClass', 'MultiClassOneVsAll'],
                             'custom_loss': [None, 'AUC', 'Accuracy'], 
                             'eval_metric': [None, 'AUC'],
                             'od_wait': [None, 500],
                             'od_type': ['IncToDec', "Iter"],
                             },
                }},
        'XGB'           : {
            'estimator' : XGBClassifier(
                            n_estimators =100,
                            objective    ='binary:logistic', # multi:softprob
                            booster      ='gblinear', #'gblinear', gbtree
                            silent       =True,
                            max_depth    =4,
                            missing      =None,
                            reg_alpha    =0,
                            reg_lambda   =1,
                            enable_categorical=True,
                            learning_rate=0.1,
                            n_jobs       =10 ),
            'parameters': {
                'level1'  : { 'colsample_bytree' : [0.75, 0.85, 0.95 ],
                            'subsample'        : [0.75, 0.85, 0.95 ],
                            'reg_alpha'        : [0, 0.1, 0.5, 1, 2  ],
                            'reg_lambda'       : [0.1, 0.5, 1, 2, 2.5 ],
                            'max_depth'        : [3, 4, 5, 6],
                            'n_estimators'     : [100],
                            'learning_rate'    : [0.01, 0.05, 0.1, 0.2],
                             'enable_categorical' :[True],
                            #'min_child_weight' : [1, 2],
                          },

                'level2'  : { 'subsample'        : [0.7, 0.8, 0.85],
                            'colsample_bytree' : [0.7, 0.8, 0.9],
                            'colsample_bylevel': [0.7, 0.8, 0.9],
                            'max_delta_step'   : [0, 1],
                            'colsample_bynode' : [1],
                            'scale_pos_weight' : [0.8, 1, 1.2],
                            'base_score'       : [0.5],
                            'n_estimators'     : [100],
                            'gamma'            : [0, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
                            'min_child_weight' : [1, 2, 3, 4, 5],
                            'min_samples_split': [2, 3, 4, 5, 6, 7],
                            'max_depth'        : [3, 4, 5, 6, 7, 8],
                            'reg_lambda'       : [0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.96, 1],
                            'reg_alpha'        : [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7],
                            'learning_rate'    : [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                           },
                }},

        'LGBM'          : {
            'estimator' : LGBMClassifier(
                            boosting_type='gbdt',
                            num_leaves=33,
                            max_depth=5,
                            learning_rate=0.1,
                            n_estimators=100,
                            class_weight=None,
                            min_split_gain=0.0,
                            min_child_weight=1e-3,
                            min_child_samples=10,
                            subsample=0.8,
                            subsample_freq=0,
                            colsample_bytree=0.8,
                            reg_alpha=0.0,
                            reg_lambda=0.0,
                            random_state=None,
                            n_jobs=-1,
                            silent=False,
                            importance_type='split',
                            verbose = -1,
                            ),
            'parameters': {
                'level1'  : { 'num_leaves'       : [9, 17, 33, 65],
                            'max_depth'        : [-1, 3, 4, 5, 6],
                            'learning_rate'    : [0.01, 0.05, 0.1, 0.2],
                            'n_estimators'     : [100],
                            'reg_alpha'        : [0.01, 0.1, 0.5, 1, 1.5, 2],
                            'reg_lambda'       : [0.01, 0.1, 0.5, 1, 1.5, 2],
                            'class_weight'     : ['balanced', None, ],
                            'subsample'        : [0.7, 0.78, 0.85],
                            'colsample_bytree' : [0.7, 0.78, 0.85],
                          },
                'level2'  : { 'num_leaves'       : [9, 17, 33, 65, 129],
                            'max_depth'        : [-1, 3, 4, 5, 6, 7],
                            'learning_rate'    : [0.01, 0.05, 0.1, 0.2],
                            'n_estimators'     : [100],
                            'reg_alpha'        : [0, 0.01, 0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],
                            'reg_lambda'       : [0, 0.01, 0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],
                            'class_weight'     : ['balanced', None, ],
                            'subsample'        : [0.7, 0.78, 0.85],
                            'colsample_bytree' : [0.7, 0.78, 0.85],
                          }
                }},

        'AdaB_DT'   : {
            'estimator' : AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', splitter='best',
                            class_weight='balanced', min_samples_leaf =1, max_features=None),
                            algorithm='SAMME.R', learning_rate=1, n_estimators=500),
            'parameters': {
                'level1'  : { 'n_estimators': range(400, 800, 1000),
                            'algorithm': ['SAMME', 'SAMME.R'],
                            "learning_rate": [0.3, 0.5, 0.7, 0.9, 0.95, 1, 2]},
                'level2'  : {}
                }},

        'MLP'           : {
            'estimator' : MLPClassifier(max_iter=3000,
                                        solver='lbfgs',
                                        alpha=1e-05,
                                        tol =1e-4,
                                        hidden_layer_sizes=(50,50,),
                                        random_state=None),
            'parameters': {
                'level1'  : { 'alpha': [10**i for i in [-6, -5, -4, -3, -1, 1]],
                            'max_iter':[15000],
                            'solver':['adam', 'lbfgs', 'sgd'],
                            'activation': ["logistic", 'identity', "relu", "tanh"],
                            'hidden_layer_sizes': [ (50, 30, 20, 10, 10,  ), (15, 25, 15, 10, ), (8, 15, 10, 10, ), (30, 10, 40, 10, ), (10, 20, 10, 10, )],
                            'learning_rate': ["constant", "invscaling", "adaptive"],
                            'tol' : [1e-4]},
                'level2'  : {'hidden_layer_sizes': [ (randint.rvs(8,30,1), randint.rvs(5,20,1), randint.rvs(5,30,1),) ,
                                                   (randint.rvs(8,40,1), randint.rvs(5,40,1), )],
                            'activation': ["logistic", 'identity', "relu", "tanh"],
                            'solver':['adam', 'lbfgs', 'sgd'],
                            'alpha': [10**i for i in [-6, -5, -4, -3, -1, 1]], #uniform(1e-06, 0.9),
                            'max_iter':[15000],
                            'learning_rate': ["constant", "invscaling", "adaptive"]}
                }},

        'LinearSVM'  : {
            'estimator' : LinearSVC(penalty='l2', dual=True, tol=0.0001, C=1, max_iter=2e6, random_state=None),
            'parameters': {
                'level1'  : { 'penalty': ['l2'],
                            'dual' : [True],
                            'loss': ['hinge', 'squared_hinge'],
                            'tol': [5e-7, 1e-7, 5e-6, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3, 1e-2, 5e-2 ],
                            #'C': [ 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 20, 50],
                            'C': np.power(10, np.linspace(-2,2, 20)),
                            'max_iter':[1.4e7]},
                'level2'  : {}
                }},
        'LinearSVMl1'   : {
            'estimator' : LinearSVC(penalty='l1', dual=False, tol=0.0001, C=1, max_iter=3e6, random_state=None),
            'parameters': {
                'level1'  : { 'penalty': ['l1', 'l2'],
                            'dual' : [False, True],
                            'loss': ['hinge', 'squared_hinge'],
                            'tol': [5e-7, 1e-7, 5e-6, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3, 1e-2, 5e-2 ],
                            'C': np.power(10, np.linspace(-2,2, 20)),
                            'max_iter':[1.4e7]},

                'level2'  : {}
                }},
        'SVMlinear'     : {
            'estimator' : SVC(kernel="linear", probability=True, C=1.0, decision_function_shape='ovr', random_state=None),
            'parameters': {
                'level1'  : [{'kernel': ['linear'],
                            'tol': [5e-7, 5e-6, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3, 1e-2, 5e-2, 5e-1 ],
                            'C': np.power(10, np.linspace(-2,2, 20)),
                           }],
                'level2'  : {}
                }},
        'SVMrbf'        : {
            'estimator' : SVC(kernel='rbf', gamma='scale', probability=True, C=1, decision_function_shape='ovr'),
            'parameters': {
                'level1'  : { 'kernel': ['rbf'],
                            'gamma': [1e1, 1e0, 1e-1, 1e-2, 1e-3,1e-4, 1e-5, 5e-3, 5e-4,'auto'],
                            'tol': [ 5e-7, 5e-6, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3, 1e-2, 5e-2, 5e-1 ],
                            'C': np.power(10, np.linspace(-2,2, 20)),
                          },
                'level2'  : {}
                }},
        'SVM'           : {
            'estimator' : SVC(kernel='rbf', gamma='scale', probability=True, C=1, decision_function_shape='ovr'),
            'parameters': {
                'level1'  : [{'kernel': ['rbf', 'sigmoid'],
                            'gamma': [1e1, 1e0, 1e-1, 1e-2, 1e-3,1e-4, 1e-5, 5e-3, 5e-4,'auto'],
                            'tol': [5e-7, 5e-6, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3, 1e-2, 5e-2, 5e-1 ],
                            'C': np.power(10, np.linspace(-2,2, 20)),
                           },
                           {'kernel': ['poly'],
                            'degree' : [2, 3, 4, 5],
                            'gamma': [1e1, 1e0, 1e-1, 1e-2, 1e-3,1e-4, 1e-5, 5e-3, 5e-4,'auto'],
                            'tol': [5e-7, 5e-6, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3, 1e-2, 5e-2, 5e-1 ],
                            'C': np.power(10, np.linspace(-2,2, 20)),
                           },
                          ],
                'level2'  : {}
                }},
        'nuSVMrbf'      : {
            'estimator' : NuSVC(kernel='rbf', gamma='scale', probability=True, nu=0.5, decision_function_shape='ovr'),
            'parameters': {
                'level1'  : { 'kernel': ['rbf'],
                            'nu': [0.8, 0.9, 1],
                            'gamma': [1e-2,1e-3,1e-4,1e-5,5e-2,5e-3,5e-4,'auto'],
                            'tol': [1e-6, 1e-5, 5e-5, 1e-4, 5e-3, 3e-3, 1e-3, 1e-2, 5e-2]},
                'level2'  : {}
                }},

        'SGD'           : {
            'estimator' : SGDClassifier(penalty='l2',loss='hinge', alpha=0.0001, max_iter =5000, tol=1e-3,n_jobs=2),
            'parameters': {
                'level1'  : [{ 'penalty' : ['l2','elasticnet'],
                             'loss': ['hinge','log','modified_huber','squared_hinge','perceptron' ],
                             'alpha': [5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1 ],
                             'l1_ratio': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 ],
                             'max_iter' : [5e5],
                             #'eta0' : [0, 0.0001, 0.001, 0.01, 0.1],
                             #'learning_rate' : ['optimal'],
                             'tol': [1e-3, 5e-3, 1e-4, 1e-5, 1e-2]},
                          ],
                'level2'  : {}
                }},

        'KNN'           : {
            'estimator' : KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2),
            'parameters': {
                'level1'  : { 'n_neighbors':[3, 5, 7, 10, 20],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto','ball_tree','kd_tree','brute'],
                            'leaf_size': [20, 30, 40],
                            'p': [2]},
                'level2'  : {}
                }},

        'RNN'           : {
            'estimator' : RadiusNeighborsClassifier(radius=1, weights='uniform',
                                                    algorithm='auto', leaf_size=30, p=2,
                                                    metric='minkowski', outlier_label=None),
            'parameters': {
                'level1'  : { 'radius': [1,2,3,4,5,10,15,20,23,26,30,35,40],
                            'weights': ['uniform', 'distance'],
                            'algorithm':['auto','ball_tree','kd_tree','brute'],
                            'leaf_size': [10,20,30,50,70,100,200],
                            'p': [1,2]},
                'level2'  : {}
                }},

        'GNB'           : {
            'estimator' : GaussianNB(priors=None, var_smoothing=1e-09),
            'parameters': {
                'level1'  : {'var_smoothing' : np.dot( np.array([[1e-11, 1e-10,1e-09,1e-08,1e-07]]).T, np.array([[1,3,5,7]]) ).flatten() },
                'level2'  : {}
                }},

        'BNB'           : {
            'estimator' : BernoulliNB(alpha=1.0, binarize=0.5, fit_prior=True, class_prior=None),
            'parameters': {
                'level1'  : {'alpha':[i/10 for i in range(1,21,1)], 'binarize' : [i/10 for i in range(1,10,1)] },
                'level2'  : {}
                }},

        'MNB'           : {
            'estimator' : MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None),
            'parameters': {
                'level1'  : {'alpha':[i/10 for i in range(1,21,1)]},
                'level2'  : {}
                }},

        'CNB'           : {
            'estimator' : ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False),
            'parameters': {
                'level1'  : {'alpha':[i/10 for i in range(1,21,1)]},
                'level2'  : {}
                }},

        'DT'    : {
            'estimator' : DecisionTreeClassifier(criterion='gini', splitter='best',
                          class_weight='balanced', min_samples_leaf =1, max_features=None),
            'parameters': {
                'level1'  : { 'max_features': (0.4, 0.5, 0.6, 0.7, 0.8, 'sqrt','log2'),
                            'min_samples_leaf' : (1,2,3) ,
                            'max_depth': (5, 8, 10, 15, 25, 30, None),
                            'criterion' : ['gini', 'entropy']},
                'level2'  : {}
                }},

        'LR'            : {
            'estimator' : LogisticRegression(random_state=None,
                                             solver='liblinear',
                                             penalty='l1',
                                             fit_intercept=True,
                                             max_iter=10000,
                                             l1_ratio=None,
                                             multi_class='auto'),
            'parameters': {
                'level1'  : [ { 'penalty'  : ['l1'],
                              'tol'      : [ 1e-3, 1e-4,  1e-5, ],
                              'l1_ratio' : [None],
                              'solver'   : ['liblinear', 'saga']},
                            { 'penalty'  : ['l2'],
                              'tol'      : [1e-3, 1e-4,  1e-5, ],
                              'l1_ratio' : [None],
                              'solver'   : ['lbfgs', 'sag']},
                            { 'penalty'  : ['elasticnet'],
                              'tol'      : [ 1e-3, 1e-4, 1e-5, ],
                              'l1_ratio' : [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                              'solver'   : ['saga']},
                          ],
                'level2'  : {}
                }},

        'LRCV'          : {
            'estimator' : LogisticRegressionCV(random_state=None,
                                             solver='saga',
                                             penalty='elasticnet',
                                             Cs = 10 ,
                                             cv = 8,
                                             fit_intercept=True,
                                             max_iter=6e3,
                                             n_jobs = 30,
                                             l1_ratios = [ 0.005, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9 ],
                                             multi_class='auto'),
            'parameters': {
                'level1'  : [ { 'penalty'  : [ 'elasticnet'],
                              'Cs'       : [ 10,
                                             np.power(10, np.arange(-4,4,0.4)),
                                             np.power(10, np.arange(-3,3,0.3))
                                           ],
                              'l1_ratios': [ [0.005, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9], ],
                              'max_iter' : [ 4e4 ],
                              'cv'       : [ 10  ],
                              'solver'   : [ 'saga']},
                          ],
                'level2'  : [ { 'penalty'  : [ 'l1'],
                              'Cs'       : [ np.power(10, np.arange(-2,4,0.3)) ],
                              'l1_ratios': [ None ],
                              'max_iter' : [ 6e4 ],
                              'cv'       : [ 10 ],
                              'solver'   : [ 'liblinear', 'saga']},
                            { 'penalty'  : [ 'l2'],
                              'l1_ratios': [ None ],
                              'max_iter' : [ 6e4 ],
                              'cv'       : [ 10   ],
                              'Cs'       : [ np.power(10, np.arange(-2,4,0.3)) ],
                              'solver'   : [ 'lbfgs', 'sag']},
                            { 'penalty'  : [ 'elasticnet'],
                              'Cs'       : [ np.power(10, np.arange(-2,4,0.3)) ],
                              'max_iter' : [ 6e4 ],
                              'cv'       : [ 10 ],
                              'solver'   : ['saga']},
                          ],
                }},

        'LassoCV'       : {
            'estimator' : LassoCV(cv=10,
                                  alphas=[0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 10],
                                  max_iter=20000,
                                  n_jobs=-1),
            'parameters': {
                'level1'  : { 'n_alphas': [200, 500],
                            'normalize':[False, True]},
                'level2'  : {}
                }},

        'Lasso'         : {
            'estimator' : Lasso(alpha=1, max_iter=15000),
            'parameters': {
                'level1'  : { 'alpha': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 10],
                            'max_iter': [15000],
                            'normalize':[False, True],
                            'precompute':[False, True]},
                'level2'  : {}
                }},

        'LLIC'          : {
            'estimator' : LassoLarsIC(criterion='aic',
                                      fit_intercept=True,
                                      verbose=False,
                                      normalize=True,
                                      precompute='auto',
                                      max_iter=10000,
                                      eps=2.220446049250313e-16,
                                      copy_X=True,
                                      positive=False),
            'parameters': {
                'level1'  : { 'criterion' : ['aic','bic'],
                            },
                'level2'  : {}
                }},

        'ENet'          : {
            'estimator' : ElasticNetCV( cv=10,
                                        alphas=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 10],
                                        max_iter=20000,
                                        random_state=None,
                                        n_jobs= 20,
                                        l1_ratio=[.01,.05, .1, .3, .5, .7, .9, .98]),
            'parameters': {
                'level1'  : { 'n_alphas' : [200, 500],
                            'max_iter' : [50000],
                            'tol'      : [1e-3, 1e-4, 1e-5, ],
                            'normalize': [False, True]},
                'level2'  : {}
                }},
        }

    def Regression(self):
        return {
        'RF'   : {
            'estimator' : RandomForestRegressor(n_jobs=-1, oob_score=True, n_estimators=200),
            'parameters': {
                'level1'  : { 'max_features': (0.4, 0.6, 0.8, 'auto'),
                            'max_depth': (5, 8, 12, 15, 25, None),
                            'n_estimators': [100, 200, 400, 600],
                            'min_samples_split':[2,5,10] },
                'level2'  : {}
                }},

        'Decis_Tree'    : {
            'estimator' : DecisionTreeRegressor(),
            'parameters': {
                'level1'  : { 'max_features': (0.4, 0.6, 0.8, 'auto'),
                            'max_depth': (5, 8, 12, 15, 25, None)},
                'level2'  : {}
                }},

        'SVM'           : {
            'estimator' : SVR(gamma='scale', kernel="linear"),
            'parameters': {
                'level1'  : {'C': [0.1, 0.5, 1, 5, 10, 100, 1000]},
                'level2'  : {}
                }},


        'MLP'           : {
            'estimator' : MLPRegressor(max_iter=3000, solver='lbfgs'),
            'parameters': {
                'level1'  : { 'alpha': [10**i for i in [-5, -3, -1, 1, 3]],
                            'max_iter':[10000, 15000],
                            'solver':['adam', 'lbfgs'],
                            'hidden_layer_sizes': [(50, 40, 20), (100, 30, 20), (200, 20), (10, 10, 10), (100, 20, 10)],
                            'learning_rate': ["constant", "invscaling", "adaptive"],
                            'activation': ["logistic", "relu", "tanh"]},
                'level2'  : {}
                }},

        'LassoCV'       : {
            'estimator' : LassoCV(cv=10, alphas=[0.0005, 0.001, 0.01,0.1, 0.5, 0.9, 1, 10], max_iter=5000),
            'parameters': {
                'level1'  : { 'n_alphas': [200, 500],
                            'max_iter':[5000, 10000, 15000],
                            'normalize':[False, True]},
                'level2'  : {}
                }},

        'Lasso'         : {
            'estimator' : Lasso(alpha=1, max_iter=5000),
            'parameters': {
                'level1'  : { 'alpha': [0.1, 0.5, 0.9, 1, 5, 10],
                            'max_iter': [5000, 10000, 15000],
                            'normalize':[False, True],
                            'precompute':[False, True]},
                'level2'  : {}
                }},

        'ENet'          : {
            'estimator' : ElasticNetCV( cv=10,
                                        alphas=[0.0001, 0.0005, 0.001, 0.01, 0.5, 0.1,1, 10],
                                        max_iter=5000,
                                        l1_ratio=[.01, .1, .5, .7, .9, .95, .99]),
            'parameters': {
                'level1'  : { 'n_alphas': [200, 500],
                            'max_iter':[5000, 10000, 15000],
                            'normalize':[False, True]},
                'level2'  : {}
                }},

        'BayesianRidge' : {
            'estimator' : '',
            'parameters': {
                'level1'  : {},
                'level2'  : {}
                }},

        'LogistR'       : {
            'estimator' : LogisticRegression(random_state=None,
                                              solver='lbfgs',
                                              fit_intercept=True,
                                              max_iter=500,
                                              multi_class='auto'),
            'parameters': {
                'level1'  : {},
                'level2'  : {}
                }},
        }

