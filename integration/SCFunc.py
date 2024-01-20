#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : SCFunc.py
* @Author  : Zhou Wei                                     *
* @Date    : 2021/06/13 18:43:32                          *
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

import scanpy as sc
import anndata as ad
import scvelo as scv
from scrublet import Scrublet
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
from scipy import sparse
from anndata import AnnData
import os
import re
from joblib import Parallel, delayed
import plotly.express as px #5.3.1
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#import plotly.offline as pyo
#pyo.init_notebook_mode()
#You need to change init_notebook_mode call and remove connected=True,
#if you want to work in offline mode.


#####decorators
def _default_kargs(_func):
    import inspect
    signature = inspect.signature(_func)
    return { k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty}

def Chdir(_func):
    from functools import wraps
    @wraps(_func)
    def wrapper(*args, **kargs):
        import os
        rawdir = os.getcwd()
        kkargs = _default_kargs(_func)
        kkargs.update(kargs)
        if ('outdir' in kkargs) and (kkargs['outdir']!=None ):
            newdir=kkargs['outdir']
            os.makedirs(newdir, exist_ok=True)
            os.chdir(newdir)
        Fun = _func(*args, **kargs)
        os.chdir(rawdir)
        print('The current workdir is '+ rawdir)
        return Fun
    return wrapper

def CHdir(outdir):
    def decorator(_func):
        from functools import wraps
        @wraps(_func)
        def wrapper(*args, **kargs):
            import os
            rawdir = os.getcwd()
            os.makedirs(outdir, exist_ok=True)
            os.chdir(outdir)        
            Fun = _func(*args, **kargs)
            os.chdir(rawdir)
            return Fun
        return wrapper
    return decorator

#######classes
class IOs:
    def __init__(self, *args, **kargs):
        self.args  = args
        self.kargs = kargs

    def getMX(self, IN):
        adata = sc.read_10x_mtx(IN, var_names='gene_symbols', cache=True)
        adata.var_names_make_unique()
        return adata

    def getH5(self, IN):
        adata = sc.read_10x_h5(IN)
        adata.var_names_make_unique()
        return adata

    def getIO(self, IN, OUT):
        os.makedirs(OUT, exist_ok=True)
        os.chdir(OUT)
        print('The outdir:' + os.getcwd())
        return sc.read_10x_mtx(IN, var_names='gene_symbols', cache=True)

    def getSCVLoom(self, fname,  cache=True):
        '''
        layers: matrix = ambiguous+spliced+unspliced
        .X ==spliced
        '''
        adata = scv.read(fname, obs_names='CellID', var_names='Gene', cache=cache )
        adata.var_names_make_unique()
        return adata

    def initinfo(self, adata, name=None):
        adata = adata.copy()
        adata.var.index.name = 'Gene'
        adata.var['mt'] = adata.var_names.str.startswith('MT-') 
        adata.var['ribo'] = adata.var_names.str.contains('^RP[SL]', regex=True)
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', regex=True)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)

        if name:
            adata.obs.index = name + ":" + adata.obs.index                                                        
        adata.obs.index.name = 'CellID'                                                                 
        adata.obs['barcordID'] = adata.obs.index
        return(adata)
        
    def addINFO(self, sinfo, genome='GRCh38', mitogene=None):
        adata = self.adata.copy()
        adata.obs['aggrnum'] = adata.obs.index.str.split('-').str[1]
        adata.obs.index.name = 'CellID'                                                                 
        adata.obs['barcordID'] = adata.obs.index
        adata.obs = adata.obs.reset_index().merge(sinfo, on='aggrnum', how="left").set_index('CellID')
        
        adata.var.index.name = 'Gene'
        adata.var['genome'] = genome
        if genome in ['GRCh38']:
            adata.var['mt'] = adata.var_names.str.startswith('MT-') 
        elif genome in ['Macaque']:
            mitogene = ["ND1", "ND2", "COX1", "COX2", "ATP8", "ATP6", "COX3",
                        "ND3", "ND4L", "ND4", "ND5", "ND6", "CYTB"]
            adata.var['mt'] = adata.var_names.isin(mitogene)
        if genome in ['mm10', 'mm']:
            adata.var['mt'] = adata.var_names.str.startswith('mt-') 
        elif mitogene:
            adata.var['mt'] = adata.var_names.isin(mitogene)

        adata.var['ribo'] = adata.var_names.str.contains('^RP[SL]|^Rp[sl]', regex=True)
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]|^Hb[^p]', regex=True)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)
        return adata

    def mergeAD(self, addict, sinfo, label='sampleid', **kargs):
        adata = ad.concat(addict, label=label, **kargs)

        advar = pd.concat([ v.var.drop('genome', axis=1) for k,v in addict.items()] , axis=0).drop_duplicates(keep='first')
        adata.var.index.name = 'Gene'
        adata.var = adata.var.join(advar)
        adata.var['mt'] = adata.var_names.str.startswith('MT-') 
        adata.var['ribo'] = adata.var_names.str.contains('^RP[SL]', regex=True)
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', regex=True)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)

        adata.obs.index = adata.obs.sampleid.str.cat( adata.obs.index, sep=":").str.rstrip('-1')                                                          
        adata.obs.index.name = 'CellID'                                                                 
        adata.obs['barcordID'] = adata.obs.index
        adata.obs = adata.obs.reset_index().merge(sinfo, on='sampleid', how="left").set_index('CellID')
        return(adata)

    def saveadata(self, adata, outpre='adata.Batch.NPCS',  todense=False):
        import scipy as sc
        adata.write(f'{outpre}.h5ad')
        metedata=adata.obs
        metedata.to_csv(f'{outpre}.metedata.obs.txt', sep='\t')
        #adatabbknn.write_loom('adata.%s.%s.loom'%(BATCH, NPCS), write_obsm_varm=True)
        adatakk = adata.copy()
        
        adatakk.X = adatakk.X.toarray() if (todense and sc.sparse.issparse(adatakk.X)) else adatakk.X
        if not 'barcordID' in adatakk.obs.columns:
            adatakk.obs['barcordID'] = adatakk.obs_names
        adatakk.obs = adatakk.obs[['barcordID']]
            
        #adatakk.var = adatakk.var.loc[:,adatakk.var.dtypes != 'bool']
        #adatakk.raw.var = adatakk.raw.var.loc[:,adatakk.raw.var.dtypes != 'bool']
        adatakk.var = adatakk.var.iloc[:,:0] 
        adatakk.var.index = adatakk.var.index.tolist()
        adatakk.raw.var.drop(adatakk.raw.var.columns, axis=1, inplace=True)
        adatakk.raw.var.index = adatakk.raw.var.index.tolist()
        adatakk.uns={}
        adatakk.write(f'{outpre}.seuratojb.h5ad')
    
    def humanAGGR(self, sinfo, label='sampleid'):
        #sinfo = sinfo.astype('category')
        DRG_hm7 = self.getH5('/home/gpfs/home/wulab17/WorkSpace/11Project/02DRG/00DataBase/', 'DRG7_20210602_aggr/')
        mapid = {'1': 'DRG1', '2': 'DRG3', '3': 'DRG4', '4': 'DRG5', '5': 'DRG8', '6': 'DRG9', '7': 'DRG10'}
        DRG_hm7idx = DRG_hm7.obs.index.to_series().str.split('-')
        DRG_hm7.obs.index = DRG_hm7idx.str[1].replace(mapid).str.cat(DRG_hm7idx.str[0], sep=':' )
        DRG_hm7.var.genome = 'GRCh38'

        HE10_DRG= self.getH5('/home/gpfs/home/wulab17/WorkSpace/11Project/02DRG/00DataBase/DRG_cellrangercount/', 'HE10_DRG')
        HE10_DRG.obs.index = 'HE10_DRG:' +HE10_DRG.obs.index.str.rstrip('-1')
        
        adata = ad.concat([DRG_hm7, HE10_DRG])

        advar = pd.concat([DRG_hm7.var, HE10_DRG.var] , axis=0).drop_duplicates(keep='first')
        adata.var.index.name = 'Gene'
        adata.var = adata.var.join(advar)
        adata.var['mt'] = adata.var_names.str.startswith('MT-') 
        adata.var['ribo'] = adata.var_names.str.contains('^RP[SL]', regex=True)
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', regex=True)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)

        adata.obs[label] = adata.obs.index.str.split(':').str[0]
        adata.obs.index.name = 'CellID'                                                                 
        adata.obs['barcordID'] = adata.obs.index
        adata.obs = adata.obs.reset_index().merge(sinfo, on=label, how="left").set_index('CellID')
        #adata.obs[label] = pd.Categorical(adata.obs[label], categories = sinfo[label].cat.categories)
        return(adata)

    def mccAGGR(self, sinfo, label='sampleid'):
        #sinfo = sinfo.astype('category')
        DRG_mcc2 = self.getH5('/home/gpfs/home/wulab17/WorkSpace/11Project/02DRG/00DataBase/', 'DRG_2macacasamples_aggr/')
        mccid = {'1': 'DRG6', '2': 'DRG7'}
        DRG_mcc2idx = DRG_mcc2.obs.index.to_series().str.split('-')
        DRG_mcc2.obs.index = DRG_mcc2idx.str[1].replace(mccid).str.cat(DRG_mcc2idx.str[0], sep=':' )
        DRG_mcc2.var.genome = 'MCC'
        mitogene = ["ND1", "ND2", "COX1", "COX2", "ATP8", "ATP6", "COX3", "ND3", "ND4L", "ND4", "ND5", "ND6", "CYTB"]
        adata = DRG_mcc2
        adata.var.index.name = 'Gene'

        adata.var['mt'] = adata.var_names.isin(mitogene)
        adata.var['ribo'] = adata.var_names.str.contains('^RP[SL]', regex=True)
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', regex=True)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)

        adata.obs[label] = adata.obs.index.str.split(':').str[0]
        adata.obs.index.name = 'CellID'                                                                 
        adata.obs['barcordID'] = adata.obs.index
        adata.obs = adata.obs.reset_index().merge(sinfo, on=label, how="left").set_index('CellID')
        #adata.obs[label] = pd.Categorical(adata.obs[label], categories = sinfo[label].cat.categories)
        return(adata)

from scrublet import Scrublet
class ScrubletNew(Scrublet):
    def __init__(self, *args,  threshold_method='Minimum', **kwargs):
        super(ScrubletNew, self).__init__(*args, **kwargs)
        self.threshold_method = threshold_method

    def call_doublets(self, threshold=None, verbose=True):
        '''
        #bimodal histogram python threshold
        #https://datascience.stackexchange.com/questions/20397/how-to-model-a-bimodal-distribution-of-target-variable
        #https://stackoverflow.com/questions/35990467/fit-mixture-of-two-gaussian-normal-distributions-to-a-histogram-from-one-set-of
        #https://scikit-image.org/docs/0.13.x/api/skimage.filters.html#skimage.filters.threshold_local
        #https://theailearner.com/2019/07/19/balanced-histogram-thresholding/
        #https://stackoverflow.com/questions/42149979/determining-a-threshold-value-for-a-bimodal-distribution-via-kmeans-clustering/42150214
        '''
        threshold_method = self.threshold_method
        if threshold is None:
            # automatic threshold detection
            # http://scikit-image.org/docs/dev/api/skimage.filters.html
            from skimage import filters
            import collections
            methods = collections.OrderedDict({
                        'Isodata': filters.threshold_isodata,
                        'Li': filters.threshold_li,
                        'Mean': filters.threshold_mean,
                        'Minimum': filters.threshold_minimum,
                        'Otsu': filters.threshold_otsu,
                        'Triangle': filters.threshold_triangle,
                        'Yen': filters.threshold_yen})
            try:
                for _i in methods.keys():
                    _t = methods[_i](self.doublet_scores_sim_)
                    print('Automaticall threshold of %s: %.4f'%(_i, _t))
                threshold = methods[threshold_method](self.doublet_scores_sim_)
                if verbose:
                    print("Automatically set threshold with %s at doublet score = %.4f"%(threshold_method, threshold))
            except:
                self.predicted_doublets_ = None
                if verbose:
                    print("Warning: failed to automatically identify doublet score threshold. Run `call_doublets` with user-specified threshold.")
                return self.predicted_doublets_

        Ld_obs = self.doublet_scores_obs_
        Ld_sim = self.doublet_scores_sim_
        se_obs = self.doublet_errors_obs_
        Z = (Ld_obs - threshold) / se_obs
        self.predicted_doublets_ = Ld_obs > threshold
        self.z_scores_ = Z
        self.threshold_ = threshold
        self.detected_doublet_rate_ = (Ld_obs>threshold).sum() / float(len(Ld_obs))
        self.detectable_doublet_fraction_ = (Ld_sim>threshold).sum() / float(len(Ld_sim))
        self.overall_doublet_rate_ = self.detected_doublet_rate_ / self.detectable_doublet_fraction_

        if verbose:
            print('Detected doublet rate = {:.1f}%'.format(100*self.detected_doublet_rate_))
            print('Estimated detectable doublet fraction = {:.1f}%'.format(100*self.detectable_doublet_fraction_))
            print('Overall doublet rate:')
            print('\tExpected   = {:.1f}%'.format(100*self.expected_doublet_rate))
            print('\tEstimated  = {:.1f}%'.format(100*self.overall_doublet_rate_))
        return self.predicted_doublets_

class Generalists:
    def __init__(self, adata, *args, **kargs):
        self.adata = adata
        self.args  = args
        self.kargs = kargs
        #EmptyDrops
 
    def SCruble(self, CountMX, callthre=None, defulthre=0.25, expected_doublet_rate=0.1, stdev_doublet_rate=0.02,
                scale_hist_obs='log', min_counts=5, min_cells=5, min_gene_variability_pctl=85, n_prin_comps=30,
                threshold_method='Minimum', **kargs):
        import scrublet as scr
        from scrublet import Scrublet
        print('*'*8, 'expected_doublet_rate: %s'%expected_doublet_rate, '*'*8)
        scrub = ScrubletNew(CountMX, expected_doublet_rate=expected_doublet_rate,
                               total_counts=None,
                                sim_doublet_ratio=2.0,
                                n_neighbors=None,
                                threshold_method=threshold_method,
                                stdev_doublet_rate=stdev_doublet_rate)
        doublet_scores, predicted_doublets = \
                scrub.scrub_doublets(min_counts=min_counts,  #2
                                        min_cells=min_cells, #3, 
                                        min_gene_variability_pctl=min_gene_variability_pctl, #85, 
                                        n_prin_comps=n_prin_comps, **kargs)
        if (not callthre is None):
            threshold = callthre
        elif (scrub.predicted_doublets_ is None) and (callthre is None):
            threshold = defulthre
        else:
            threshold = None
        if not threshold is None:
            print('***********', 'set the call_doublets(threshold=%s)'%threshold, '***********')
            predicted_doublets = scrub.call_doublets(threshold=threshold)

        scrub.plot_histogram(scale_hist_obs='log', scale_hist_sim='linear')
        scrub.set_embedding('UMAP', scr.get_umap(scrub.manifold_obs_, 10, min_dist=0.3))
        scrub.plot_embedding('UMAP', order_points=True)
        plt.show()
        print('***********doublet info', len(predicted_doublets+1), sum(predicted_doublets),'***********')
        return (doublet_scores, predicted_doublets)

    def CCscore(self, idata, 
        species = 'human',
        ccdata='/home/gpfs/home/wulab17/WorkSpace/01DataBase/CellCircle/cell_cycle_vignette_files/human_mouse_cell_cycle_genes.csv'):
        adata = idata.copy()
        ccgenes= pd.read_csv(ccdata)
        if species in ['human', 'hs']:
            genecol= 'hs_gene'
        elif  species in ['mouse', 'mm']:
            genecol= 'mm_gene'
        var_names = adata.raw.var_names if (adata.raw) else adata.var_names
        ccgenes   = ccgenes[(ccgenes[genecol].isin(var_names))].copy()
        s_genes   = ccgenes.loc[(ccgenes['cc_type']=='s_genes'), genecol].tolist()
        g2m_genes = ccgenes.loc[(ccgenes['cc_type']=='g2m_genes'), genecol].tolist()

        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        adata.obs['CC_diff'] = adata.obs.S_score - adata.obs.G2M_score
        return adata.obs[['CC_diff','S_score', 'G2M_score', 'phase']]

    def Doublet(self, splitby, replace=True, method=[ 'scruble' ]):
        adata = self.adata if replace else self.adata.copy()
        splitN = adata.obs[splitby].drop_duplicates(keep='first').tolist()
        if 'scruble' in method:
            doubleT = []
            for _g in splitN:
                print('*'*8, 'Scrublet scoring ' + _g, '*'*8)
                D = self.SCruble(adata[adata.obs[splitby] == _g,].X, callthre=0.25)
                idx = adata.obs.index[adata.obs[splitby]==_g]

                doubleT.append(pd.DataFrame(np.array(D).T, index=idx, columns=['doubletS', 'doubletP']))
            doubleT   = pd.concat(doubleT,axis=0)
            adata.obs = pd.concat([adata.obs, doubleT], axis=1) 
        return adata

    def GetScore(self, splitby, method=['scruble', 'ccscore'], 
                 species = 'human',
                 scthre=dict(), specified=None, **kargs):
        adata =  self.adata.copy()
        Score = []       
        adatadict = TransAdata(adata).splitAD(splitby)
        for _g, idata in adatadict.items():
            if (not specified is None) and (not _g in specified): continue
            idata = idata.copy()
            iscore= []
            callthre = None if (not _g in scthre) else scthre[_g]
            print('*'*8, 'getting the score of '+str(_g), '*'*8)
            if 'scruble' in method:
                idx = adata.obs.index[adata.obs[splitby]==_g]
                DBs = self.SCruble(idata.X.copy(), callthre=callthre, **kargs)
                DBs = pd.DataFrame(np.array(DBs).T, index=idx, columns=['doubletS', 'doubletP'])
                DBs['doubletP'] = DBs['doubletP'].astype(bool)
                iscore.append(DBs)
            if 'ccscore' in method:
                CCs = self.CCscore(idata.copy(), species=species)
                iscore.append(CCs)
            Score.append( pd.concat(iscore, axis=1) )
        Score = pd.concat(Score, axis=0)
        adata.obs = pd.concat([adata.obs, Score], axis=1)
        return adata

class Preprocession:
    def __init__(self, adata, *args, **kargs):
        self.adata = adata
        self.args  = args
        self.kargs = kargs

    def Normal(self, replace=False, saveraw=True, savecounts=False, islog=True, target_sum=1e4, **kargs):
        adata = self.adata if replace else self.adata.copy()
        if savecounts:
            adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=target_sum, **kargs)
        if islog:
            sc.pp.log1p(adata)
        if saveraw:
            adata.raw = adata 
        return adata

    def HVGs(self, batch_key=None, replace=False, minnbat=2, min_mean=0.0125, min_disp=0.5, subset=False,
             flavor='seurat', max_mean=4, layer=None, n_bins=20, **kargs):
        adata = self.adata if replace else self.adata.copy()
        sc.pp.highly_variable_genes(adata, batch_key = batch_key, min_mean=min_mean, min_disp=min_disp,
                                    flavor=flavor,
                                    layer=layer, n_bins=n_bins,max_mean=max_mean, subset=subset,**kargs)
        if batch_key:
            print(adata.var.highly_variable_nbatches.value_counts())
            if (not minnbat is None):
                adata.var.highly_variable = ((adata.var.highly_variable_nbatches>=minnbat) & (adata.var.highly_variable))
        print(adata.var.highly_variable.sum())
        #sc.pl.highly_variable_genes(adata)
        return adata

    def dropFuncGene(self, replace=False,  dropMT=False, dropRibo=False, dropHb=False):
        adata = self.adata if replace else self.adata.copy()

        if  not 'mt' in adata.var.columns:
            adata.var['mt'] = adata.var_names.str.contains('^M[Tt]', regex=True)

        if not 'ribo' in adata.var.columns:
            adata.var['ribo'] = adata.var_names.str.contains('^RP[SL]', regex=True)

        if not 'hb' in adata.var.columns:
            adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', regex=True)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)

        MTinHGVs = adata.var['mt'] & (adata.var.highly_variable)
        RiboinHGVs = adata.var['ribo'] & (adata.var.highly_variable)
        HbHGVs = adata.var['hb'] & (adata.var.highly_variable)
        print('MT in HGVs: %s'%MTinHGVs.sum())
        print('Ribo in HGVs: %s'%RiboinHGVs.sum())
        print('Hb in HGVs: %s'%HbHGVs.sum())

        if dropMT:
            adata.var.highly_variable = ((~adata.var['mt']) & (adata.var.highly_variable))
        if dropRibo:
            adata.var.highly_variable = ((~adata.var['ribo']) & (adata.var.highly_variable))
        if dropRibo:
            adata.var.highly_variable = ((~adata.var['hb']) & (adata.var.highly_variable))

        print(adata.var.highly_variable.sum())
        return adata

    def Scale(self, usehvgs=True, replace=False, n_jobs=50, zero_center=True, max_value=None, vargress=None):
        adata = self.adata if replace else self.adata.copy()
        adata = adata[:, adata.var.highly_variable] if usehvgs else adata
        if (not vargress is None) and (len(vargress)>0):
            #vargress=['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'CC_diff']
            sc.pp.regress_out(adata, vargress, n_jobs =n_jobs) #overcorrected
        sc.pp.scale(adata, zero_center=zero_center, max_value=max_value)
        '''
        sc.pp.recipe_zheng17(adata)
        sc.pp.recipe_weinreb17
        sc.pp.recipe_seurat
        '''
        return adata

    def NormScale(self, batch_key='sampleid', isnormal=True, isscale=True, n_top_genes=None, n_jobs=None,
                  local_features = None,
                  dropMT=False, dropRibo=False, dropHb=False,
                  savecounts=False, saveraw=True, vargress=None, usehvgs=True, **kwargs):
        print('input raw counts adata!')
        adata = self.adata.copy()
        if savecounts:
            adata.layers['counts'] = adata.X.copy()
        adata = Preprocession(adata).Normal() if isnormal else adata
        if saveraw: 
            adata.raw = adata 
        if local_features is None:
            adata = Preprocession(adata).HVGs(batch_key=batch_key, n_top_genes=n_top_genes, **kwargs)
        else:
            adata.var["highly_variable"] = adata.var_names.isin(local_features)
            print(f'the number of local features is {len(local_features)}.')
        adata = Preprocession(adata).dropFuncGene(replace=False,  dropMT=dropMT, dropRibo=dropRibo, dropHb=dropHb)
        if local_features is None:
            sc.pl.highly_variable_genes(adata)
        adata = adata[:, adata.var.highly_variable] if usehvgs else adata
        adata = Preprocession(adata).Scale(n_jobs=n_jobs, usehvgs=usehvgs, vargress=vargress) if isscale else adata
        return adata

    def highly_variable_genes_pr(self, markers=None):
        adata = self.adata.copy()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        hvgs = adata.var["highly_variable"]

        ax.scatter(
            adata.var["mean_counts"], adata.var["residual_variances"], s=3, edgecolor="none"
        )
        ax.scatter(
            adata.var["mean_counts"][hvgs],
            adata.var["residual_variances"][hvgs],
            c="tab:red",
            label="selected genes",
            s=3,
            edgecolor="none",
        )
        if not markers is None:
            ax.scatter(
                adata.var["mean_counts"][np.isin(adata.var_names, markers)],
                adata.var["residual_variances"][np.isin(adata.var_names, markers)],
                c="k",
                label="known marker genes",
                s=10,
                edgecolor="none",
            )
        ax.set_xscale("log")
        ax.set_xlabel("mean expression")
        ax.set_yscale("log")
        ax.set_ylabel("residual variance")

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        plt.legend()
        plt.show()

    def HVG_PR(self, batch_key=None, n_top_genes =None,  minnbat=2, subset=False,
                flavor='pearson_residuals',  layer=None, 
               dropMT=False, dropRibo=False, dropHb=False, **kargs):
        adata = self.adata.copy()
        sc.experimental.pp.highly_variable_genes(
                    adata, batch_key = batch_key,
                    flavor=flavor, layer=layer, n_top_genes=n_top_genes, subset=subset,**kargs)

        if batch_key:
            print(adata.var.highly_variable_nbatches.value_counts())
            if (not minnbat is None):
                adata.var.highly_variable = ((adata.var.highly_variable_nbatches>=minnbat) & (adata.var.highly_variable))

        MTinHGVs = adata.var['mt'] & (adata.var.highly_variable)
        RiboinHGVs = adata.var['ribo'] & (adata.var.highly_variable)
        HbHGVs = adata.var['hb'] & (adata.var.highly_variable)
        print('MT in HGVs: %s'%MTinHGVs.sum())
        print('Ribo in HGVs: %s'%RiboinHGVs.sum())
        print('Hb in HGVs: %s'%HbHGVs.sum())

        if dropMT:
            adata.var.highly_variable = ((~adata.var['mt']) & (adata.var.highly_variable))
        if dropRibo:
            adata.var.highly_variable = ((~adata.var['ribo']) & (adata.var.highly_variable))
        if dropRibo:
            adata.var.highly_variable = ((~adata.var['hb']) & (adata.var.highly_variable))

        print(adata.var.highly_variable.sum())
        highly_variable_genes_pr(adata)
        return adata
        '''
        adataR = neuron.copy()
        adataR = HVG_PR(adataR, batch_key='BatchID', n_top_genes=1000)
        adataR = adataR[:, adataR.var.highly_variable]
        sc.pp.normalize_total(adataR, inplace=True)
        adataR.X =  np.sqrt(adataR.X)
        '''

class DiffExp:
    def __init__(self, adata, inplace=True, *args, **kargs):
        self.adata = (adata.copy() if inplace else adata)
        self.args  = args
        self.kargs = kargs
        self._geneinfo()
    def _geneinfo(self, 
            hsmmgene ='/share/home/zhonw/WorkSpace/01DataBase/Homology/HOM_MouseHumanSequence.rpt.txt',
            gene_info='/share/home/zhonw/WorkSpace/01DataBase/Genome/Human/Gencode/V44/gencode.v44.annotation.gtf.bed'):

        gene_info = pd.read_csv(gene_info, sep='\t')
        gene_type = dict(zip(gene_info['gene_name'], gene_info['gene_type']))

        M2H = pd.read_csv(hsmmgene, sep='\t')
        M2Hdict = dict(zip(M2H['mouse'], M2H['human']))
        H2Mdict = dict(zip(M2H['human'], M2H['mouse']))
        self.gene_type = gene_type
        self.M2Hdict = M2Hdict
        self.H2Mdict = H2Mdict
        self.M2H = M2H
        return(self)

    def deg_table(self, unigroup='rank_genes_groups_filtered', pvals=0.05, fc=0.1, 
                  n_jobs=5, backend='threading',
                  order=['logfoldchanges']):
        adata = self.adata.copy()
        result = adata.uns[unigroup] #rank_genes_groups
        groups = result['names'].dtype.names

        def getEach(g, COL = ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']):
            G = pd.DataFrame([result[c][g] for c in COL], index=COL).T
            G['groups'] = g
            G = G[(~G['names'].isna())]
            G['pts'] = result['pts'].loc[G['names'],g].to_list()
            G['pts_rest'] = result['pts_rest'].loc[G['names'], g].to_list()
            G = G[((G['pvals'] < pvals) & (G['logfoldchanges']>fc))]
            G.sort_values(by=order, ascending=[False], inplace=True)
            return G

        DEG = Parallel( n_jobs= n_jobs, backend=backend)(map(delayed(getEach), groups))
        DEGdf = pd.concat(DEG, axis=0)
        del DEG
        return DEGdf
    
    def degsc(self, clumethod, degmethod='wilcoxon', min_in_group_fraction=0,
              only_filter=False,
              min_fold_change=0.25, max_out_group_fraction=1, use_raw=True, **kargs):
        if 'log1p' in self.adata.uns.keys():
            del self.adata.uns['log1p']
        if not only_filter:
            sc.tl.rank_genes_groups(self.adata, 
                                    clumethod, 
                                    method=degmethod,
                                    key_added = degmethod, 
                                    use_raw=use_raw, 
                                    pts =True)
        sc.tl.filter_rank_genes_groups(self.adata, 
                                    key=degmethod, 
                                    key_added=degmethod+'_filtered', 
                                    min_in_group_fraction=min_in_group_fraction,
                                    min_fold_change=min_fold_change,
                                    max_out_group_fraction=max_out_group_fraction,
                                    **kargs)

    def gettopn(self, DEG, ClustOrder=None,
                top_n=None,
                min_in_group_fraction=None,
                min_fold_change=None,
                max_out_group_fraction=None,
                min_diff_group=None):
        DEG = DEG.copy()
        ClustOrder = DEG.groups.drop_duplicates().tolist() if ClustOrder is None else ClustOrder
        
        if not min_in_group_fraction is None:
            DEG = DEG[(DEG.pts >=min_in_group_fraction)].copy()
        if not max_out_group_fraction is None:
            DEG = DEG[(DEG.pts_rest <=max_out_group_fraction)].copy()
        if not min_fold_change is None:
            DEG = DEG[(DEG.logfoldchanges >=min_fold_change)].copy()
        if not min_diff_group is None:
            DEG = DEG[(DEG.pts - DEG.pts_rest>=min_diff_group)].copy()
    
        DEGTop = DEG.sort_values(by=['groups', 'logfoldchanges'], ascending=[True, False])
        if not top_n is None:
            DEGTop=DEGTop.groupby(by='groups').head(top_n)
        DEGTopdict = { i: DEGTop[(DEGTop.groups==i)]['names'].tolist() for i in ClustOrder }
        return({'top':DEGTop, 'topdict':DEGTopdict})

class Unsupervise:
    def __init__(self, adata, *args, **kargs):
        self.adata = adata
        self.args  = args
        self.kargs = kargs

    def Add23d(self, method='umap', npcs=3, min_dist=0.1, **kargs):
        adata0 = self.adata.copy()
        adata = adata0.copy()
        if method=='umap':
            sc.tl.umap(adata, n_components=npcs, min_dist=min_dist, **kargs)
        elif method=='tsne':
            sc.tl.tsne(adata, n_pcs=npcs, n_jobs=-1, **kargs)

        adata0.obsm['X_%s_%sd'%(method, npcs)] = adata.obsm['X_' + method].copy()
        return adata0
        
    def Trans23d(self, method='umap', npcs=3, update=True):
        adata = self.adata.copy()
        K = adata.obsm['X_' + method].shape[1]
        if update:
            adata.obsm['X_%s_%sd'%(method, K)] = adata.obsm['X_' + method].copy()

        if 'X_%s_%sd'%(method, npcs) in adata.obsm:
            adata.obsm['X_' + method] = adata.obsm['X_%s_%sd'%(method, npcs)].copy()
        return adata

    def DimVisual(self, batch_key='sampleid', NPCS=50, Res=1, npcs=2, **kwargs):
        adata = self.adata.copy()
        sc.tl.leiden(adata,resolution=Res,  key_added = 'leiden_scanorama')
        sc.tl.louvain(adata,resolution=Res, key_added = 'louvain_scanorama')
        sc.tl.umap(adata, n_components=NPCS)
        sc.tl.tsne(adata, n_pcs=NPCS,n_jobs=-1, perplexity=50, learning_rate=500)

        sc.pl.embedding( adata, basis='X_scanorama', color=[batch_key])

        plt.figure(figsize=(24,8))
        sc.pl.umap(adata, color=[batch_key,'leiden_scanorama','louvain_scanorama'])
        plt.tight_layout()
        plt.savefig( 'scanorama.%s.%s.umap.pdf'%(NPCS, Res))
        sc.pl.tsne(adata, color=['sampleid','leiden_scanorama','louvain_scanorama'])

    def deg_table(self, unigroup='rank_genes_groups_filtered', pvals=0.05, fc=0.1, 
                  n_jobs=5, backend='threading',
                  order=['logfoldchanges']):
        adata = self.adata.copy()
        result = adata.uns[unigroup] #rank_genes_groups
        groups = result['names'].dtype.names

        def getEach(g, COL = ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']):
            G = pd.DataFrame([result[c][g] for c in COL], index=COL).T
            G['groups'] = g
            G = G[(~G['names'].isna())]
            G['pts'] = result['pts'].loc[G['names'],g].to_list()
            G['pts_rest'] = result['pts_rest'].loc[G['names'], g].to_list()
            G = G[((G['pvals'] < pvals) & (G['logfoldchanges']>fc))]
            G.sort_values(by=order, ascending=[False], inplace=True)
            return G

        DEG = Parallel( n_jobs= n_jobs, backend=backend)(map(delayed(getEach), groups))
        DEGdf = pd.concat(DEG, axis=0)
        del DEG
        return DEGdf

    def PCANei(self, batch_key='sampleid', n_comps=75, n_neighbors=15, NPCS=None, save=None):
        adata = self.adata.copy()
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_comps)
        sc.pl.pca(adata, color=batch_key)
        sc.pl.pca_variance_ratio(adata, log=True, n_pcs=n_comps, save=save)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=NPCS)
        return adata

    def _clust(self, iadata, method='leiden', ires=1, **kargs):
        import scanpy as sc
        ident = '%s_%.2f'%(method, ires)
        eval('sc.tl.%s'%method)(iadata, resolution=ires, key_added=ident, **kargs)
        return iadata.obs[ident]

    def Clust(self, clust=['leiden', 'louvain'], njobs=20, RESs=np.arange(0, 10, 0.25)):
        from scipy.sparse import csr_matrix
        adata = self.adata.copy()
        adataN= self.adata.copy()
        adataN.X= csr_matrix(adataN.X.shape)
        resolut = Parallel( n_jobs=njobs, verbose=1 )( delayed( self._clust )(adataN, method=_c, ires=_r) 
                            for _c in clust for _r in RESs)
        resolut = pd.concat(resolut, axis=1)
        adata.obs[resolut.columns] = resolut
        return adata

    def MAP(self, batch_key='sampleid', method='BBKNN', embedgroup=None, clust=['leiden', 'louvain'],
                 QCs=['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb','phase'],
                 clims= {'n_genes_by_counts':8000, 'total_counts':10000, 'pct_counts_mt':20, 'pct_counts_ribo':30, 'pct_counts_hb':30},
                 use_rep='X_pca', n_jobs=None, use_fast_tsne=False,
                doumap=True, dotsne=False, usepaga=False, dopaga=False, dodendrogram=False,
                dodpt=False, dodiffmap=False, dodrawgraph=False, dodiffmap_drawgraph=False,
                NPCS=50, npcs=2, diffnpcs=50, min_dist=0.3, Res=2, **kwargs):
        adata = self.adata.copy()
        if 'leiden' in clust:
            sc.tl.leiden(adata, resolution=Res)
        if 'louvain' in clust:
            sc.tl.louvain(adata, resolution=Res)
        COLS = clust if batch_key is None else [batch_key] + clust

        if usepaga or dopaga:
            sc.tl.paga(adata, groups='leiden', use_rna_velocity=False)
            sc.pl.paga(adata, color='leiden', fontsize=10,edge_width_scale=0.5, 
                       arrowsize=20, normalize_to_color=False, cmap=None,
                       save='%s.%s.%s.pada.pdf'%(method, NPCS, Res))
            sc.pl.paga_compare( adata, threshold=0.03, title='', right_margin=0.2, 
                               size=10, edge_width_scale=0.5,legend_fontsize=12, fontsize=12, 
                               frameon=False, edges=True, 
                               save='%s.%s.%s.paga_compare.pdf'%(method, NPCS, Res))
            '''
            sc.pl.paga_path(adata)
            sc.tl.umap(adata, n_components=npcs, min_dist=min_dist, init_pos='pada')
            sc.pl.umap(adata, color=COLS, wspace =0.6, legend_fontsize =10, ncols=len(COLS),
                        save='%s.%s.%s.pagaumap.pdf'%(method, NPCS, Res))
            '''
  
        if dodpt:
            sc.tl.dpt(adata)
            sc.pl.paga_path(adata)
        if dodendrogram:
            sc.tl.dendrogram(adata)
            
        if doumap:
            sc.tl.umap(adata, n_components=npcs, min_dist=min_dist, init_pos='spectral' if not usepaga else 'paga' )
            sc.pl.umap(adata, color=COLS, wspace =0.6, legend_fontsize =10, ncols=len(COLS),
                        save='.%s.%s.%s.mindist%s.pdf'%(method, NPCS, Res, min_dist))
            
            Plot(adata).QCmap(QCs=QCs, clims=clims, wspace =0.3, ncols=3,
                                save='.%s.%s.%s.mindist%s.QCFeature.pdf'%(method, NPCS, Res,min_dist))
            if (batch_key) and len(adata.obs[batch_key].unique()) >1:
                Plot(adata).splitplot(groupby=batch_key, 
                                      out='%s.%s.split.%s.%s.mindist%s.pdf'%(method, batch_key, NPCS, Res, min_dist))
            
        if dotsne:
            sc.tl.tsne(adata, n_pcs=NPCS, n_jobs=n_jobs, perplexity=30, learning_rate=1000, 
                       use_fast_tsne=use_fast_tsne, use_rep=use_rep)
            sc.pl.tsne(adata, color=COLS,wspace =0.6, legend_fontsize =10, ncols=len(COLS),
                    save='%s.%s.%s.tsne.pdf'%(method, NPCS, Res))
   
        if dodiffmap:
            sc.tl.diffmap(adata, n_comps=diffnpcs)
            sc.pl.diffmap(adata, color=COLS, components=['1,2','1,3'], wspace =0.6, legend_fontsize =10, ncols=len(COLS),
                            save='%s.%s.%s.diffmap.pdf'%(method, NPCS, Res))

        if dodrawgraph:
            sc.tl.draw_graph(adata, init_pos=None if not usepaga else 'paga')
            sc.pl.draw_graph(adata, color=COLS, wspace =0.6, legend_fontsize =10, ncols=len(COLS),
                            save='%s.%s.%s.drawgraph.pdf'%(method, NPCS, Res))

        if dodiffmap_drawgraph:
            sc.tl.diffmap(adata, n_comps=diffnpcs)
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=None,  key_added='diffnei', use_rep='X_diffmap')
            sc.tl.draw_graph(adata, neighbors_key='diffnei')
            sc.pl.draw_graph(adata, color=COLS, wspace =0.6, legend_fontsize =10, ncols=len(COLS),
                            save='%s.%s.%s.diffmap_drawgraph.pdf'%(method, NPCS, Res))
        try:
            embedgroup = embedgroup if embedgroup else batch_key
            sc.tl.embedding_density(adata, groupby=embedgroup)
            sc.pl.embedding_density(adata, groupby=embedgroup, bg_dotsize=10, fg_dotsize=40,
                                    save='%s.%s.%s.embedding_density.pdf'%(method, NPCS, Res))
        finally:
            return(adata)

    @Chdir
    def SplitCheck(self, split_key='sampleid', target=None, batch_key=None, 
                   isnormal=True, isscale=True, dropMT=False, dropRibo=False,
                   min_cells=100, min_mean=0.0125, min_disp=0.45, max_mean=6,
                   vargress=None, markers=None, n_comps=75, NPCS=50, n_top_genes=None, **kargs):
        print('input raw counts adata!')
        adata = self.adata.copy()
        adatadict = TransAdata(adata).splitAD(split_key)
        my_cmap=PlotUti().continuous_cmap(["lightgrey", "blue", "mediumblue",'red','yellow'])
        for k, adatai in adatadict.items():
            if (not target is None) and (len(target)>0) and (not k in target):
                continue      
            print(k)   
            if adatai.X.shape[0]< min_cells:
                print(f'the cell number of the batch id {k} is lower than {min_cells}. PASS!' )   
                continue
            sc.pp.filter_cells(adatai, min_genes=200)
            sc.pp.filter_genes(adatai, min_cells=10)    
            adatai = Preprocession(adatai.copy()).NormScale(batch_key=None, isnormal=isnormal, isscale=isscale,
                                                               dropMT=dropMT, dropRibo=dropRibo,
                                                               n_top_genes = n_top_genes,
                                                               minnbat=0, min_mean=min_mean, min_disp=min_disp, max_mean=max_mean)
    
            adatai = Unsupervise(adatai).PCANei(batch_key=split_key, n_comps=n_comps, save='.%s.%s.pca.vars.pdf'%(k, NPCS))
            adatai = Unsupervise(adatai).MAP(batch_key=batch_key, NPCS=NPCS, method='splitcheck_'+k, **kargs)
            adatai.obs.doubletP = adatai.obs.doubletP.map({False: 'Singlet', True:'Doublet'})
            sc.pl.umap(adatai, save='.%s.%s.umap.QC.pdf'%(k, NPCS),wspace =0.3, ncols=4,
               color=['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 
                      'pct_counts_ribo', 'pct_counts_hb','doubletP','phase'])
            if (not markers is None) and (len(markers)>0):
                sc.pl.umap(adatai, color=[i for i in markers if i in adatai.raw.var.index], ncols=9, color_map=my_cmap, 
                          save='.%s.%s.gene.classic.umap.png'%(k,NPCS))
    @Chdir
    def RawCheck(self, batch_key=None, method='quickcheck', n_comps=50, n_neighbors=15, **kargs):
        print('input scaled/normalized adata!')
        adata = self.adata.copy()
        adata = Unsupervise(adata).PCANei(batch_key=batch_key, n_neighbors=n_neighbors, n_comps=n_comps)
        adata = Unsupervise(adata).MAP(batch_key=batch_key, NPCS=n_comps, method=method, **kargs)
        return adata

class Integretion:
    def __init__(self, adata, *args, **kargs):
        self.adata = adata
        self.args  = args
        self.kargs = kargs

    def BBKNN(self, batch_key='sampleid', n_comps=100, use_rep='X_pca', NPCS=50, annoy_n_trees=10, savetree=False, **kwargs):
        print('input scaled adata!')
        adata = self.adata.copy()
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_comps)
        sc.external.pp.bbknn(adata, n_pcs=NPCS, use_rep=use_rep, batch_key=batch_key, annoy_n_trees=annoy_n_trees, **kwargs)
        sc.pl.pca(adata, color=batch_key)
        sc.pl.pca_variance_ratio(adata, log=True, n_pcs=n_comps)
        if savetree:
            adata.uns['BBKNN']={'connectivities': adata.obsp['connectivities'].copy(),
                                 'distances'    : adata.obsp['distances'].copy()}
        '''
        bbknn.ridge_regression(adata, batch_key=['batch'], confounder_key=['celltype'])
        sc.pp.pca(adata)
        bbknn.bbknn(adata, batch_key='batch')
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=['batch','celltype'])
        '''
        return(adata)

    def SCANOPCA(self, batch_key='sampleid', n_comps=50, NPCS=50, basis='X_pca', knn=20, sigma=15, batch_size=8000, 
              n_neighbors=15, n_knn=True, adjusted_basis='X_scanorama', **kwargs):
        print('input scaled adata!')
        adata = self.adata.copy()
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_comps)
        sc.pl.pca(adata, color=batch_key)
        sc.pl.pca_variance_ratio(adata, log=True, n_pcs=n_comps)
        sc.external.pp.scanorama_integrate(adata, batch_key, basis=basis, knn=knn, batch_size=batch_size,
                                           sigma=sigma, adjusted_basis=adjusted_basis, **kwargs )
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=NPCS, knn=n_knn, use_rep=adjusted_basis)
        return(adata)

    def SCANO(self, batch_key='sampleid', NPCS=50, save_correct_x=True, return_dimred=True ,
              isnormal=False, usehvgs=False, repca=False, svd_solver='arpack',
              vargress=None,
              knn=20, sigma=15, alpha=0.1, batch_size=8000, verbose=2, 
              minnbat=2, min_mean=0.0125, min_disp=0.5,  flavor='seurat', max_mean=4, 
              dropMT=False, dropRibo=False, n_bins=20,
              n_neighbors=15, n_knn=True, adjusted_basis='X_scanorama', **kargs):
        print('input log normal adata!')
        parameters = locals()
        del parameters['self']
        print(parameters)
    
        adata = self.adata.copy()
        adataraw = adata.raw.to_adata().copy()
        adata = Preprocession(adata).Normal() if isnormal else adata

        if usehvgs:
            adata = Preprocession(adata).HVGs(batch_key=batch_key, minnbat=minnbat, min_mean=min_mean, min_disp=min_disp,
                                              max_mean=max_mean, dropMT=dropMT, dropRibo=dropRibo)
            adata.var.highly_variable = nordata.var.highly_variable
            adata = adata[:, adata.var.highly_variable] 
            
        alldata = TransAdata(adata).splitAD(batch_key)
        adatas  = list(alldata.values())
        print("the order is: %s"%(list(alldata.keys())))
        import scanorama
        from scipy.sparse import csr_matrix
        #gamma=0.5*sigma sigma lower, biase geater
        if save_correct_x:
            adatas = scanorama.correct_scanpy(adatas, return_dimred=return_dimred, dimred=NPCS, sigma=sigma, 
                                                 alpha=alpha, knn=knn, batch_size=batch_size, verbose=verbose, **kargs)
        else:
            scanorama.integrate_scanpy(adatas, dimred=NPCS, sigma=sigma, 
                                        alpha=alpha, knn=knn, batch_size=batch_size, verbose=verbose, **kargs)

        adatas = [idata[:, adata.var_names].copy() for idata in adatas ]
        adatas = ad.concat(adatas.copy())
        adatas.X = csr_matrix(adatas.X.toarray())
        adatas = adatas[adata.obs_names, adata.var_names].copy()
        adata.obsm[adjusted_basis] = adatas.obsm[adjusted_basis]
        if save_correct_x:
            print(f'--> added\n    {adjusted_basis}_corX, array')
            adata.X = adatas.X.copy()
            adata.uns[f'{adjusted_basis}_corX']=adatas.X.copy()
        print('The X_pca was corrected. So only downsteam processes of PCA are need to tone.')
        adata.raw = adataraw
        if (repca):
            #adata = Preprocession(adata).Scale(n_jobs=n_jobs, vargress=vargress)
            sc.tl.pca(adata, svd_solver=svd_solver, n_comps=NPCS)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=NPCS, knn=n_knn, use_rep='X_pca')
        else:
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=NPCS, knn=n_knn, use_rep=adjusted_basis)
        
        adata.uns['scanorama_para'] = parameters
        return (adata)


    def Harmony(self, batch_key='sampleid', basis='X_pca', NPCS=50,
                  n_neighbors=15, n_knn=True, adjusted_basis='X_pca_harmony', **kargs):
        print('input scaled adata!')
        adata = self.adata.copy()
        sc.tl.pca(adata, svd_solver='arpack', n_comps=NPCS)
        sc.pl.pca(adata, color=batch_key)
        sc.pl.pca_variance_ratio(adata, log=True, n_pcs=NPCS)
        sc.external.pp.harmony_integrate(adata, plot_convergence=True, key=batch_key ,basis=basis,
                                         adjusted_basis=adjusted_basis, **kargs)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=NPCS, knn=n_knn, use_rep=adjusted_basis)
        sc.pl.scatter(adata, color=batch_key, basis=re.sub('^X_', '', adjusted_basis))
        return(adata)

    def Combat(self, batch_key='sampleid', isnormal=False, isscale=False, n_jobs=None, usehvgs=True,
                vargress=['total_counts', 'n_genes_by_counts', 'pct_counts_mt','CC_diff'],**kargs):
        print('input must raw counts adata!')
        adata = self.adata.copy()
        adata = Preprocession(adata).Normal() if isnormal else adata
        sc.pp.combat(adata, key=batch_key, inplace=True) #dense matrix format 
        adata = Preprocession(adata).HVGs(batch_key=batch_key, **kargs)
        adata = adata[:, adata.var.highly_variable] if usehvgs else adata
        adata = Preprocession(adata).Scale(n_jobs=n_jobs, vargress=vargress) if isscale else adata
        return adata

    def CombatHgvs0(self, batch_key='sampleid', isnormal=True, isscale=False, n_jobs=None, usehvgs=True,
                    vargress=['total_counts', 'n_genes_by_counts', 'pct_counts_mt','CC_diff'],**kargs):
        print('input raw counts adata!')
        adata = self.adata.copy()
        adata = Preprocession(adata).Normal() if isnormal else adata
        adata = Preprocession(adata).HVGs(batch_key=batch_key, **kargs)
        adata = adata[:, adata.var.highly_variable] if usehvgs else adata
        sc.pp.combat(adata, key=batch_key, inplace=True) #dense matrix format 
        adata = Preprocession(adata).Scale(n_jobs=n_jobs, usehvgs=False, vargress=vargress) if isscale else adata
        return adata

    def CombatHgvs(self, batch_key='sampleid', isnormal=False, isscale=False, n_jobs=None, usehvgs=True, minnbat=3,
                   dropMT=False, dropRibo=False, dropHb=False,
                    n_top_genes=None, vargress=['total_counts', 'n_genes_by_counts', 'pct_counts_mt','CC_diff'], **kargs):
        print('input raw counts adata!')
        adata = self.adata.copy()
        nordata = Preprocession(adata).Normal()
        nordata = Preprocession(nordata).HVGs(batch_key=batch_key, minnbat=minnbat,
                                              n_top_genes=n_top_genes, **kargs)
        nordata = Preprocession(nordata).dropFuncGene(replace=False,  dropMT=dropMT, dropRibo=dropRibo, dropHb=dropHb)
        var_genes = nordata.var.highly_variable
        var_genes = var_genes.index[var_genes].tolist()

        adata = nordata if isnormal else adata
        if usehvgs:
            adata.var.highly_variable = nordata.var.highly_variable
            adata = adata[:, var_genes] 
        
        sc.pp.combat(adata, key=batch_key, inplace=True) #dense matrix format 
        adata = Preprocession(adata).Scale(n_jobs=n_jobs, usehvgs=False, vargress=vargress) if isscale else adata
        return adata
    
    
    def MNN(self, isnormal=True, isscale=True, batch_key='sampleid', NPCS=50, minnbat=2, n_top_genes=None, 
            s_jobs=50, n_jobs=60, dropMT=False, vargress=['total_counts', 'n_genes_by_counts', 'pct_counts_mt','CC_diff'], **kargs):
        '''
        It is recommended to pass log-transformed matrices/AnnData objects to mnn_correct, 
        and use HVGs instead of all the genes.
        '''
        print('input must raw counts adata!')
        adata = self.adata.copy()
        nordata = Preprocession(adata).Normal()
        nordata = Preprocession(nordata).HVGs(batch_key=batch_key, minnbat=minnbat, dropMT=dropMT, 
                                              n_top_genes=n_top_genes, **kargs)

        if isnormal:
            alldata = TransAdata(nordata).splitAD(batch_key)
        else:
            alldata = TransAdata(adata).splitAD(batch_key)

        var_genes = nordata.var.highly_variable
        var_genes = var_genes.index[var_genes].tolist()

        cdata = sc.external.pp.mnn_correct(*alldata.values(), batch_categories= alldata.keys(),
                                            n_jobs=n_jobs, svd_dim = NPCS, batch_key = batch_key,
                                            save_raw = False, var_subset = var_genes, **kargs)
        corr_data = cdata[0]
        corr_data.var = nordata.var.loc[corr_data.var.index,]
        corr_data = corr_data[:,var_genes]
        #corr_data = Preprocession(corr_data).Scale(n_jobs=s_jobs, usehvgs=True, vargress=vargress) if isscale else adata
        return corr_data

class TransAdata:
    def __init__(self, adata, *args, **kargs):
        self.adata = adata
        self.args  = args
        self.kargs = kargs

    def splitAD(self, groupby, isorder=True):
        # split per batch into new objects.
        adata = self.adata.copy()
        try:
            batches = adata.obs[groupby].cat.categories.tolist()
        except:
            adata.obs[groupby] = adata.obs[groupby].astype('category')
            batches = adata.obs[groupby].drop_duplicates(keep='first').tolist()
        if isorder:
            import collections
            adatadict = collections.OrderedDict()
        else:
            adatadict = {}
        for batch in batches:
            adatadict[batch] = adata[adata.obs[groupby] == batch,]
        return adatadict

    def mergeAD(self, order=None, crossvar=False, convar=False, label="mergeid", join="outer", **kargs):
        import anndata as ad
        adatadict = self.adata.copy()
        if order is None:
            order = list(adatadict.keys())
        adata = ad.concat(adatadict, label=label, join=join, **kargs)
        adata.var.index.name = 'Gene'

        if crossvar:
            advar = pd.concat([adatadict[i].var for i in order], axis=1)
            advar.columns = [ '%s_%s'%(j,i) for i in order for j in adatadict[i].var.columns]
            adata.var = advar.loc[adata.var.index, ]      
            print('Please fillna in var or set crossvar as False if write h5ad!')
        if convar:
            advar = pd.concat([v.var for k,v in adatadict.items()], axis=0).drop_duplicates(keep='first')
            adata.var = adata.var.join(advar,  how='left')
        #adata.obs = adata.obs.infer_objects()
        #adata.var = adata.var.infer_objects()
        return adata

class Statistics:
    def __init__(self, *args, **kargs):
        self.args  = args
        self.kargs = kargs
    def Partition(self, adata, groupby='sampleid', features=['phase', 'doubletP'], isratio=True):
        Obs = adata[[groupby]+features]
        GG = Obs.groupby(groupby)
        for i in features:
            KK = GG[i].value_counts()
            if isratio:
                KK /= GG[i].size()

class PlotUti:
    def __init__(self, *args, **kargs):
        self.args  = args
        self.kargs = kargs

    def colorset(self, get='vega_20_scanpy'):
        #https://github.com/theislab/scanpy/commit/58fae77cc15893503c0c34ce0295dd6f67af2bd7
        #https://github.com/theislab/scanpy/issues/387
        from matplotlib import cm, colors
        vega_20 = [
            '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728',
            '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2',
            '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
        ]
        '''
        Add = [ #0780cf - 765005 - fa6d1d - 0e2c82 - b6b51f - da1f18 - 701866 - f47a75 - 009db2 - 024b51 - 0780cf - 765005
                #6beffc
                #3b45dd
                #b240ce]
        '''
        vega_20 = list(map(colors.to_hex, cm.tab20.colors))

        vega_20_scanpy = [ 
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
            '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31']
        #*vega_20[0:14:2], *vega_20[16::2],
        #*vega_20[1:15:2], *vega_20[17::2],
        return(eval(get))

    def _sccolor(self, adata, groupby):
        '''
        plt.figure(figsize=(8, 2))
        sc.pl._tools.scatterplots._get_palette(adata, 'SID').values()
        for i in range(28):
            plt.scatter(i, 1, c=sc.pl.palettes.default_20[i], s=200)
            plt.scatter(i, 1, c=sc.pl.palettes.default_102[i], s=200)
            sc.pl._tools.scatterplots._get_palette(adata, 'SID').values()
        plt.show()
        '''
        return sc.pl._tools.scatterplots._get_palette(adata, groupby)

    def rgb_to_hex(self, rgb):
        return '%02x%02x%02x' % rgb

    def hex_revers(self, value):
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        rgb =  tuple(255- int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return ('#%02x%02x%02x' % rgb)
    
    def hex_to_rgb(self, value):
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_dec(self, value):
        return [v/256 for v in value]
        
    def continuous_cmap_html(self, hex_list, float_list=None):
        ''' 
        creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map

        >>>hex_list = ['#0091ad', '#3fcdda', '#83f9f8', '#d6f6eb', '#fdf1d2', '#f8eaad', '#faaaae', '#ff57bb']
        >>>float_list = [0, 0.05, 0.5, 0.6, 0.85, 0.9, 0.92, 1]
        >>>cmap=continuous_cmap_html(hex_list,float_list=float_list )
        '''
        def rgb_to_hex(rgb):
            return '%02x%02x%02x' % rgb

        def hex_to_rgb(value):
            value = value.strip("#") # removes hash symbol if present
            lv = len(value)
            return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

        def rgb_to_dec(value):
            return [v/256 for v in value]

        rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
        if float_list:
            pass
        else:
            float_list = list(np.linspace(0,1,len(rgb_list)))
            
        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    def continuous_cmap_name(self, names='viridis', diversity=99, alpha=1, trans=np.log2, shift=1):
        import matplotlib
        import matplotlib.pyplot as plt 
        Nspace = list(np.linspace(0, 1,diversity))
        Lspace = list(trans(np.linspace(0,1,diversity) + shift))
        Carray = plt.get_cmap(names)(Lspace)
        return matplotlib.colors.ListedColormap(Carray)

    def continuous_cmap(self, cor_list):
        '''
        >>>continuous_cmap(["lightgrey", "blue", "mediumblue",'red','yellow'])
        '''
        return matplotlib.colors.LinearSegmentedColormap.from_list("my_cmp", cor_list)
    
    def ListedColormap(self, cor_list):
        return matplotlib.colors.ListedColormap(cor_list)
    
    def cmapsplit(self, colormap='viridis_r'):
        cmap=self.continuous_cmap(['white', 'mistyrose','purple','darkblue'])
        cmapsub = matplotlib.cm.get_cmap(colormap)([0.4, 0.7, 1])
        cmapsub = np.r_[ [[1,1,1,1], [1,1,0.6,1]], cmapsub]
        cmap=self.continuous_cmap(cmapsub)

    def vartype(self, pdseries):
        if pdseries.dtype in ['float32', 'float64', 'float', 'int32', 'int64', 'int']:
            return 'continuous'
        elif pdseries.dtype in ['category', 'object', 'bool']:
            return 'discrete'
            
    def pxsetcolor(self, color, ctype='discrete'):
        import plotly.express as px
        if color is None:
            return {}
        elif ctype=='discrete':
            if type(color)==str and color in dir(px.colors.qualitative):
                #px.colors.named_colorscales()
                COLOR = eval('px.colors.qualitative.%s'%color)
            elif type(color)==np.ndarray:
                COLOR = color.tolist()
            elif type(color)==list:
                COLOR = color
            elif type(color)==dict:
                COLOR = color.values()
            else:
                COLOR = color
            return {'color_discrete_sequence': COLOR}
        elif ctype=='continuous':
            if color in dir(px.colors.sequential):
                COLOR = eval('px.colors.sequential.%s'%color)
                return {'color_continuous_scale': COLOR}
            else:
                #[(0,"lightgrey"), (0.33,'yellow'), (0.67,'red'), (1,'darkred')]
                return {'color_continuous_scale': color}
        else:
            return {}

    def colrows(self, ncell, nrows=None, ncols=None, soft=False):
        import math
        if (ncols is None) and (nrows is None):
            ncols = 4
        if not ncols is None:
            nrows = math.ceil(ncell/ncols)
            ncols = min(ncell, ncols)
        elif not nrows is None:
            ncols = math.ceil(ncell/nrows)
            nrows = min(ncell, nrows)
        if soft and ncell> 1 and (ncell - ncols*(nrows-1)<=1):
            ncols += 1
            nrows -= 1
        return (nrows, ncols)

    def extend_colormap(self, value, under='#EEEEEE', over=-0.2, bits=256, color_name='my_cmp'):
        import matplotlib
        med_col = value
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        under = 'white' if under is None else under
        rgb = [ int(value[i:i + lv // 3], 16)/bits for i in range(0, lv, lv // 3)]

        if (over is None) or ( type(over) is int and over==0):
            return(matplotlib.colors.LinearSegmentedColormap.from_list(color_name, [under, med_col]))
        elif type(over) is int:
            rgb = [ i + over/bits for i in rgb ]
        elif type(over) is float and (0<= abs(over) <= 1) :
            rgb = [ i + over for i in rgb ]
        rgb = [ np.median([0, i, 1]) for i in rgb ]
        return(matplotlib.colors.LinearSegmentedColormap.from_list(color_name, [under, med_col, rgb]))

    def plot_color(self, cmap_dict, width=10, heigth_ratio=0.3, fontsize=9):

        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        cmaps = {}

        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        nrows = len(cmap_dict)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * heigth_ratio
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(width, figh))
        fig.subplots_adjust(top=1 - 0.35 / figh, 
                            bottom=0.15 / figh,
                            left=0.2, right=0.99)

        for ax, name in zip(axs, cmap_dict.keys()):
            imaps =  cmap_dict[name]
            if isinstance(imaps, str):
                icmap = plt.get_cmap(imaps)
            elif isinstance(imaps, list):
                icmap = matplotlib.colors.ListedColormap(imaps)
            else:
                icmap = imaps
            ax.imshow(gradient, aspect='auto', cmap=icmap)
            ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=fontsize,
                    transform=ax.transAxes)

        for ax in axs:
            ax.set_axis_off()
        cmaps.update(cmap_dict)  
        
class Plot:
    def __init__(self, adata=None, *args, **kargs):
        self.adata = adata
        self.args  = args
        self.kargs = kargs
        import matplotlib.pyplot as plt
        plt.rcParams['axes.grid'] = False

    def plots(self, color=None,
                    basis='X_umap', 
                    method=None,
                    legend_loc='right margin', #['right margin', None]
                    lloc='center left',
                    bbox_to_anchor=(1, 0, 0.5, 1), #(1, 0.5),
                    frameon=False,
                    ncols=5, scale=5, werror=0, size=None, 
                    markerscale=4, 
                    lncol=None, mode='expand',
                    shows=True, save=None, largs={}, **kargs):
        adata = self.adata.copy()
        adata.obsm[f'X_{basis}'] = adata.obsm[basis]

        if method is None:
            if basis in ['X_umap','umap']:
                method = 'umap'
            elif basis in ['X_tsne','tsne']:
                method = 'tsne'
            else:
                method = 'scatter'

        if color is None:
            color= adata.obs.columns[:1]
        elif isinstance(color, str):
            color = [color]

        if len(color) < ncols: ncols=len(color)
        nrows = np.ceil(len(color)/ncols)

        fig, axes = plt.subplots(int(nrows), int(ncols),
                                 figsize=(ncols*(scale+werror), nrows*scale))
        for n,i in enumerate(color):
            if len(color)==1:
                AX = axes
            elif min(ncols, nrows)==1:
                AX = axes[n]
            else:
                AX = axes[n//ncols,n%ncols]

            if method=='scatter':
                eval('sc.pl.%s'%method)(adata, basis=basis, color=i, show=False,
                                         size=size, title=i, legend_loc =legend_loc, ax=AX, **kargs)
            else:
                eval('sc.pl.%s'%method)(adata, color=i, na_in_legend =False, show=False,
                                        size=size, title=i, legend_loc =legend_loc, ax=AX, **kargs)

            handles, labels = AX.get_legend_handles_labels()
            if lncol is None:
                icol = max(1, int(np.ceil(len(labels)/15)))
            elif isinstance(lncol, int): 
                icol = lncol
            else:
                icol = lncol[n]
            AX.legend(handles, labels, 
                      ncol=icol,
                        loc=lloc, 
                        frameon=frameon,
                        mode=mode,
                        markerscale=markerscale,
                        bbox_to_anchor=bbox_to_anchor,
                        **largs)

        if nrows*ncols - len(color) >0:
            for j in range(nrows*ncols - len(color)):
                fig.delaxes(axes[-1][ -j-1])
        fig.tight_layout()
        if save:
            plt.savefig(save, bbox_inches='tight')
        if shows:
            plt.show()
        else:
            return fig, axes

    def splitplot(self, groupby='sampleid', splitby=None, basis='X_umap', method=None,
                  legend_loc='on data',
                  lloc="best", # lloc="center left",
                  ncols=5, scale=5, werror=0, size=None, markerscale=4,
                  bbox_to_anchor=None,# bbox_to_anchor=(1, 0, 0.5, 1),
                  lncol=1, mode=None,
                  bg_size=4, fontsize=10, bg_color='lightgray', 
                  shows=True, out=None,  **kargs):
        adata = self.adata.copy()
        adata.obsm[f'X_{basis}'] = adata.obsm[basis]
        if method is None:
            if basis in ['X_umap','umap']:
                method = 'umap'
            elif basis in ['X_tsne','tsne']:
                method = 'tsne'
            else:
                method = 'scatter'

        import math
        try:
            G = adata.obs[groupby].cat.categories
        except:
            G = adata.obs[groupby].unique()
        if len(G) < ncols: ncols=len(G)
        nrows = math.ceil(len(G)/ncols)

        if splitby:
            _data  = adata.obsm[basis]
            _sccor = sc.pl._tools.scatterplots._get_palette(adata, splitby)
            _datacor = adata.obs[splitby].map(_sccor)
            try:
                S = adata.obs[splitby].cat.categories
            except:
                S = adata.obs[splitby].unique()

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*(scale+werror), nrows*scale))
        for n,i in enumerate(G):
            if nrows==1:
                AX = axes[n]
            else:
                AX = axes[n//ncols,n%ncols]
            if splitby is None:
                if method=='scatter':
                    eval('sc.pl.%s'%method)(adata, basis=basis, color=groupby, groups =i, show=False,
                         size=size, title=i, legend_loc =legend_loc, ax=AX, **kargs)
                else:
                    eval('sc.pl.%s'%method)(adata, color=groupby, groups =i, na_in_legend =False, show=False,
                                            size=size, title=i, legend_loc =legend_loc, ax=AX, **kargs)
            else:
                _idx = adata.obs[groupby]==i
                if size is None:
                    size = 5
                AX.scatter( _data[:, 0][~_idx], _data[:, 1][~_idx], s=bg_size, marker=".", c=bg_color)
                _sid = [k for k in S if k in adata.obs.loc[_idx, splitby].unique()]
                if len(_sid)>0:
                    for _s in _sid:
                        _iidx = ((adata.obs[groupby]==i) & (adata.obs[splitby]==_s))
                        AX.scatter(_data[:, 0][_iidx], _data[:, 1][_iidx], s=size,  marker=".", c=_datacor[_iidx], label=_s)
                AX.set_title(i,size=fontsize)
                AX.set_ylabel((basis+'1').upper(), fontsize = fontsize)
                AX.set_xlabel((basis+'2').upper(), fontsize = fontsize)
                AX.set_yticks([])
                AX.set_xticks([])
                AX.grid(False)
                AX.legend(loc=lloc, bbox_to_anchor=bbox_to_anchor, 
                          mode=mode, ncol = lncol, markerscale=markerscale)

        if nrows*ncols - len(G) >0:
            for j in range(nrows*ncols - len(G)):
                fig.delaxes(axes[-1][ -j-1])
        fig.tight_layout()
        if out:
            plt.savefig(out)
        if shows:
            plt.show()
        else:
            return fig, axes

    @Chdir
    def QCCounts(self, adataraw, groupby='SID', show=True,  save='QC.cell.counts.',  **kargs):
        adata = self.adata.copy()
        QC = pd.concat([ adataraw.obs.groupby(groupby).size(), adataraw.obs.groupby(groupby)['n_genes_by_counts'].mean(),
                         adata.obs.groupby(groupby).size(), adata.obs.groupby(groupby)['n_genes_by_counts'].mean(),], axis=1)
        QC.columns = ['raw_cells', 'raw_meangene', 'keep_cells', 'keep_meangene']
        QC['drop_cells']=QC.raw_cells-QC.keep_cells
        QF = QC[['raw_cells','keep_cells','drop_cells']].copy()
        QF['keep_cells'] /= QF['raw_cells'] 
        QF['drop_cells'] /= QF['raw_cells']         
        
        dcolor =  plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes = plt.subplots(1, 3, figsize=(15, 6),constrained_layout=False)

        QC[['raw_cells','keep_cells','drop_cells']].plot.bar(rot=90, ax=axes[0], color=dcolor[0:3])
        QC[['keep_cells','drop_cells']].plot.bar(rot=90,stacked=True, ax=axes[1],color=dcolor[1:3],legend=False)
        QF[['keep_cells','drop_cells']].plot.bar(rot=90,stacked=True, ax=axes[2],color=dcolor[1:3],legend=False)

        plt.tight_layout()
        if show:
            plt.show()
        if save:
            plt.savefig(save+'pdf')
        plt.close()
        
        QC = QC.append( pd.Series(QC.sum(0), name='SUM'))
        if save:
            QC.to_csv(save+'xls', sep='\t')
        return QC

    @Chdir
    def QCplot(self, groupby='sampleid', header='before', ncols=3, lcol=2, pform='png',
               hlines = {}, hcolor='grey',
            clims= {'n_genes_by_counts':4000, 'total_counts':5000, 'pct_counts_mt':15, 'pct_counts_ribo':30, 'pct_counts_hb':30},
            COLs = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb', 'doubletS', 'CC_diff'], **kargs):
        adata = self.adata.copy()
        obsdata = adata.obs.sort_values('pct_counts_mt')
        COLs = [i for i in COLs if i in adata.obs.columns]

        fig, axes = plt.subplots(2, 4, figsize=(20, 8),constrained_layout=False)
        f1 = sns.histplot(data=obsdata, x="n_genes_by_counts", hue=groupby,  
                          multiple="stack", linewidth=0, ax=axes[0,0], legend=False)
        f2 = sns.histplot(data=obsdata, x="total_counts", hue=groupby, multiple="stack",linewidth=0,  ax=axes[0,1], legend=False)
        f3 = sns.scatterplot(data=obsdata, x="total_counts", y="n_genes_by_counts", hue=groupby,
                            ax=axes[0, 2],  s=1,linewidth=0, legend='full')
        f4 = sns.scatterplot(data=obsdata, x="total_counts", y="pct_counts_mt", hue=groupby,
                            ax=axes[0, 3],  s=1,linewidth=0, legend=False)
        f5 = sns.scatterplot(data=obsdata, x="total_counts", y="pct_counts_ribo", hue=groupby,
                            ax=axes[1, 0], s=1, linewidth=0, legend=False)
        f6 = sns.scatterplot(data=obsdata, x="pct_counts_mt", y="pct_counts_ribo", hue=groupby,
                            ax=axes[1, 1], s=1, linewidth=0,legend=False)

        f7 = axes[1, 2].scatter(obsdata['total_counts'], 
                                  obsdata['n_genes_by_counts'],
                                  cmap='cool',
                                  c=obsdata['pct_counts_mt'],
                                  vmin=0, vmax=clims['pct_counts_mt'],
                                  s=5,linewidth=0 ) 

        f8 = axes[1, 3].scatter(adata.obs.sort_values('pct_counts_ribo')['total_counts'], 
                                  adata.obs.sort_values('pct_counts_ribo')['n_genes_by_counts'],
                                  cmap='RdPu',
                                  c=adata.obs.sort_values('pct_counts_ribo')['pct_counts_ribo'],
                                  vmin=0, vmax=clims['pct_counts_mt'],
                                  s=5, linewidth=0)

        axes[1, 2].set_xlabel('total_counts')
        axes[1, 2].set_ylabel('n_genes_by_counts')
        cax7 = fig.add_axes([1.0, 0.33, 0.015, 0.21])
        fig.colorbar(f7, ax=axes[1, 3], label='pct_counts_mt', cax=cax7)
        axes[1, 3].set_xlabel('total_counts')
        axes[1, 3].set_ylabel('n_genes_by_counts')
        cax8 = fig.add_axes([1, 0.1, 0.015, 0.21])
        fig.colorbar(f8, ax=axes[1, 3], label='pct_counts_ribo', cax=cax8)

        for r in range(2):
            for c in range(4):
                axes[r, c].ticklabel_format( useOffset=False, style='sci', axis='both')
        axes[0,0].set_xlim([0, clims['n_genes_by_counts']])
        axes[0,1].set_xlim([0, clims['total_counts']])
        axes[0,2].set_xlim([0, clims['total_counts']])
        axes[0,2].set_ylim([0, clims['n_genes_by_counts']])
        axes[0,3].set_xlim([0, clims['total_counts']])
        axes[0,3].set_ylim([0, clims['pct_counts_mt']])
        axes[1,0].set_xlim([0, clims['total_counts']])
        axes[1,0].set_ylim([0, clims['pct_counts_ribo']])
        axes[1,1].set_xlim([0, clims['pct_counts_mt']])
        axes[1,1].set_ylim([0, clims['pct_counts_ribo']])
        axes[1,2].set_xlim([0, clims['total_counts']])
        axes[1,2].set_ylim([0, clims['n_genes_by_counts']])
        axes[1,3].set_xlim([0, clims['total_counts']])
        axes[1,3].set_ylim([0, clims['n_genes_by_counts']])

        lines, labels= f3.get_legend_handles_labels()
        axes[0,2].get_legend().remove()
        fig.legend(lines,
                    labels=labels[::-1],
                    bbox_to_anchor=(1.0, 0.56),
                    loc="lower left",
                    borderaxespad=0.,  
                    title=groupby,
                    ncol = lcol)
        fig.tight_layout()
        plt.savefig('density.count.%s.%s.%s'%(groupby, header, pform), bbox_inches='tight' )

        self.VioLin(groupby=groupby, COLs=COLs, 
                    ncols=ncols, hlines=hlines,hcolor=hcolor,
                    out='violin.count.%s.%s.%s'%(groupby,header, pform))
        self.genePlot(groupby = groupby, header=header, show=True, save=True, pform=pform)
        sc.pl.highest_expr_genes(adata, n_top=20 )

    def VioLin(self, groupby='sampleid', ncols=5, size=6, s=1, out=None, logy=True, ylim=None,
               ewidth=None, ehight=None, trans=False,linewidth=None,
               hlines = {}, hcolor='red',
               COLs = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo','doubletS'], **kargs):
        adata = self.adata.copy()
        try:
            G = adata.obs[groupby].cat.categories
        except:
            G = adata.obs[groupby].unique()
        ewidth = size/12*len(G) if ewidth is None else ewidth
        ehight = size if ehight is None else ehight

        if len(COLs) < ncols: ncols=len(COLs)
        nrows = int(np.ceil(len(COLs)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*ewidth, nrows*ehight))

        for n,i in enumerate(COLs):
            if len(COLs)==1:
                AX = axes
            elif (len(COLs) > 1) and (nrows==1):
                AX = axes[n]
            elif (len(COLs) > 1) and (ncols==1):
                AX = axes[n]
            else:
                AX = axes[n//ncols,n%ncols]
            if trans:
                ax = sns.violinplot(y=groupby, x=i, data=adata.obs, scale='width', cut=0, width=0.8, ax=AX, 
                                    linewidth=linewidth, **kargs)
                ax = sns.stripplot( y=groupby, x=i, data=adata.obs, size=1, edgecolor=None, 
                                    linewidth=0, jitter=0.2, zorder=1, alpha=0.8, ax=AX, **kargs)
                ax.set_title(i)
                if logy: ax.set_xscale("log")
                if not ylim is None:ax.set(xlim=ylim)
            else:
                ax = sns.violinplot(x=groupby, y=i, data=adata.obs, scale='width', cut=0, width=0.8, ax=AX, 
                                    linewidth=linewidth, **kargs)
                ax = sns.stripplot( x=groupby, y=i, data=adata.obs, size=s, edgecolor=None, 
                                    linewidth=0, jitter=0.2, zorder=1, alpha=0.8, ax=AX, **kargs)
                ax.set_title(i)
                if logy: ax.set_yscale("log")
                if not ylim is None:ax.set(ylim=ylim)
                ax.set_xticklabels(
                    ax.get_xticklabels(), 
                    rotation=90, 
                    ha='center',
                    va='center_baseline',
                    fontsize=10,
                )
            if i in hlines:
                for hl in hlines[i]:
                    ax.axhline(y=hl, color=hcolor, linestyle="--")

        if nrows*ncols - len(COLs) >0:
            for j in range(nrows*ncols - len(COLs)):
                fig.delaxes(axes[-1][ -j-1])

        fig.tight_layout()
        if out:
            plt.savefig(out)
        else:
            return fig, axes
    

    def genePlot(self, use_res=None, groupby = 'SID', header='before', show=True, save=None, 
                 title='the statistics of gene counts and cells', ylim_cell=None, pform='png',
                 yim_counts=None, logy=True, **kargs):
        from scipy.sparse import issparse
        adata = self.adata.copy()
        gene_counts = []
        use_res = 'X' if use_res is None else use_res
        if use_res =='X':
            adataX = adata.X
        elif use_res=='raw':
            adataX = adata.raw.X
        elif use_res in adata.layers.keys():
            adataX =  adata.layers[use_res]
        elif use_res =='shared':
            Xs, Xu = adata.layers["spliced"], adata.layers["unspliced"]
            nonzeros = ((Xs > 0).multiply(Xu > 0) if issparse(Xs) else (Xs > 0) * (Xu > 0))
            adataX= ( nonzeros.multiply(Xs) + nonzeros.multiply(Xu)
                        if issparse(nonzeros)
                        else nonzeros * (Xs + Xu))
        else:
            raise ValueError('`use_res` needs to be one of in "X, None, raw, shared, spliced, unspliced "')
        for k in adata.obs[groupby].cat.categories:
            iadata = adataX[np.flatnonzero(adata.obs[groupby]==k),]
            icounts= np.vstack([(iadata>0).sum(0), iadata.sum(0),
                        [k] * iadata.shape[1], adata.var.index ])
            gene_counts.append(icounts.T)
        gene_counts = pd.DataFrame(np.vstack(gene_counts), 
                                   columns=['n_cells_by_gene', 'n_counts_by_gene', groupby, 'gene'])
        gene_counts = gene_counts.infer_objects()
        gene_counts_sum=gene_counts.drop(groupby, axis=1).groupby('gene').sum(1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10),constrained_layout=False)
        fig.suptitle(f'{title} ({header} {use_res})', fontsize=12)
        ax0 = sns.histplot(data=gene_counts_sum, x="n_cells_by_gene", binwidth=5, binrange=[1,500],
                                multiple="stack", linewidth=0.1, ax=axes[0,0], legend=True,  **kargs)
        ax0.set_title('n_cells_by_gene_all')
        ax1 = sns.histplot(data=gene_counts, x="n_cells_by_gene", hue=groupby, binwidth=5, binrange=[1,500],
                                multiple="stack", linewidth=0.1, ax=axes[0,1], legend=True,  **kargs)
        ax1.set_title('n_cells_by_gene_each')
        ax2 = sns.violinplot(x=groupby, y='n_cells_by_gene', data=gene_counts, scale='width', cut=0, 
                             width=0.8, ax=axes[0,2])
        ax2 = sns.stripplot( x=groupby, y='n_cells_by_gene', data=gene_counts, size=1, edgecolor=None, 
                            linewidth=0, jitter=0.2, zorder=1, alpha=0.8, ax=axes[0,2])
        ax2.set_title('n_cells_by_gene')
        if logy: ax2.set_yscale("log")
        if not ylim_cell is None:ax2.set(ylim=ylim_cell)
        ax2.set_xticklabels(
            ax2.get_xticklabels(), 
            rotation=90, 
            ha='center',
            va='center_baseline',
            fontsize=10,
        )
        ax3 = sns.histplot(data=gene_counts_sum, x="n_counts_by_gene", bins=100, binrange=[1,1000],
                                multiple="stack", linewidth=0.1, ax=axes[1,0], legend=True,  **kargs)
        ax3.set_title('n_counts_by_gene_all')
        ax4 = sns.histplot(data=gene_counts, x="n_counts_by_gene", hue=groupby, bins=100, binrange=[1,1000],
                                multiple="stack", linewidth=0.1, ax=axes[1,1], legend=True,  **kargs)
        ax4.set_title('n_counts_by_gene_each')
        ax5 = sns.violinplot(x=groupby, y='n_counts_by_gene', data=gene_counts, scale='width',
                             cut=0, width=0.8, ax=axes[1,2])
        ax5 = sns.stripplot( x=groupby, y='n_counts_by_gene', data=gene_counts, size=1, edgecolor=None, 
                            linewidth=0, jitter=0.2, zorder=1, alpha=0.8, ax=axes[1,2])
        ax5.set_title('n_counts_by_gene')
        if logy: ax5.set_yscale("log")
        if not yim_counts is None:ax5.set(ylim=yim_counts)
        ax5.set_xticklabels(
            ax5.get_xticklabels(), 
            rotation=90, 
            ha='center',
            va='center_baseline',
            fontsize=10,
        )
        fig.tight_layout()
        if save:
            plt.savefig(f'gene.density.count.{groupby}.{header}.{use_res}.{pform}', bbox_inches='tight' )
        if show:
            plt.show()
        plt.close()

    def QCmap(self, 
                QCs=['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb','phase'],
                clims= {'n_genes_by_counts':8000, 'total_counts':10000, 'pct_counts_mt':20, 'pct_counts_ribo':30, 'pct_counts_hb':30},
                wspace =0.3, ncols=3, **kargs):
        adata = self.adata.copy()
        QCs = [ i for i in QCs if i in adata.obs.columns]
        for iqc in QCs:
            if iqc in  clims.keys():
                max_v = clims[iqc]
                adata.obs.loc[(adata.obs[iqc]>= max_v), iqc] = max_v
        if QCs:
            sc.pl.umap(adata, wspace =wspace,  ncols=ncols, color=QCs, **kargs)

    def mltpie(self, counts, nrows=None, ncols=2, scale=7, **kargs):
        G = counts.columns.tolist()
        nrows, ncols = PlotUti().colrows(len(G), ncols=ncols, nrows=nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*scale, nrows*scale))
        for n,i in enumerate(G):
            if nrows==1:
                _ax = axes[n]
            else:
                _ax = axes[n//ncols,n%ncols]

            wedges, texts, autotexts = \
                _ax.pie(counts[i], #labels=counts.index.tolist(), labeldistance=1.2,
                        autopct='%1.1f%%', pctdistance=0.6, **kargs)
            _ax.set_title(i)
            '''
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            kw = dict(arrowprops=dict(arrowstyle="-"),#bbox_props
                       zorder=0, va="center")
            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                _ax.annotate(counts.index[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                            horizontalalignment=horizontalalignment, **kw)
            '''
            if n == len(G)-1:
                _ax.legend(wedges, counts.index.tolist(),
                  title="Distance",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.show()

    def pxpie(self, counts, ncols=2, scale=500, soft=False, 
              sort=True, out=None, header=None, show=False, **kargs):
        groups= [i for i in counts.columns if i not in ['Type', 'Colors']]
        ncell = len(groups)
        nrows, ncols = PlotUti().colrows(ncell, ncols=ncols, soft=soft)

        fig = make_subplots(rows=nrows, cols=ncols, 
                            horizontal_spacing=0.1/ncols,
                            vertical_spacing=0.1/nrows,
                            specs=[[{"type": "domain"}]*ncell])
        for n,g in enumerate(groups):
            irow, icol = n//ncols+1, n%ncols+1
            fig.add_trace(go.Pie(labels=counts['Type'], values=counts[g],
                                 title={'text':g, 'position':'bottom center'},
                                 marker={'colors':counts['Colors']},
                                 sort=sort,
                                 **kargs),
                          row=irow, col=icol)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=scale*nrows, width=scale*ncols,
                          legend_font_size=10,
                          legend=dict( itemsizing = 'trace'))
        if show:
            fig.show()
        if header or out :
            ohtml = '%s.%s.pie.html'%(header, '.'.join(groups)) if header else out
            fig.write_html(ohtml)
            fig.write_image(ohtml.replace('html','pdf'))    

class Nxplot:
    def __init__(self, adata=None, *args, **kargs):
        self.adata = adata
        self.args  = args
        self.kargs = kargs

    def pagamap(self, groupby=None, method='umap', isscatter=True, isnet=True,markon=True,
                node_size=0, legendms=4,legendfs=10,font_size=10, figsize=(8,5),ncol=1,
                edge_cmap='default', edge_width_scale=0.3, max_edge_width=30, show=True, 
                edge_alpha=0.9, save=None):
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib import rcParams
        adata = self.adata.copy()

        if edge_cmap=='default':
            import matplotlib.colors as mcolors
            colors=plt.cm.binary(np.linspace(0.9, 1, 128))
            edge_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        '''
        sc.tl.paga(adata, groups=clucol)
        sc.pl.paga_compare(adata, basis='umap', projection='2d', edges=True, threshold=0.05,
                        arrowsize=5, node_size_scale=1, edge_width_scale=0.1)
        '''

        #plt.rcParams["figure.figsize"] = figsize
        plt.figure(figsize=figsize)
        if not groupby:
            groupby   = adata.uns["paga"]["groups"]

        grouplist = adata.obs[groupby].cat.categories
        groupcolor= adata.uns[groupby+'_colors']

        adjacency_solid = pd.DataFrame(adata.uns['paga']['connectivities'].toarray(),columns=grouplist,index=grouplist)
        adj_tree   = pd.DataFrame(adata.uns["paga"]["connectivities_tree"].toarray(),columns=grouplist,index=grouplist)
        map_data  = adata.obsm['X_'+ method]
        pos = adata.uns["paga"]["pos"]

        nx_g_solid= nx.Graph(adjacency_solid)
        nx_g_tree = nx.Graph(adj_tree)

        POS={ g:tuple(pos[i,:]) for i,g in enumerate(grouplist)}

        _sccor = sc.pl._tools.scatterplots._get_palette(adata, groupby)
        map_cor = adata.obs[groupby].map(_sccor)

        widths = [x[-1]["weight"] for x in nx_g_solid.edges(data=True)]
        print('max width:', max(widths))
        if edge_width_scale is not None:
            edge_width_scale = edge_width_scale * 5 * rcParams["lines.linewidth"]
            widths = edge_width_scale * np.array(widths)
        if max_edge_width is not None:
            widths = np.clip(widths, None, max_edge_width)

        if isscatter:
            plt.grid(False)
            scatt = plt.scatter(map_data[:,0], map_data[:,1], s=0.5, alpha=0.6, c=map_cor)

        if isnet:
            edges = nx.draw_networkx_edges(
                        nx_g_solid,
                        POS,
                        width=widths,
                        edge_color=widths,
                        edge_cmap=edge_cmap,
                        style="-",
                        alpha=edge_alpha,
                    )
            node = nx.draw_networkx_nodes(nx_g_tree, POS, node_size=node_size, node_color='black')
            #node = nx.draw_networkx(nx_g_tree, POS, node_size=0, node_color=groupcolor)
            if markon:
                labe = nx.draw_networkx_labels(nx_g_tree,POS,font_size=font_size,font_color='black')
                
        if isscatter:
            legend_elements = [ Patch(facecolor=groupcolor[i], edgecolor=None,label=c) 
                                for i,c in enumerate(grouplist)]
            legend_elements1 = [ Line2D([0], [0], marker='o', color=groupcolor[i], lw=0,
                                    label=c, markerfacecolor=None, markersize=legendms)
                                for i,c in enumerate(grouplist)]

            plt.legend(handles=legend_elements1, ncol=ncol, prop={'size':legendfs}, loc='center left', 
                    title=groupby,
                    bbox_to_anchor=(1.0, 0.5))

        plt.tight_layout()
        if save:
            plt.savefig(save)
            #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

class GetMk:
    def __init__(self, adict, *bdict, split=','):
        self.A = [adict, *bdict]
        self.S = split
        self._B = []
        self._D = {}
        
    def _drop(self, L):
        return [i for n,i in enumerate(L) if L.index(i) == n]
    def _split(self, l):
        return({ k:v.strip(self.S).split(self.S) for k,v in l.items() })
    def UP(self, *bdict):
        self.B = [self._split(ia) for ia in self.A ]
        self.B = [*self.B, *bdict]
        for l in self.B:
            for k,v in l.items():
                if k in self._D:
                    self._D[k].extend(v)
                else:
                    self._D[k] = v
                self._D[k] = self._drop(self._D[k])
        return self

    @property
    def D(self):
        return self._D          
    @property
    def K(self):
        return self.D.keys()
    @property
    def V(self):
        return self._drop(sum(self.D.values(),[]))
    @property
    def Vs(self):
        return self.D.values()
    @property
    def VK(self):
        vk = {}
        for k,v in self.D.items():
            for iv in v:
                if iv in vk:
                    vk[iv].append(k)
                else:
                    vk[iv] = [k]
        return vk

class Ply():
    def __init__(self, adata, *args, **kargs):
        self.adata = adata.copy()

    def SData(self, basis='X_umap_3d', method='umap', pcs=None, groups=['sampleid', 'leiden'], **kargs):
        if pcs is not None:
            self.dims=len(pcs)
            self.pcs = pcs
        else:
            self.dims = self.adata.obsm[basis].shape[1]
            self.pcs  = range(self.dims)

        self.dimsname = [ method + str(i) for i in self.pcs]
        self.Data = pd.DataFrame(self.adata.obsm[basis][:,self.pcs], columns=self.dimsname )

        var_groups = self.adata.raw.var_names.intersection(groups)
        obs_groups = self.adata.obs.columns.intersection(groups)       
        self.groups = np.concatenate([var_groups, obs_groups])
  
        if len(obs_groups)>0:
            self.Data[obs_groups] = self.adata.obs[obs_groups].reset_index(drop=True)
        if len(var_groups)>0:
            self.Data[var_groups] = self.adata.raw[:,var_groups].X.toarray()
    
        obs_groups_d = [ i for i in obs_groups if PlotUti().vartype(self.Data[i])=='discrete']
        self.Colors = {}
        for i in obs_groups_d:
            try:
                iorders = self.adata.obs[i].cat.categories.tolist()
            except:
                iorders = self.adata.obs[i].unique().tolist()
            iuns_cor = f'{i}_colors'
            if (iuns_cor in self.adata.uns.keys()) and len(iorders)<= len(self.adata.uns[iuns_cor]):
                self.Colors[i]=self.adata.uns[iuns_cor]
            else:
                icolors = PlotUti()._sccolor(self.adata, i) 
                self.Colors[i] = [ icolors[k] for k in iorders ]
        return self

    def Scatter2ds(self, Data, groups=None, out=None, header=None, size=1.0, show=False, colormap='guess',
                   template='none', scene_aspectmode='data',
                   ncols=2, soft=False, scale=600, error=20, **kargs):

        groups = Data.groups if (groups is None) else groups
        groups = [ i for i in groups if PlotUti().vartype(Data.Data[i])=='discrete']
        ncell = len(groups)
        nrows, ncols = PlotUti().colrows(ncell, ncols=ncols, soft=soft)
        
        if len(groups)<1:
            raise ValueError('Only discrete groups will be ploted!!')
        dimdict= dict(zip(list('xy'), Data.dimsname))
        # order 
        category_orders = {i: Data.Data[i].cat.categories.tolist() for i in groups}
        category_orders = {'Type': np.hstack([Data.Data[i].cat.categories for i in groups]).tolist()}
        if colormap=='guess':
            colormap = Data.Colors
        color_discrete_sequence = np.hstack([colormap[i] for i in groups]).tolist()
        #plot
        SData = pd.melt(Data.Data, id_vars= Data.dimsname, value_vars=groups, var_name='groups', value_name='Type')
        fig = px.scatter(SData, color="Type", facet_col="groups", 
                         facet_col_wrap=ncols,
                         width=ncols*scale+error, height=nrows*scale,
                         color_discrete_sequence=color_discrete_sequence,
                         category_orders=category_orders,
                         **dimdict)
        fig.update_layout(legend=dict( itemsizing = 'constant', itemwidth=30+len(groups)),
                          scene_aspectmode=scene_aspectmode,
                          template=template,
                            scene=dict(
                                xaxis=dict(visible=True, showticklabels=True),
                                yaxis=dict(visible=True, showticklabels=True),
                                zaxis=dict(visible=True, showticklabels=True),
                            ),
                          plot_bgcolor='#FFFFFF',) #
                          #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                          #paper_bgcolor='#000000',
                          #plot_bgcolor='#000000'
                          #fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_traces(marker=dict(size=size,
                          line=dict(width=0,color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        if show:
            fig.show()
        if header or out:
            out= header + '.'+ '.'.join(groups) + '.2d.html' if header else out
            fig.write_html(out)
            
    def Scatter3d(self, Data, group, out=None, show=False, markersize=0.8, width=800, height=800,
                  scene_aspectmode='data', keep_all=True,
                  colormap='guess', order='guess', template='none', **kargs):
        if Data.dims <3:
            raise ValueError('The dims must be larger than 2!!')
        dimdict = dict(zip(list('xyz'), Data.dimsname))
        # ctype
        ctype   = PlotUti().vartype(Data.Data[group])
        if colormap == 'guess':
            if group in Data.Colors.keys():
                colormap = Data.Colors[group] 
            elif ctype=='continuous':
                colormap = 'Viridis'
        dimdict.update(PlotUti().pxsetcolor(colormap, ctype=ctype))
        # order
        if ctype == 'discrete':
            if order =='guess':
                try:
                    order = Data.Data[group].cat.categories.tolist()
                    if not keep_all:
                        Data.Data[group] = Data.Data[group].cat.remove_unused_categories()
                        keep_order = Data.Data[group].cat.categories.tolist()
                        colors = dimdict['color_discrete_sequence']
                        if type(colors)==list:
                            colors =[c for c,o in zip(colors, order) if o in keep_order]
                            dimdict['color_discrete_sequence'] = colors
                        order = keep_order
                        print(order)
                except:
                    order = Data.Data[group].unique().tolist()
            category_orders={group: order}
            dimdict.update({'category_orders': category_orders}) #'animation_frame': group

        elif ctype == 'continuous':
            if order =='guess' or  order == True:
                Data.Data = Data.Data.sort_values(by = group, ascending=True)
        fig = px.scatter_3d(Data.Data, color=group,  **dimdict) #width=width, height=height,
        fig.update_layout(legend=dict(itemsizing = 'constant'),
                          scene_aspectmode=scene_aspectmode,
                          template=template,
                            scene=dict(
                                xaxis=dict(visible=True, showticklabels=True),
                                yaxis=dict(visible=True, showticklabels=True),
                                zaxis=dict(visible=True, showticklabels=True),
                            ),
                          plot_bgcolor='#FFFFFF',) #
                          #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                          #paper_bgcolor='#000000',
                          #plot_or='#000000'
                          #fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_traces(marker=dict(size=markersize,
                          line=dict(width=0,color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        if show:
            fig.show()
        if out:
            fig.write_html(out)

    def Scatter3ds(self, Data, groups=None, header=None, **kargs):
        groups = Data.groups if (groups is None) else groups
        for i in groups:
            out = None if header is None else '%s.%s.3d.html'%(header, i)
            self.Scatter3d(Data, i, out=out, **kargs)

    def Scatter23ds(self, Data, groups=None, out=None, header=None, show=False, ncols=2, scale=800, scene_aspectmode='data',
                   error=10, size=1, legendwscape=0.1, order ='guess', soft=False, colormap='guess', **kargs):
        groups= Data.groups if (groups is None) else groups
        ncell = len(groups)
        nrows, ncols = PlotUti().colrows(ncell, ncols=ncols, soft=soft)
        if Data.dims==2:
            GoS = go.Scatter
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=groups)
        elif Data.dims==3:
            GoS = go.Scatter3d
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=groups, specs=[[{"type": "scene"}]*ncell])

        legendps = 1.02+legendwscape if 'discrete' in [PlotUti().vartype(Data.Data[i]) for i in groups] else 1.02
        for n,i in enumerate(groups):
            irow, icol = n//ncols+1, n%ncols+1
            iData = Data.Data
            ctype = PlotUti().vartype(iData[i])
            # plot
            if ctype == 'discrete':
                Colors = Data.Colors[i]  if colormap == 'guess' and i in Data.Colors.keys() else colormap[i]
                Colors = PlotUti().pxsetcolor(Colors, ctype=ctype)
                if order =='guess':
                    try:
                        orders = iData[i].cat.categories.tolist()
                    except:
                        orders = iData[i].unique().tolist()
                else:
                    orders = order #list
                cordict = dict(zip(orders, Colors['color_discrete_sequence']))
                for _n in orders:
                    iiData  = iData[iData[i]==_n]
                    dimdict = { i[0]: iiData[i[1]] for i in zip(list('xyz'), Data.dimsname) }
                    dimdict.update({'name':_n, 'legendgrouptitle':{'text':i, 'font': {'size':14}}, 'legendgroup' : str(n+1)})
                    #if Colors:
                    dimdict.update({'marker': dict(color=cordict[_n],
                                                   size=size, 
                                                   line=dict(width=0,color='DarkSlateGrey'))})
                    fig.append_trace(GoS(mode="markers", showlegend=True, **dimdict, **kargs), row=irow, col=icol)
            elif ctype == 'continuous':
                Colors = 'Viridis' if colormap == 'guess' else colormap[i]
                Colors = PlotUti().pxsetcolor(Colors, ctype=ctype)
                if order =='guess' or  order == True:
                    iData = iData.sort_values(by = i, ascending=True)
                dimdict = { i[0]: iData[i[1]] for i in zip(list('xyz'), Data.dimsname)}
                dimdict.update({'name': i, 'legendgroup' : str(n+1)})
                colorscale = Colors['color_continuous_scale'] if Colors else None
                dimdict.update({'marker': dict(colorscale=colorscale, 
                                               showscale=True,
                                               color=iData[i],
                                               size=size,
                                               line=dict(width=0,color='DarkSlateGrey'),
                                               colorbar=dict(thickness=5, title=i, len=1, x=legendps,y=0.5, outlinewidth=0))})
                fig.append_trace(GoS(mode="markers", showlegend=False, **dimdict, **kargs), row=irow, col=icol)
                legendps += legendwscape
        fig.update_layout(height=scale*nrows, width=scale*ncols,
                          scene_aspectmode=scene_aspectmode,
                          legend_tracegroupgap = 10,
                          legend_font_size=14,
                          legend=dict( itemsizing = 'constant', itemwidth=30+len(groups)))
        if show:
            fig.show()
        if header or out :
            out = '%s.%s.%sd.html'%(header, '.'.join(groups), Data.dims) if header else out
            fig.write_html(out)
            

class Scvplot:
    def __init__(self):
        pass
    def compute_velocity_on_grid1(
        self,
        X_emb,
        V_emb,
        density=None,
        smooth=None,
        n_neighbors=None,
        min_mass=None,
        autoscale=True,
        adjust_for_stream=False,
        cutoff_perc=None,
    ):
        from sklearn.neighbors import NearestNeighbors
        from scipy.stats import norm as normal

        # remove invalid cells
        idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
        X_emb = X_emb[idx_valid]
        V_emb = V_emb[idx_valid]

        # prepare grid
        n_obs, n_dim = X_emb.shape
        density = 1 if density is None else density
        smooth = 0.5 if smooth is None else smooth

        grs = []
        for dim_i in range(n_dim):
            m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
            m = m - 0.01 * np.abs(M - m)
            M = M + 0.01 * np.abs(M - m)
            gr = np.linspace(m, M, int(50 * density))
            grs.append(gr)

        meshes_tuple = np.meshgrid(*grs)
        X_grid = np.vstack([i.flat for i in meshes_tuple]).T

        # estimate grid velocities
        if n_neighbors is None:
            n_neighbors = int(n_obs / 50)
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(X_emb)
        dists, neighs = nn.kneighbors(X_grid)

        scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
        weight = normal.pdf(x=dists, scale=scale)
        p_mass = weight.sum(1)

        V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
        V_grid /= np.maximum(1, p_mass)[:, None]
        if min_mass is None:
            min_mass = 1

        if adjust_for_stream:
            #X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
            X_grid = np.stack([ np.unique(X_grid[:, i]) for i in range(X_grid.shape[1]) ])
            ns = int(round(len(V_grid[:, 0])**(1/n_dim),0))
            #ns = int(np.sqrt(len(V_grid[:, 0])))
            V_grid = V_grid.T.reshape(n_dim, *[ns]*n_dim)

            mass = np.sqrt((V_grid ** 2).sum(0))
            min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
            min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
            cutoff = mass.reshape(V_grid[0].shape) < min_mass

            if cutoff_perc is None:
                cutoff_perc = 5

            length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
            length = length.reshape(*[ns]*n_dim)
            cutoff |= length < np.percentile(length, cutoff_perc)

            #V_grid[0][cutoff] = np.nan
        else:
            min_mass *= np.percentile(p_mass, 99) / 100
            X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

            if autoscale:
                V_grid /= 3 * quiver_autoscale(X_grid, V_grid)
        return X_grid, V_grid
    
    def scv3d(self, adata, basis='umap_3d', groupby='celltype_4', vkey='velocity', 
              mode='dynamical', density = 1, autoscale = False, show=False,  template='none',
              scene_aspectmode='data',
              smooth=0.5, n_neighbors=None, min_mass=None, adjust_for_stream=False, cutoff_perc=5):
        from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid
        from scvelo.tools.velocity_embedding import velocity_embedding

        import plotly.offline as iplot
        import plotly as py
        py.offline.init_notebook_mode(connected = True)

        import plotly.express as px #5.3.1
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        #plt.quiver
        #ipyvolume.quiver

        scv.tl.velocity_embedding(adata, basis=basis, vkey=vkey)
        X_emb = adata.obsm[f"X_{basis}"]
        V_emb = adata.obsm[f"{vkey}_{basis}"] if (f"{vkey}_{basis}" in adata.obsm.keys()) else adata.obsm[vkey]

        cvog = self.compute_velocity_on_grid1 if adjust_for_stream  else compute_velocity_on_grid
        X_grid, V_grid = cvog(
            X_emb=X_emb,
            V_emb=V_emb,
            density=density,
            autoscale=autoscale,
            smooth=smooth,
            n_neighbors=n_neighbors,
            min_mass=min_mass,
            adjust_for_stream=adjust_for_stream,
            cutoff_perc=cutoff_perc
        )
    
        colormap  = PlotUti()._sccolor(adata, groupby)
        colorlist = adata.obs[groupby].map(colormap)
        '''
        scat3d = go.Scatter3d(mode="markers", x=X_grid[:,0], y=X_grid[:,1], z=X_grid[:,2], 
                              marker= dict( size=0.7, color=colorlist, line=dict(width=0,color='DarkSlateGrey')),
                              showlegend=True)

        line3d = go.Scatter3d(mode="lines", x=X_grid[:,0], y=X_grid[:,1], z=X_grid[:,2], 
                              line= dict( width=1,color='rgb(102,135,231)'),
                              showlegend=True)

        gostream = go.Streamtube( x=X_grid[:,0], y=X_grid[:,1], z=X_grid[:,2], 
                                  u=V_grid[:,0], v=V_grid[:,1], w=V_grid[:,2],)
        '''
        gocone = go.Cone( x=X_grid[:,0], y=X_grid[:,1], z=X_grid[:,2], 
                          u=V_grid[:,0], v=V_grid[:,1], w=V_grid[:,2],
                          showlegend=True, showscale=False,
                          sizemode='scaled',sizeref=1, #sclor
                          name = 'velocity_grid',
                          colorscale=[(0, 'black'), (1, 'black')],
                        )
        gocones = []
        for n,i in enumerate(adata.obs[groupby].cat.categories.tolist()):
            idx = adata.obs[groupby]==i
            iX_emb = X_emb[idx,:]
            iV_emb = V_emb[idx,:]
            icone = go.Cone(  x=iX_emb[:,0], y=iX_emb[:,1], z=iX_emb[:,2], 
                              u=iV_emb[:,0], v=iV_emb[:,1], w=iV_emb[:,2],
                              sizemode='scaled',sizeref=6,
                              showlegend=True,showscale=False,
                              name = i,
                              colorscale=[(0, colormap[i]), (1,colormap[i])] )
            gocones.append(icone)

        camera = dict(up=dict(x=0, y=0, z=1),
                      center=dict(x=0, y=0, z=0),
                      eye=dict(x=-.75, y=-1.35, z=0.85))

        layout = go.Layout(scene=dict(xaxis=dict(title='Longitude'),
                                      yaxis=dict(title='Latitude'),
                                      zaxis=dict(title='Elevation'),
                                      camera=camera))

        fig = go.Figure(data=[gocone]+gocones) #, layout=layout

        fig.update_layout(legend=dict(itemsizing = 'constant'),
                          scene_aspectmode=scene_aspectmode,
                          template=template,
                            scene=dict(
                                xaxis=dict(visible=True, showticklabels=True),
                                yaxis=dict(visible=True, showticklabels=True),
                                zaxis=dict(visible=True, showticklabels=True),
                            ),
                          plot_bgcolor='#FFFFFF',) #
                          #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                          #paper_bgcolor='#000000',
                          #plot_bgcolor='#000000'
                          #fig.update_xaxes(visible=False, showticklabels=False)
        
        fig.update_layout( legend_font_size=14,
                           width =1300,
                           height=1000,
                           legend=dict( itemsizing = 'constant'))
        if show:
            fig.show()
        fig.write_html(f'scvelo_{groupby}.{mode}.cones.grid.3d.html')

        fig1 = go.Figure(data=gocones)
        fig1.update_layout(legend=dict(itemsizing = 'constant'),
                            scene_aspectmode=scene_aspectmode,
                            template=template,
                            scene=dict(
                                xaxis=dict(visible=True, showticklabels=True),
                                yaxis=dict(visible=True, showticklabels=True),
                                zaxis=dict(visible=True, showticklabels=True),
                            ),
                          plot_bgcolor='#FFFFFF',) #
                          #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                          #paper_bgcolor='#000000',
                          #plot_bgcolor='#000000'
                          #fig.update_xaxes(visible=False, showticklabels=False)
        fig1.update_layout( legend_font_size=14,
                           width =1300,
                           height=1000,
                           legend=dict( itemsizing = 'constant'))
        if show:
            fig1.show()
        fig1.write_html(f'scvelo_{groupby}.{mode}.cones.3d.html')


##### global variabel
my_cmap=PlotUti().continuous_cmap(["lightgrey", 'yellow', 'red','darkred'])
M2H = pd.read_csv('/share/home/zhonw/WorkSpace/01DataBase/Homology/HOM_MouseHumanSequence.rpt.txt', sep='\t')
M2Hdict = dict(zip(M2H['mouse'], M2H['human']))
H2Mdict = dict(zip(M2H['human'], M2H['mouse']))