import pandas as pd
import numpy as np
import scanpy as sc
from joblib import Parallel, delayed

from cellcloud3d.preprocessing._genetrans import genetrans

class degsp:
    def __init__(self, adata, inplace=True,
                   spacies='human', 
                    hsmmpair = None,
                    gene_info = None,
                    genegtf = { 'human':'/share/home/zhonw/WorkSpace/01DataBase/Genome/Human/Gencode/V44/gencode.v44.annotation.gtf.gz',
                                'mouse':'/share/home/zhonw/WorkSpace/01DataBase/Genome/Mouse/Gencode/V33/gencode.vM33.annotation.gtf.gz'},
                    keepcol=['gene_name','gene_id', 'gene_type'],
                    rpt_file='/share/home/zhonw/WorkSpace/01DataBase/Homology/HOM_MouseHumanSequence.rpt', 
                    save_mgi='auto',
                    save_bed='auto',
                    url='http://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt',
                    *args, **kargs):
        self.adata = (adata.copy() if inplace else adata)
        self.args  = args
        self.kargs = kargs
        self.gt = genetrans.infor(hsmmpair=hsmmpair,
                                        spacies=spacies,
                                         gene_info=gene_info,
                                         genegtf=genegtf,
                                         keepcol = keepcol,
                                         rpt_file=rpt_file,
                                         save_mgi=save_mgi,
                                         save_bed=save_bed,
                                        )

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