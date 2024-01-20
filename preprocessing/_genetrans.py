#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : _genetrans.py
* @Author  : Wei Zhou                                     *
* @Date    : 2023/08/02 16:28:48                          *
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

import os 
import pandas as pd
import numpy as np

class genetrans():
    def __init__(self):
        pass
    
    @classmethod
    def infor(cls, 
                    hsmmpair = None,
                    gene_info = None,
                    genegtf = { 'human':'/share/home/zhonw/WorkSpace/01DataBase/Genome/Human/Gencode/V44/gencode.v44.annotation.gtf.gz',
                                'mouse':'/share/home/zhonw/WorkSpace/01DataBase/Genome/Mouse/Gencode/V33/gencode.vM33.annotation.gtf.gz'},
                    keepcol=['gene_name','gene_id', 'gene_type'],
                    rpt_file='/share/home/zhonw/WorkSpace/01DataBase/Homology/HOM_MouseHumanSequence.rpt', 
                    save_mgi='auto',
                    save_bed='auto',
                    spacies = 'human',
                    url='http://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt',
                    ):
        if hsmmpair is None:
            try:
                M2H = pd.read_csv(f'{rpt_file}.txt', sep='\t')
            except:
                M2H = cls.h2m_mgi_rpt(rpt_file=rpt_file, save = save_mgi, url=url)
        elif os.path.exists(hsmmpair):
            M2H = pd.read_csv(hsmmpair, sep='\t')
        elif isinstance(adatasc.obs, pd.DataFrame):
            M2H = hsmmpair.copy()

        if gene_info is None:
            try:
                gtf =  genegtf[spacies]
            except:
                gtf = genegtf
            try:
                gene_df = pd.read_csv(f"{gtf.replace('.gz', '.bed')}", sep='\t')
            except:
                gene_df = cls.gene_bed(gtf, save=save_bed, keepcol=keepcol)
        elif os.path.exists(gene_info):
            gene_df = pd.read_csv(gene_info, sep='\t')
        elif isinstance(adatasc.obs, pd.DataFrame):
            gene_df = gene_info.copy()

        if not gene_df is None:
            cls.gene_type = dict(zip(gene_df['gene_name'], gene_df['gene_type']))
            cls.gene_df = gene_df

        if not M2H is None:
            cls.M2Hdict = dict(zip(M2H['mouse'], M2H['human']))
            cls.H2Mdict = dict(zip(M2H['human'], M2H['mouse']))
            cls.M2H = M2H
        return cls

    @staticmethod
    def gene_bed(hg38_gtf, keepcol=['gene_name','gene_id', 'gene_type'], save=None):
        if hg38_gtf.endswith('.gz'):
            import gzip
            f =  gzip.open(hg38_gtf,'rt')
        else:
            f=open(hg38_gtf, 'r')
        genes = []
        for l in f.readlines():
            if l.startswith('#'):
                continue
            ll= l.strip(';|\n| ').split('\t')
            if ll[2] !='gene':
                continue
            gs = { i.split(' ')[0]:i.split(' ')[1] for i in ll[8].split('; ') }
            kgs= [ll[0], ll[3], ll[4]] + [gs[i].strip('"') for i in keepcol]
            genes.append(kgs)
        f.close()
        genes = pd.DataFrame(genes, columns=['chr','start', 'end'] + keepcol )
        genes[['start', 'end']] = genes[['start', 'end']].astype(int)
        #genes['gene_name'] = genes['gene_name'].str.strip('"')
        if not save is None:
            if save =='auto':
                genes.to_csv(f"{hg38_gtf.replace('.gz', '.bed')}", sep='\t', index=False)
            else:
                genes.to_csv(save, sep='\t', index=False)
        return genes

    @staticmethod
    def h2m_mgi_rpt(rpt_file=None, save=None,
                    url='http://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt'):
        print(rpt_file, save, url)
        hsmt = ['MT-ND1', 'MT-ND2', 'MT-CO1', 'MT-CO2', 'MT-ATP8', 'MT-ATP6', 'MT-CO3', 'MT-ND3', 
                'MT-ND4L', 'MT-ND4', 'MT-ND5', 'MT-ND6', 'MT-CYB']
        mmmt = ['mt-Nd1', 'mt-Nd2', 'mt-Co1', 'mt-Co2', 'mt-Atp8', 'mt-Atp6', 'mt-Co3', 'mt-Nd3', 
                'mt-Nd4l', 'mt-Nd4', 'mt-Nd5', 'mt-Nd6', 'mt-Cytb']
        mmhs = [{'DBkey': '', 'mouse': mmmt[i], 'mouseEGID': np.nan, 'human': k, 'humanEGID': np.nan}
                    for i,k in enumerate(hsmt) ]
        if rpt_file is None:
            HOM_Mouse = pd.read_csv(url, sep='\t')
        else:
            HOM_Mouse = pd.read_csv(rpt_file, sep='\t')

        HOM_MouseHumanS = dict()
        for i, irow in HOM_Mouse.iterrows():
            DBkey = irow['DB Class Key']
            if not DBkey in HOM_MouseHumanS:
                HOM_MouseHumanS[DBkey] = {'DBkey':DBkey}
            species = irow['Common Organism Name'].replace(', laboratory', '')
            HOM_MouseHumanS[DBkey][species]=irow['Symbol']
            HOM_MouseHumanS[DBkey][species+'EGID']=irow['EntrezGene ID']
        HOM_MouseHumanS = pd.concat([ pd.Series(v) for v in list(HOM_MouseHumanS.values()) + mmhs], axis=1).T

        if ((isinstance(save, bool) and save) or (isinstance(save, str) and save=='auto')) and (not rpt_file is None):
            HOM_MouseHumanS.to_csv(f'{rpt_file}.txt', sep='\t', index=False)
        elif isinstance(save, str):
            HOM_MouseHumanS.to_csv(save, sep='\t', index=False)
        return(HOM_MouseHumanS)