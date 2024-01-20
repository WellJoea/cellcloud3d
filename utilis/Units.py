def flattenlist(stacklist):
    if not isinstance(stacklist, list):
        return(stacklist)
    flat = []
    def _flatt(stacklist):
        for i in stacklist:
            if isinstance(i, list):
                _flatt(i)
            else:
                flat.append(i)
    _flatt(stacklist)
    return(flat)

#def gene_annote(gtf, loci):
def gene_bed(hg38_gtf, keepcol=['gene_name','gene_id', 'gene_type']):
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
    return genes

def mapgene(irow, gene_bed):
    Chr   = irow['Chr']
    Start = irow['Start']
    igene = gene_bed[( (gene_bed['chr']== Chr) & (gene_bed['start']<= Start) & (gene_bed['end']>=Start) )]
    return igene['gene_name'].str.cat(sep=';') if igene.shape[0] >0 else '.'


def uniq_feature(adataR, 
    hsmmgene ='/gpfs2/PUBLIC/DataBase/Homology/HOM_MouseHumanSequence.rpt.txt'):
    M2H = pd.read_csv(hsmmgene, sep='\t')
    M2H = M2H[((~M2H.mouse.isna()) & (~M2H.human.isna()))]
    M2Hdict = dict(zip(M2H['mouse'], M2H['human']))

    Features = pd.DataFrame({'total': np.array(adataR.X.sum(0)).flatten(),
                             'means': np.array(adataR.X.mean(0)).flatten(),
                             'mouse': adataR.var_names})

    Features  = Features.merge(M2H, on='mouse', how='inner')
    kFeatures = Features.sort_values(by=['total','means'], ascending=[False, False])\
                        .groupby(by='human').head(1)
    return(dict(zip(kFeatures.mouse, kFeatures.human)))

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

    if save is None:
        return(HOM_MouseHumanS)
    elif ((isinstance(save, bool) and save) or (isinstance(save, str) and save=='True')) and (not rpt_file is None):
        HOM_MouseHumanS.to_csv(f'{rpt_file}.txt', sep='\t', index=False)
    elif isinstance(save, str):
        HOM_MouseHumanS.to_csv(save, sep='\t', index=False)