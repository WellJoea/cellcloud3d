import pandas as pd

def dropGene(genes, dropMT=False, dropRibo=False, dropHb=False):
    genen = pd.Series(genes)
    mito = genen.str.contains("^MT-|^mt-|^Mt-", regex=True)
    ribo = genen.str.contains("^RP[SL]|^Rp[sl]", regex=True)
    hbs  = genen.str.contains("^HB[^(P)]|^Hb[^(p)]", regex=True)
    
    if dropMT:
        genen = genen[~mito]
    if dropRibo:
        genen = genen[~ribo]
    if dropHb:
        genen = genen[~hbs]

    print(f'drop featues: {len(genes)-len(genen)}.')
    return genen.values