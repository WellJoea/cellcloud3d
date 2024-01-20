import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

def mesh_color_hex( rgbs=(3,7,5), 
                    amin=0, amax=1, 
                    std_min = 0.05,
                    mean_max=0.9,
                    drop_color = ['grey', 'gray', '#808080']):
    rgbs = (3,5,7) if rgbs is None else rgbs
    if isinstance(rgbs, int):
        mgs = np.meshgrid( *[np.linspace(amin,amax, rgbs)] *3 )
    else:
        mgs = np.meshgrid(np.linspace(amin,amax,rgbs[0]), 
                          np.linspace(amin,amax,rgbs[1]), 
                          np.linspace(amin,amax,rgbs[2]))

    colors = np.vstack(map(np.ravel, mgs))
    colors = pd.DataFrame(colors.T, columns=list('rgb'))
    colors['hex'] = colors.apply(mcolors.to_hex, axis=1)
    colors['std'] = np.std(colors.iloc[:,:3], axis=1)
    colors['sch'] = pd.Categorical(colors.iloc[:,:3].idxmax(axis=1), categories=list('rbg'))
    colors['mean'] = np.mean(colors.iloc[:,:3], axis=1)
    colors.sort_values(by = ['std','mean','sch'], ascending=[False,True, True],inplace=True )
    colors = colors[((colors['std']>=std_min) & 
                     (colors['mean']<=mean_max) &
                     (~colors['hex'].isin(drop_color))
                    )]
    #PlotUti().ListedColormap(colors['hex'])
    return(colors['hex'].values)

def get_sigcss4(colors = None, 
                max_error=0.12, 
                std_min = 0.05,
                mean_max=0.9, 
                drop_color = ['grey', 'gray', '#808080'],
                verbose=False ):
    colors = mcolors.CSS4_COLORS if colors is None else colors
    css4 = {k: mcolors.to_rgb(v) for k,v in colors.items()}
    css4 = pd.DataFrame(css4, index=list('rgb')).T
    css4V = css4.round(1).sort_values(by=list('rgb'), ascending=[True]*3)
    css4 = css4.loc[css4V.index,:]
    i = 1
    while i < css4.shape[0]:
        if np.max(np.abs(css4.iloc[i,:3].values - css4.iloc[i-1,:3].values))<=max_error:
            verbose and print(f'drop {css4.index[i]}')
            css4.drop(css4.index[i], inplace=True)
        else:
            i +=1
    verbose and print(f'keep rows: {css4.shape[0]}')
    css4['std'] = np.std(css4.iloc[:,:3], axis=1)
    css4['sch'] = pd.Categorical(css4.iloc[:,:3].idxmax(axis=1), categories=list('rbg'))
    css4['mean'] = np.mean(css4.iloc[:,:3], axis=1)
    css4['hex'] = css4.index.map(colors)
    css4.sort_values(by = ['std','mean','sch'], ascending=[False,True, True],inplace=True )

    css4 = css4[((css4['std']>=std_min) & 
                 (css4['mean']<=mean_max) &
                 (~css4['hex'].isin(drop_color))
                )]
    verbose and print(f'final rows: {css4.shape[0]}')
    return(css4.index.values)

def labels2colors(labels, zeroIsGrey=True, colorSeq=None, naColor="grey"):
    #Vcols = labels.Value[~labels.Value.isna()].unique()
    labels = pd.Series(labels)
    Vcols = labels[~pd.isna(labels)].unique()
    ncols = Vcols.shape[0]
    grid = np.ceil(np.power(ncols,1/3))
    ccs4  = get_sigcss4(colors = colorSeq,
                        drop_color=[naColor, 'gray', '#808080'])
    mhex = mesh_color_hex(drop_color=[naColor, 'gray', '#808080'])
    if ncols<= len(ccs4):
        colors = ccs4
    elif ncols <= len(mhex):
        colors = mhex
    else:
        grid = np.ceil(np.power(ncols,1/3))
        colors = mesh_color_hex(rgbs=[grid, grid+1, grid], 
                                drop_color=[naColor, 'gray', '#808080'])

    Vmaps = dict(zip(Vcols, colors))
    Vmaps[None] = naColor
    return labels.map(Vmaps).values
