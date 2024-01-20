#import Cluster as cl
#from Config import Config as cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

def sci_cluster(Tree, 
                 figsize = None,
                 color_threshold=None, 
                 labels=None,
                 leaf_rotation=90,
                 leaf_font_size=8,
                 link_colors = list(map(mpl.colors.rgb2hex, plt.get_cmap('tab20').colors)),
                 save=None, **kargs):
    figsize = [ max(25, (Tree.shape[0]+1)/ 20), 8] if figsize is None else figsize
    plt.figure(figsize=figsize, facecolor='white')

    sch.set_link_color_palette(link_colors)
    dendro_info = sch.dendrogram(Tree,
                            leaf_rotation=leaf_rotation,
                            leaf_font_size=leaf_font_size,
                            labels = labels,
                            color_threshold=color_threshold, 
                            **kargs)
    if not color_threshold is None:
        plt.axhline(y=color_threshold, c='grey', lw=1, linestyle='dashed')
    plt.title('Sample clustering to detect outliers')
    plt.xlabel('Samples')
    plt.ylabel('Distances')
    plt.tight_layout()
    if not save is None:
        plt.savefig(save)

def soft_threshold(sft_df, figsize=(10, 5), R2Cut = None,
                    use_adjust = False,
                    save = None, fontsize=8, size =100):
    fig, ax = plt.subplots(ncols=2, figsize=figsize, facecolor='white')
    X_R2 = 'truncated R.sq' if use_adjust else 'SFT.R.sq'
    sft_df['SFT.R2'] = -1 * np.sign(sft_df['slope']) * sft_df[X_R2]
    if not R2Cut is None:
        ax[0].axhline(R2Cut, color='gray', linestyle='--', linewidth=3, zorder=1)
    ax[0].scatter(sft_df['Power'], sft_df['SFT.R2'], s =size, 
                    edgecolors = 'black', linewidths=1,
                    marker= 'o', color='lightgrey', zorder=1)
    for i,_l in sft_df.iterrows():
        ax[0].text(_l['Power'], _l['SFT.R2'], int(_l['Power']),
                    ha="center", va="center", color='red', 
                    fontsize=fontsize,
                    weight='normal')

    ax[0].set_xlabel("Soft Threshold (power)")
    ax[0].set_ylabel("Scale Free Topology Model Fit,signed R^2")
    ax[0].title.set_text('Scale independence')

 
    ax[1].scatter(sft_df['Power'], sft_df['mean(k)'], s =size, 
                    edgecolors = 'black', linewidths=1,
                    marker= 'o', color='lightgrey', zorder=1)
    for i,_l in sft_df.iterrows():
        ax[1].text(_l['Power'], _l['mean(k)'], int(_l['Power']),
                    ha="center", va="center", color='blue', 
                    fontsize=fontsize,
                    weight='normal')

    ax[1].set_xlabel("Soft Threshold (power)")
    ax[1].set_ylabel("Mean Connectivity")
    ax[1].title.set_text('Mean connectivity')

    fig.tight_layout()
    if not save is None:
        plt.savefig(save)

