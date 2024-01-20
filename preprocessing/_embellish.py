import numpy as np
import pandas as pd
import skimage as ski
from skimage.exposure import match_histograms

def img_similarity(img1, img2, match_hist=True, similar='nmi'):
    if match_hist:
        matched = match_histograms(img2, img1, channel_axis=None)
    else:
        matched = img2
    if similar=='ss':
        return ski.metrics.structural_similarity(img1, matched, data_range=1 - img1.min())
    elif similar=='nmi':
        return ski.metrics.normalized_mutual_information(img1, matched)
    else:
        return np.nan
        #ski.metrics.structural_similarity(img1, matched, data_range=matched.max() - matched.min())
        #ski.metrics.normalized_mutual_information(img1, img2)
        #ski.metrics.normalized_root_mse(img2, img1) ##nonsym
        #ski.metrics.variation_of_information(img1, img2),
        #ski.metrics.mean_squared_error(img1, img2),
        #ski.metrics.peak_signal_noise_ratio(img1, img2),
        #ski.metrics.adapted_rand_error(img1, img2)
        #ski.metrics.contingency_table(img1, img2)

def scaledimg(images):
    if (np.issubdtype(images.dtype, np.integer) or
        (images.dtype in [np.uint8, np.uint16, np.uint32, np.uint64])) and \
        (images.max() > 1):
        return False
    else:
        return True

def onechannels(image):
    if image.ndim==3:
        return True
    elif image.ndim==4:
        return False
    else:
        raise ValueError('the image must have 3 or 4_ dims.')
        
def step_similarity(images, isscale=None, back_for = [3,0], 
                    match_hist=False, similar='nmi',
                    nascore=1, plot=True):
    if isscale is None:
        isscale = onechannels(images)
    if not isscale:
        images  = ski.color.rgb2gray(images)
    # if images.ndim==4:
    #     images = images[...,0]

    similarity = []
    for i in range(images.shape[0]):
        wds = [*range(i-back_for[0], i) , *range(i, i+back_for[1]+1)]
        simi = []
        for iwd in wds:
            if (iwd < 0) or (iwd == i) or (iwd>=images.shape[0]):
                score = nascore
            else:
                score = img_similarity(images[i], images[iwd], match_hist=match_hist, similar=similar)
            simi.append(round(score, 8))
        similarity.append(simi)
    similarity = pd.DataFrame(similarity,
                              index=np.arange(len(similarity)) +1,
                              columns=list(range(-back_for[0], back_for[1]+1)))

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,1, figsize=(30,7))
        sns.heatmap(similarity.T, cmap = 'viridis', linewidth=.5, ax=ax[0])
        sns.lineplot(data=similarity, ax=ax[1])
        fig.show()

    return similarity