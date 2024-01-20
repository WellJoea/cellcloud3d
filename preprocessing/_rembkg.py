import numpy as np
import matplotlib.pylab as plt

def image_color_hist(img, layout='rgb', figsize=(20,5), 
                   logyscale=True,
                   bin = None, iterval=None, show=True,):
    #from scipy.signal import find_peaks
    #find_peaks(counts, distance =10, width =3)
    peaks= []

    iimg = img.copy()
    if len(iimg.shape) ==2:
        iimg = iimg[:,:,np.newaxis]
    if np.round(iimg.max())<=1:
        iterval=(0, 1)
        bins=255
        xtick = np.round(np.linspace(0,1,bins+1, endpoint=True), 2)
    else:
        intmax = int(np.ceil(iimg.max()))
        iterval=(0, intmax)
        bins=255
        xtick = list(range(0,intmax,5))
    iimg = iimg[:,:,:]

    fig, ax = plt.subplots(1,1, figsize=figsize)
    for i in range(iimg.shape[2]):
        x = iimg[:,:,i].flatten()
        counts, values=np.histogram(x, bins=bins, range=iterval)
        max_value = values[np.argmax(counts)]
        peaks.append(max_value)
        xrange = np.array([values[:-1], values[1:]]).mean(0)
        ax.plot(xrange, counts, label=f"{i} {layout[i]} {max_value}", color=layout[i])
        ax.axvline(x=max_value, color=layout[i], linestyle='-.')

    ax.legend(loc="best")
    ax.set_xticks(xtick)
    ax.set_xticklabels(
        xtick,
        rotation=90, 
        ha='center',
        va='center_baseline',
        fontsize=10,
    )
    if logyscale:
        ax.set_yscale('log')
    #ax.set_axis_on()
    if show:
        fig.show()
    else:
        plt.close()
    return np.array(peaks)

def hardclipbkg(iimg,
            clips=None,
            error=None,
            bgcolor=(0, 0, 0),
            layout='rgb',
            show_peak = True,
            figsize_peak=(20,5),
            figsize=(10,20),
            show=True):
    if iimg.ndim == 2:
        chanel = 1
    elif (iimg.ndim == 3) and (iimg[0,0,:].shape[0]==3):
        chanel = 3
    else:
        raise ValueError('image must 2/3 dim.')

    fore_peak= image_color_hist(iimg.copy(), 
                              layout=layout, 
                              figsize=figsize_peak,
                              show=show_peak)

    if clips is None:
        clips = [fore_peak]
    clips = np.array(clips)
    assert clips.ndim==2 and clips.shape[1] == chanel, \
        'clip array must have 2 dims, and the second dim must be equal the number of image channels'

    if error is None:
        error = [[10/255.*iimg.max()]*chanel*2]
    error = np.array(error)

    assert error.ndim==2 and (error.shape[1] == (2*chanel)), \
        'error array must have 2 dims, and the second dim must be twice the number of image channels'

    img_np = iimg.copy()
    mask = np.zeros(iimg.shape[:2], dtype=bool)
    for iclip,ierr in zip(clips,error):
        bclip = iclip - ierr[:chanel]
        fclip = iclip + ierr[chanel:]

        bidx = np.all(img_np>=bclip, axis=-1)
        fidx = np.all(img_np<=fclip, axis=-1)
        mask = (mask | (bidx & fidx))
    img_np[mask,] = bgcolor[:chanel]

    img_rk = img_np.copy()
    img_rk[ img_rk != iimg ] = 0
    #img_rk[idx,] = 255
    #img_rk[idx,] = (100,200,255)
    if show:
        fig, axs = plt.subplots(1,2,figsize=figsize)
        axs[0].imshow(mask.astype(np.int64), cmap='gray')
        #axs[0].imshow(img_rk)
        axs[1].imshow(img_np)
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        fig.show()
    return img_np, mask