import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(res, title="", show=True, save_file="fig.png"):
    keys = list(res.keys())
    ncol = 4
    nrow = 2
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*3., nrow*5.)) # width, height
    for i, j in np.ndindex(axs.shape):
        if i*ncol+j < len(keys):
            key = keys[i*ncol+j]
            t = np.arange(0., len(res[key]))
            axs[i, j].plot(t, res[key], linewidth=1)
            axs[i, j].set_title(key, fontsize=7)
            axs[i, j].xaxis.get_label().set_fontsize(7)
            axs[i, j].tick_params(labelsize=7)
            axs[i, j].grid()

    fig.suptitle(title, fontsize=7)
    plt.savefig(save_file, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
