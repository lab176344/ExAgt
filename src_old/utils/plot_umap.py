import umap
import matplotlib.pyplot as plt
import numpy as np

def plot_umap(data, label, epoch, num_unlabeled_classes, save_dir):
    U = umap.UMAP(n_components = 2,n_neighbors=20)
    print('Shape_Extracted', data.shape)
    embedding2 = U.fit_transform(data)
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(embedding2[:, 0], embedding2[:, 1], s= 5, c=label, cmap='Spectral')
    savename = save_dir + r'\UMAP_'+str(epoch)+'.png'
    cbar = plt.colorbar(boundaries=np.arange(num_unlabeled_classes+1)-0.5)
    cbar.set_ticks(np.arange(num_unlabeled_classes))
    #cbar.set_ticklabels(classes)
    plt.savefig(savename)
    plt.close()
    plt.clf()
    plt.cla()