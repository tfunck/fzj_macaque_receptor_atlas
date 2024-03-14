import numpy as np
import nibabel as nib

from utils import plot_explained_variance
import brainbuilder.utils.mesh_utils as mesh_utils

from scipy.stats import spearmanr

from brainspace.gradient import GradientMaps

from matplotlib_surface_plotting import plot_surf
import matplotlib.pyplot as plt 

def get_roi_receptor_values(receptor_surfaces, atlas):
    roi_list = np.unique(atlas)[1:]

    n_receptors = len(receptor_surfaces)
    n_roi = len(roi_list)

    receptor_features = np.zeros((n_receptors, n_roi))

    for i, receptor_filename in enumerate(receptor_surfaces):
        receptor_data = np.load(receptor_filename)

        for j, label in enumerate(roi_list) : 
            idx = atlas == label
            # mean receptor value in each ROI
            receptor_features[i, j] = np.mean(receptor_data[idx])

    return receptor_features


def plot_gradient_surface(cortical_surface, grad, output_dir, component, cmap_label='Gradient', prefix='', vmin=None, vmax=None):
    # Plot first gradient on the cortical surface.
    coords_l, faces_l = mesh_utils.load_mesh_ext(cortical_surface)
    coords_r, faces_r = mesh_utils.load_mesh_ext(cortical_surface)
    
    if vmin is None:
        vmin = np.min(grad[~np.isnan(grad)])
    if vmax is None :
        vmax = np.max(grad[~np.isnan(grad)])

    filename = f'{output_dir}/{prefix}{component}.png'

    pvals = np.ones_like(grad)

    plot_surf( coords_l, 
                faces_l, 
                grad, 
                rotate=[90, 270], 
                filename=filename,
                vmax = vmax, 
                vmin = vmin, 
                pvals=pvals,
                cmap='nipy_spectral',
                cmap_label=cmap_label) 
    plt.close()
    plt.cla()
    plt.clf()



def roi_gradients(receptor_surfaces, surface_atlas, cortex_surface,  output_dir):

    atlas = nib.load(surface_atlas).darrays[0].data

    receptor_features = get_roi_receptor_values(receptor_surfaces, atlas)

    corr = spearmanr(receptor_features)[0]

    gm = GradientMaps(kernel=None,
                      n_components=receptor_features.shape[1], 
                      approach='pca')

    gm.fit(corr)

    plot_explained_variance(gm, output_dir)

    for i in range(4) :
        grad = gm.gradients_[:, i]
        vtr = np.zeros(atlas.shape[0])
        for j, label in enumerate(np.unique(atlas)[1:]) :
            idx = atlas == label
            vtr[idx] = grad[j]

        plot_gradient_surface(cortex_surface, vtr, output_dir, i, prefix='roi_gradient_')
    
def vertex_gradients(receptor_surfaces, cortex_surface, output_dir):

    receptor_features = np.array([ np.mean(np.load(receptor_surface), axis=1) for receptor_surface in receptor_surfaces])


    corr = spearmanr(receptor_features)[0]

    gm = GradientMaps(
                        kernel='spearman',
                        n_components=receptor_features.shape[0], 
                        approach='pca')


    gm.fit(receptor_features.T)

    plot_explained_variance(gm, output_dir)

    for i in range(4) :
        grad = gm.gradients_[:, i]
        plot_gradient_surface(cortex_surface, grad, output_dir, i, prefix='vertex_gradient_')

    return gm.gradients_





