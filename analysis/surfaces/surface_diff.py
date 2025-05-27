import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

from matplotlib import cm
from matplotlib_surface_plotting import plot_surf
from brainbuilder.utils import mesh_utils, mesh_io


def get_ngh(ngh, i, j, count):
    new_j = j 
    out_j = j
    for k in range(count):
        new_j = [ q for r in new_j for q in ngh[r] if q != i and q not in out_j]
        new_j = np.unique(new_j)
        j.append(new_j)
    out_j = np.unique(out_j)
    return out_j

def calc_surface_gradient(receptor_surfaces:list, coords:np.array, ngh:np.array):

    gradients = np.zeros(coords.shape[0])
    directions = np.zeros([coords.shape[0],3])

    for i in range(coords.shape[0]):
        j = ngh[i]
    
        #j = get_ngh(ngh, i, j, 2) 

        central_coords = coords[i]
        central_receptors = receptor_surfaces[:,i]
        if i == 0:
            print('receptors 1', central_receptors, np.mean(central_receptors))

        ngh_coords = coords[j]

        nngh = len(j)

        ngh_receptors = receptor_surfaces[:,j].reshape(nngh, receptor_surfaces.shape[0], receptor_surfaces.shape[2])

        # we have nngh 3D vectors. 
        # we then want to sum these vectors to get the gradient
        dcoords = (ngh_coords - central_coords)
        dcoords = np.sqrt(np.sum(np.power(dcoords,2),axis=1))[:,None]

        dreceptors = ngh_receptors - central_receptors
        dreceptors = np.sqrt(np.sum(np.power(dreceptors,2), axis=2)) 
        
        dreceptors /= dcoords
        grad = np.sqrt(np.sum(np.power(dreceptors,2) ) )

        if i == 0:
            print(dreceptors)
            print(grad)

        gradients[i] = grad 
    print('Num. Vertices:', coords.shape[0])
    return gradients, directions



def surface_diff(receptor_surfaces:list, cortex_filename:str, inflated_filename:str, output_dir:str, label:str=''):
    """Calculate the gradient of change across all receptors over the cortical surface."""
    os.makedirs(output_dir, exist_ok=True)
    receptor_features = np.array([np.load(receptor_surface) for receptor_surface in receptor_surfaces])
    # dimensions = n receptors, n vertices, n layers

    coords, faces = mesh_utils.load_mesh_ext(cortex_filename)
    coords_inflated, _ = mesh_utils.load_mesh_ext(inflated_filename)

    ngh = mesh_io.get_neighbours(faces)[0]

    diff_grad, _ = calc_surface_gradient(receptor_features, coords, ngh)

    vmin = np.percentile(diff_grad[~np.isnan(diff_grad)], [5])
    vmax = np.percentile(diff_grad[~np.isnan(diff_grad)], [95])

    filename = f'{output_dir}/differential_gradient_{label}.png'

    pvals = np.ones_like(diff_grad)
    """
    selection = np.random.choice(coords.shape[0],2000)
    arrow_colours = cmap(selection/np.max(selection))
    cmap=cm.get_cmap('magma')
    plot_surf(  coords, 
                faces, 
                diff_grad, 
                rotate=[90, 270], 
                filename=filename_arrows,
                vmax = vmax, 
                vmin = vmin,
                pvals=pvals,
                arrows = directions,
                arrow_subset = selection,
                arrow_colours=arrow_colours,
                arrow_size = 0.05,
                arrow_head=0.01,
                cmap='viridis',
                cmap_label='Differential gradient') 
    plt.close()
    plt.cla()
    plt.clf()
    """

    plot_surf(  coords, 
                faces, 
                diff_grad, 
                rotate=[90, 270], 
                filename=filename,
                vmax = vmax, 
                vmin = vmin,
                pvals = pvals,
                cmap = 'RdYlBu_r',
                cmap_label='Differential gradient') 
    plt.close()
    plt.cla()
    plt.clf()
