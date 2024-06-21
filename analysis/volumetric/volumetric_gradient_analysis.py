import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

from brainspace.gradient import GradientMaps

from brainbuilder.interp.surfinterp import fill_in_missing_voxels


from utils import plot_explained_variance



def get_random_voxels(mask_file, n=10000):
    """Get random x, y, z voxel indices from mask."""
    mask = nib.load(mask_file)
    mask_vol = mask.get_fdata()

    vxl = np.where(mask_vol > 0)

    n_vxl = vxl[0].shape[0]

    if n_vxl < n:
        n = n_vxl

    idx = np.random.choice(n_vxl, n, replace=False)

    return (vxl[0][idx], vxl[1][idx], vxl[2][idx])

def get_voxel_receptor_values(receptor_volumes, vxl, output_dir):
    """Get receptor values for each voxel."""
    nvox = len(vxl[0])

    receptor_features = np.zeros((nvox, len(receptor_volumes)))

    for i, receptor in enumerate(receptor_volumes):
        receptor_vol = nib.load(receptor)
        receptor_data = receptor_vol.get_fdata()

        #z score receptor data
        receptor_data = (receptor_data - receptor_data.mean()) / receptor_data.std()

        receptor_features[:, i] = receptor_data[vxl]

    return receptor_features

def create_component_volumes(gm, vxl, mask_file, n, output_filenames, clobber=False):
    """Create nifti files for each gradient component."""
    mask = nib.load(mask_file)

    starts = mask.affine[0:3, 3]
    dimensions = mask.shape
    steps = mask.header.get_zooms()

    mask_vol = mask.get_fdata()


    sparse_vol = np.zeros(mask_vol.shape)

    if False in [os.path.exists(file) for file in output_filenames] or clobber :
        for grad_filename, component in zip(output_filenames, range(n)):
            grad_values = gm.gradients_[:, component]

            sparse_vol[vxl] = (grad_values-  grad_values.min()) + 1
            print('Component:', component)
            print(np.unique(grad_values))
            print(np.unique(mask_vol))
            out_vol = fill_in_missing_voxels(
                sparse_vol,
                mask_vol,
                starts[1],
                dimensions[1] * steps[1] + starts[1],
                starts[1],
                steps[1],
            )

        
            nib.Nifti1Image(out_vol, mask.affine).to_filename(
                grad_filename
            )
    return output_filenames


def volumetric_gradient_analysis(mask_file, receptor_volumes, output_dir, approach='pca', n=20000, clobber=False):
    """Perform volumetric gradient analysis on receptor data."""
    output_dir = f'{output_dir}/{approach}/'
    os.makedirs(output_dir, exist_ok=True)
    
    output_filenames = [  f'{output_dir}/macaque_gradient_{i}.nii.gz' for i in range(3)]
    run_grad_analysis =  False in [os.path.exists(file) for file in output_filenames]
    print('n=',n)
    if run_grad_analysis or clobber :
        vxl = get_random_voxels(mask_file, n=n)

        # Return a 2D array of n_voxels x n_receptors
        receptor_features = get_voxel_receptor_values(receptor_volumes, vxl, output_dir)

        # Calculate voxel-wise correlation between receptor features
        #corr = np.corrcoef(receptor_features)
        from scipy.stats import spearmanr
        corr = spearmanr(receptor_features, axis=1)[0]
        plt.cla(); plt.clf(); plt.close()
        plt.imshow(corr,cmap='nipy_spectral')
        plt.savefig(f'{output_dir}/correlation_matrix.png')

        # Calculate receptor gradients
        gm = GradientMaps(kernel=None,
                        n_components=receptor_features.shape[1], 
                        approach=approach)
        gm.fit(corr)

        n_final = plot_explained_variance(gm, output_dir)
        print('Number of components:', n_final)

        create_component_volumes(gm, vxl, mask_file, n_final, output_filenames)
    return output_filenames
