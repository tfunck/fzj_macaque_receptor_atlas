import matplotlib.pyplot as plt
import os
import numpy as np
import nibabel as nib
from kneed import KneeLocator



def plot_explained_variance(gm, output_dir, threshold=0.95):
    lambdas = gm.lambdas_




    
    x = np.arange(len(lambdas))
    kneedle = KneeLocator(x, lambdas, S=1.0, curve="convex", direction="decreasing")

    n = kneedle.elbow
    kneedle.plot_knee_normalized()
    plt.savefig(os.path.join(output_dir, f'explained_variance.png'))

    exp_var = lambdas[0:n] / lambdas.sum()

    print('Explained Variance:', exp_var)
    print('Sum of Explained Variance:', exp_var.sum())

    #exp_var= exp_var[exp_var > 0.05 ]
    
    plt.figure(figsize=(10, 5))
    plt.plot(exp_var, 'o-')
    plt.xlabel('Component')
    plt.ylabel('Explained variance')
    plt.savefig(os.path.join(output_dir, f'explained_variance.png'))

    return n

def get_files_from_list(files, target_strings):

    return [file for file in files if any(target in file for target in target_strings)]


def get_volume_from_list(input_list, target_string, zscore=True):
    files = get_files_from_list(input_list, [target_string])

    assert len(files) == 1, f'Expected 1 file, got {len(files)}'

    vol = nib.load(files[0]).get_fdata()

    if zscore:
        vol = (vol - vol.mean()) / vol.std()

    return vol


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

