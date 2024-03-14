import matplotlib.pyplot as plt
import os
import numpy as np
import nibabel as nib


def plot_explained_variance(gm, output_dir):
    lambdas = gm.lambdas_

    exp_var = lambdas / lambdas.sum() 
    print('Explained Variance:', exp_var)
    plt.figure(figsize=(10, 5))
    plt.plot(exp_var, 'o-')
    plt.xlabel('Component')
    plt.ylabel('Explained variance')
    plt.savefig(os.path.join(output_dir, f'explained_variance.png'))


def get_files_from_list(files, target_strings):

    return [file for file in files if any(target in file for target in target_strings)]


def get_volume_from_list(input_list, target_string, zscore=True):
    files = get_files_from_list(input_list, [target_string])

    assert len(files) == 1, f'Expected 1 file, got {len(files)}'

    vol = nib.load(files[0]).get_fdata()

    if zscore:
        vol = (vol - vol.mean()) / vol.std()

    return vol

