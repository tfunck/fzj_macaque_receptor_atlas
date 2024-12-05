import matplotlib.pyplot as plt
import os
import numpy as np
import nibabel as nib
from kneed import KneeLocator

global ligand_receptor_dict
ligand_receptor_dict={  'ampa':'AMPA', 
                        'kain':'Kainate', 
                        'mk80':'NMDA', 
                        'ly34':'mGluR2/3', 
                        'flum':'GABA$_A$ Benz.', 
                        'cgp5':'GABA$_B$', 
                        'musc':'GABA$_A$ Agonist',
                        'sr95':'GABA$_A$ Antagonist', 
                        'pire':r'Muscarinic M$_1$', 
                        'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_3$',
                        'epib':r'Nicotinic $\alpha_4\beta_2$',
                        'oxot':r'Muscarinic M$_2$ (oxot)', 
                        'praz':r'$\alpha_1$',
                        'uk14':r'$\alpha_2$ (agonist)',
                        'rx82':r'$\alpha_2$ (antagonist)', 
                        'dpat':r'5-HT$_{1A}$',
                        'keta':r'5HT$_2$', 
                        'sch2':r"D$_1$", 'dpmg':'Adenosine 1', 'cellbody':'Cell Body', 'myelin':'Myelin'}

def get_category(f):
    fsplit = os.path.basename(f).split('_') 
    desc_idx = [i for i, x in enumerate(fsplit) if 'desc' in x][0]
    category = '_'.join(fsplit[3:desc_idx-1])
    return category

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

    #vol = nib.load(files[0]).get_fdata()
    data = nib.load(files[0]).darrays[0].data

    if zscore:
        data = (data - data.mean()) / data.std()

    return data


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

