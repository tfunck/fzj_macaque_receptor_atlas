import nibabel as nib
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from volumetric_gradient_analysis import volumetric_gradient_analysis



def get_receptor_volumes(volume_dir):
    ligand_receptor_dict={'ampa':'AMPA', 'kain':'Kainate', 'mk80':'NMDA', 'ly34':'mGluR2/3', 'flum':'GABA$_A$ Benz.', 'cgp5':'GABA$_B$', 'musc':'GABA$_A$ Agonist', 'sr95':'GABA$_A$ Antagonist', 'pire':r'Muscarinic M$_1$', 'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_3$','epib':r'Nicotinic $\alpha_4\beta_2$','oxot':r'Muscarinic M$_2$ (oxot)', 'praz':r'$\alpha_1$','uk14':r'$\alpha_2$ (agonist)','rx82':r'$\alpha_2$ (antagonist)', 'dpat':r'5-HT$_{1A}$','keta':r'5HT$_2$', 'sch2':r"D$_1$", 'dpmg':'Adenosine 1', 'cellbody':'Cell Body', 'myelin':'Myelin'}

    dir_path='~/projects/fzj_macaque_receptor_atlas/data/reconstruction/version_7_nlflow/volume_averages/*nii.gz'

    file_list=glob(os.path.expanduser(dir_path))

    ligands = [ os.path.basename(f).split('_')[1].replace('.nii.gz','') for f in file_list ] 

    print('Ligands found:', ligands)

    receptors =  [ ligand_receptor_dict[ligand] for ligand in ligands ]

    print('Receptors found:', receptors)


    receptors_out = []
    files_out = []

    for f, ligand, receptor in zip(file_list, ligands, receptors):
        if receptor == 'Cell Body' or receptor == 'Myelin' or ligand == 'epib':
            continue

        receptors_out.append(receptor)
        files_out.append(f)

    return receptors_out, files_out

def create_mask_from_volumes(receptor_volumes, output_filename, clobber=False):
    """Create a mask from receptor volumes using Otsu's thresholding."""
    if not os.path.exists(output_filename) or clobber :
        print('Creating mask from receptor volumes...')


        # Create mask
        img = nib.load(receptor_volumes[0])
        mask_data = np.zeros(img.shape, dtype=np.float32)

        for vol_file in receptor_volumes:
            vol = nib.load(vol_file)
            data = np.array(vol.dataobj, dtype=np.float32)

            mask_data += data
            del vol
            del data

        thresh = threshold_otsu(mask_data)
        print('Otsu threshold:', thresh)
        mask_data = (mask_data >= thresh).astype(np.uint8)

        mask_img = nib.Nifti1Image(mask_data, img.affine)
        nib.save(mask_img, output_filename)
        print('Mask saved to:', output_filename)
    else:
        print('Mask file already exists:', output_filename)


volume_dir = './data/reconstruction/version_7_nlflow/volume_averages/'
out_dir = './results/volumetric_gradient_analysis/'
os.makedirs(out_dir, exist_ok=True)

receptors, receptor_volumes = get_receptor_volumes(volume_dir)

mask_file = f'{out_dir}/macaque_receptor_mask.nii.gz'
create_mask_from_volumes(receptor_volumes, mask_file, clobber=False)


volumetric_gradient_analysis(mask_file, receptor_volumes, out_dir, approach='pca', n=20000, clobber=False)