import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from surf_utils import project_to_surface, write_gifti, plot_receptor_surf

from utils import ligand_receptor_dict

def read_valid_voxels(receptor_filename, mask_vol):
    """Read valid voxels from a receptor volume"""
    receptor = nib.load(receptor_filename)
    receptor_data = receptor.get_fdata()

    valid_idx = np.where(mask_vol > 0)

    return receptor_data[valid_idx]


def calc_volume_probabilities(volume_data, nbins): 
    voxels_prob = np.zeros(volume_data.shape[0])
    # calculate pixel intensity probabilities for entropy calculation
    # e.g., volume_data = [.1, .1, .5]
    print('Calculating probabilities with bins:', nbins, '...')
    print('volume_data.shape', volume_data.shape)

    valid_idx = ~ np.isnan(volume_data)

    counts, bins = np.histogram(volume_data[valid_idx], bins=nbins)

    # assign the voxel into bins based on their intensities
    # e.g., voxel_bins = [1, 1, 2]
    voxel_bins_short = np.digitize(volume_data[valid_idx], bins[1:])+1
    voxel_bins = np.zeros(volume_data.shape[0])
    voxel_bins[valid_idx] = voxel_bins_short
    # calculate the probability of each voxel intensity
    # e.g., voxel_prob = [2/3, 2/3, 1/3]
    # iterate over the bins 
    total_p = 0
    assert np.sum(valid_idx) == np.sum(voxel_bins>0), 'Invalid bins'    

    for b in np.unique(voxel_bins[valid_idx]):

        # calculate the total number of voxels in the bin
        idx = voxel_bins == b

        assert np.sum(idx) > 0, 'No voxels in bin'

        # divide by the total number of voxels to get the probability
        p = np.sum(idx) / np.sum(valid_idx)

        print('Bin:', b, 'p(Bin):', np.round(p,5)) 
        assert p > 0 , f'Probability is zero'

        # assign the probability to the voxel for receptor i and bin b
        voxels_prob[idx] = p

        total_p += p

    voxels_prob[~valid_idx] = np.nan

    assert np.abs(total_p - 1) < 1e-6, f'Total probability is not 1: {total_p}'

    return voxels_prob


def write_prob_volume(voxels_prob, valid_idx, volume_file, prob_filename):
    prob_volume = np.zeros(nib.load(volume_file).darrays[0].data.shape).astype(np.float32)

    prob_volume[valid_idx] = voxels_prob

    print('Write:',prob_filename)
    #nib.Nifti1Image(prob_volume, affine).to_filename(prob_filename)

def plot_entropy_figure(entropy, prob_filename_list, output_dir):
    """Use matplotlib display the entropy and probability distributions. 
    Volumes are shown along y- and z-axes, based on center of mass of the entropy volume"""

    # calculate the center of mass of the entropy volume
    from scipy.ndimage.measurements import center_of_mass
    x0, _, z0 = np.rint(center_of_mass(entropy)).astype(int)

    n = len(prob_filename_list)
    ncol = np.ceil(np.sqrt(n)).astype(int)
    nrow = n // ncol

    ncol*=2
    nrow+=1

    # create a figure with 2 plots for the entropy volume and 2 plots for each of the probability volumes
    print('Create figure', nrow, ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=( 5*ncol, 5*nrow))

    # plot the entropy volume
    #vmin, vmax = np.percentile(entropy, [0, 100])
    #axs[0, 0].imshow(entropy[:, :, z0],vmin=vmin, vmax=vmax, cmap='nipy_spectral')
    #axs[0, 0].set_title('Entropy')
    #axs[0, 1].imshow(np.rot90(entropy[x0, :, :],1), vmin=vmin, vmax=vmax, cmap='nipy_spectral')
    #for j in range(2,ncol):
    #    axs[0,j].set_visible(False)

    # plot the probability volumes
    #counter = 0
    #for i in range(1,nrow):
    #    for j in range(0,ncol,2):
    #        prob_filename = prob_filename_list[counter]
    #        counter += 1
    #
    #        prob_volume = nib.load(prob_filename).get_fdata()
    #        vmin, vmax = np.percentile(prob_volume, [1, 99])
    #        label = os.path.basename(prob_filename).replace("_prob.nii.gz", "")

    #       axs[i, j].imshow(prob_volume[:, :, z0], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    #        axs[i, j].set_title(f'Probability: {label}')
    #        axs[i, j+1].imshow(np.rot90(prob_volume[x0, :, :],1), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    
    #for i in range(nrow):
    #    for j in range(ncol):
    #        axs[i, j].axis('off')
    # decrease spacing between subplots
    #plt.subplots_adjust(wspace=0.1, hspace=0.0)
    #plt.tight_layout()
    #plt.savefig(f'{output_dir}/figure_entropy.png')


def calculate_probability_volumes(receptor_surfaces, output_dir, nbins=64, clobber=False):
    prob_filename_list = []

    prob_filename_list = [ f'{output_dir}/{os.path.basename(volume_file).replace(".surf.gii", "_prob.surf.gii")}' for volume_file in receptor_surfaces ]
    
    # iterate over the receptor volumes
    for i, (prob_filename, receptor_file) in enumerate(zip(prob_filename_list, receptor_surfaces)):

        surf_data = nib.load(receptor_file).darrays[0].data
        #volume_data = volume_data[valid_idx]
        valid_idx = np.where( ~ np.isnan(surf_data) )

        sum_check = np.sum(np.abs(surf_data[valid_idx]))

        assert sum_check > 0, f'Volume {receptor_file} is empty'
        assert ~ np.isnan(sum_check), f'Volume {receptor_file} has nan values'

        # calculate the probability of each voxel intensity
        if not os.path.exists(prob_filename) or clobber :
            voxels_prob = calc_volume_probabilities(surf_data, nbins).astype(np.float16) 
            assert np.sum(voxels_prob == 0 ) == 0 
            write_gifti(voxels_prob, prob_filename)

    return prob_filename_list


def entropy_analysis(self, nbins:int=16, clobber:bool=False):
        """Calculate the voxelwise entropy over a set of volumes"""
        output_dir = self.output_dir + '/entropy/'

        os.makedirs(output_dir, exist_ok=True)
        
        entropy_surf_filenames = []

        prob_npy_filename =  f'{output_dir}/prob_features.npy'

        prob_surf_files = calculate_probability_volumes( self.receptor_surfaces, output_dir, nbins=nbins, clobber=clobber )
        
        if not os.path.exists(prob_npy_filename) or clobber :
            prob = np.array([ np.array( nib.load(file).darrays[0].data ) for file in prob_surf_files ])
            np.save(prob_npy_filename, prob)
        else :
            prob = np.load(prob_npy_filename)
        
        entropy_filename = os.path.join(output_dir, f'entropy.func.gii')
        total_entropy_filename = f'{output_dir}/total_entropy.npy'

        if not os.path.exists(entropy_filename) or clobber: 
            # calculate the entropy of the voxels across the receptor volumes
            print(prob.shape)
            entropy = - np.sum( prob * np.log2(prob), axis=0)
            print(entropy.shape)

            write_gifti(entropy, entropy_filename) 
            
            print('-->', entropy_filename)

            plot_receptor_surf(
                [entropy_filename], self.cortical_surface, output_dir, label='Entropy '+self.label, cmap='RdBu_r', threshold=[2,98]
                )
    
            if not os.path.exists(total_entropy_filename) or clobber:
                entropy = nib.load(entropy_filename).darrays[0].data

                total_entropy = -np.sum(prob * np.log2(prob)) / np.log2(prob.shape[1])

                np.save(total_entropy_filename, total_entropy)
            else :
                total_entropy = np.load(total_entropy_filename)
        
        return entropy_surf_filenames, total_entropy_filename

def calculate_std_dev(volumes, mask_file):

    mask = nib.load(mask_file)
    mask_vol = mask.get_fdata()
    
    valid_idx = np.where(mask_vol > 0)

    voxels = np.zeros((len(volumes), valid_idx[0].shape[0]))
    for i, volume_file in enumerate(volumes):
        # standardize the voxel intensities
        voxels[i] = (volume_data - np.mean(volume_data)) / np.std(volume_data)






