import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

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
    counts, bins = np.histogram(volume_data, bins=nbins)


    # assign the voxel into bins based on their intensities
    # e.g., voxel_bins = [1, 1, 2]
    voxel_bins = np.digitize(volume_data, bins[1:])

    # calculate the probability of each voxel intensity
    # e.g., voxel_prob = [2/3, 2/3, 1/3]
    # iterate over the bins 
    for b in np.unique(voxel_bins):
        # calculate the total number of voxels in the bin
        idx = voxel_bins == b

        # divide by the total number of voxels to get the probability
        p = np.sum(idx) / np.sum(counts)

        # assign the probability to the voxel for receptor i and bin b
        voxels_prob[idx] = p

    return voxels_prob


def write_prob_volume(prob_volume, voxels_prob, valid_idx, output_dir, volume_file, affine):
    prob_volume[valid_idx] = voxels_prob
    prob_filename = f'{output_dir}/{os.path.basename(volume_file).replace(".nii.gz", "_prob.nii.gz")}'
    print('Write:',prob_filename)
    nib.Nifti1Image(prob_volume, affine).to_filename(prob_filename)
    return prob_filename

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
    vmin, vmax = np.percentile(entropy, [0, 100])
    axs[0, 0].imshow(entropy[:, :, z0],vmin=vmin, vmax=vmax, cmap='nipy_spectral')
    axs[0, 0].set_title('Entropy')
    axs[0, 1].imshow(np.rot90(entropy[x0, :, :],1), vmin=vmin, vmax=vmax, cmap='nipy_spectral')
    for j in range(2,ncol):
        axs[0,j].set_visible(False)

    # plot the probability volumes
    counter = 0
    for i in range(1,nrow):
        for j in range(0,ncol,2):
            prob_filename = prob_filename_list[counter]
            counter += 1

            prob_volume = nib.load(prob_filename).get_fdata()
            vmin, vmax = np.percentile(prob_volume, [1, 99])
            label = os.path.basename(prob_filename).replace("_prob.nii.gz", "")

            axs[i, j].imshow(prob_volume[:, :, z0], vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axs[i, j].set_title(f'Probability: {label}')
            axs[i, j+1].imshow(np.rot90(prob_volume[x0, :, :],1), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    
    for i in range(nrow):
        for j in range(ncol):
            axs[i, j].axis('off')
    # decrease spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    plt.tight_layout()

    plt.savefig(f'{output_dir}/figure_entropy.png')


def entropy_analaysis(mask_file, volumes, output_dir, nbins=2048, clobber=False):
    """Calculate the entropy of receptor volumes and compare to gradient volumes"""
    os.makedirs(output_dir, exist_ok=True)    

    output_filename = os.path.join(output_dir, 'entropy.nii.gz')
    std_output_filename = os.path.join(output_dir, 'std.nii.gz')
    mean_output_filename = os.path.join(output_dir, 'mean.nii.gz')

    if not os.path.exists(output_filename) or\
        not os.path.exists(std_output_filename) or\
        not os.path.exists(mean_output_filename) or\
        clobber:

        prob_filename_list = []

        mask = nib.load(mask_file)
        mask_vol = mask.get_fdata()
        
        prob_volume = np.zeros(mask_vol.shape)

        valid_idx = np.where(mask_vol > 0)

        voxels_prob = np.zeros((len(volumes), valid_idx[0].shape[0]))
        voxels = np.zeros((len(volumes), valid_idx[0].shape[0]))

        # iterate over the receptor volumes
        for i, volume_file in enumerate(volumes):
            volume = nib.load(volume_file)
            volume_data = volume.get_fdata()
            volume_data = volume_data[valid_idx]

            # calculate the probability of each voxel intensity
            voxels_prob[i] = calc_volume_probabilities(volume_data, nbins) 
            prob_filename = write_prob_volume(prob_volume, voxels_prob[i], valid_idx, output_dir, volume_file, volume.affine) 
            prob_filename_list.append(prob_filename)

            # standardize the voxel intensities
            voxels[i] = (volume_data - np.mean(volume_data)) / np.std(volume_data)

        # calculate the entropy of the voxels across the receptor volumes    
        entropy = -np.sum(voxels_prob * np.log2(voxels_prob), axis=0)
        entropy /= np.log2(voxels_prob.shape[0])
        entropy_vol = np.zeros(mask_vol.shape)
        entropy_vol[valid_idx] = entropy

        plot_entropy_figure(entropy_vol, prob_filename_list, output_dir)

        std_vol = np.zeros(mask_vol.shape)
        mean_vol = np.zeros(mask_vol.shape)
        std_vol[valid_idx] = np.std(voxels, axis=0) 
        mean_vol[valid_idx] = np.mean(voxels, axis=0)

        nib.Nifti1Image(mean_vol, nib.load(volumes[0]).affine).to_filename(mean_output_filename)
        nib.Nifti1Image(std_vol, nib.load(volumes[0]).affine).to_filename(std_output_filename)
        nib.Nifti1Image(entropy_vol, nib.load(volumes[0]).affine).to_filename(output_filename)

    return output_filename, std_output_filename, mean_output_filename