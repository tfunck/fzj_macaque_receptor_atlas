import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from volumetric.surf_utils import project_to_surface, project_and_plot_surf, load_gifti, write_gifti, plot_receptor_surf

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
    print(len(counts))
    
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


def write_prob_volume(voxels_prob, valid_idx, output_dir, volume_file):
    prob_volume = np.zeros(nib.load(volume_file).shape).astype(np.float32)
    affine = nib.load(volume_file).affine

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


def calculate_probability_volumes(volumes, mask_file, output_dir, nbins=64, clobber=False):
    prob_filename_list = []

    mask = nib.load(mask_file)
    mask_vol = mask.get_fdata()
   
    dimensions = mask_vol.shape

    valid_idx = np.where(mask_vol > 0)

    prob_filename_list = [ f'{output_dir}/{os.path.basename(volume_file).replace(".nii.gz", "_prob.nii.gz")}' for volume_file in volumes ]
    
    # iterate over the receptor volumes
    for i, (prob_file, volume_file) in enumerate(zip(prob_filename_list, volumes)):

        volume = nib.load(volume_file)
        volume_data = volume.get_fdata()
        volume_data = volume_data[valid_idx]

        # calculate the probability of each voxel intensity
        if not os.path.exists(prob_file) or clobber :
            voxels_prob = calc_volume_probabilities(volume_data, nbins).astype(np.float16) 
            assert np.sum(voxels_prob == 0 ) == 0 
            prob_filename = write_prob_volume(voxels_prob, valid_idx, output_dir, volume_file) 

    return prob_filename_list

def calculate_std_dev(volumes, mask_file):

    mask = nib.load(mask_file)
    mask_vol = mask.get_fdata()
    
    valid_idx = np.where(mask_vol > 0)

    voxels = np.zeros((len(volumes), valid_idx[0].shape[0]))
    for i, volume_file in enumerate(volumes):
        # standardize the voxel intensities
        voxels[i] = (volume_data - np.mean(volume_data)) / np.std(volume_data)


def entropy_analysis(
        mask_file, medial_wall_mask, volumes, wm_surf_filename, gm_surf_filename, subsets, output_dir, descriptor='all', nlayers=5, nbins=64, clobber=False
        ):
    """Calculate the voxelwise entropy over a set of volumes"""
    os.makedirs(output_dir, exist_ok=True)
    
    entropy_surf_filenames = []
    total_entropy_filenames = []

    prob_npy_filename =  f'{output_dir}/prob_features.npy'

    prob_volume_list = calculate_probability_volumes(volumes, mask_file, output_dir, nbins=nbins, clobber=clobber)
    exit(0)

    prob_files = project_to_surface( prob_volume_list, wm_surf_filename, gm_surf_filename, output_dir, n=nlayers, zscore=False, clobber=False)
    
    if not os.path.exists(prob_npy_filename) or clobber :
        prob = np.array([ np.array(nib.load(file).darrays[0].data) for file in prob_files ])
        np.save(prob_npy_filename, prob)
    else :
        prob = np.load(prob_npy_filename)
    # prob = receptor x vertex x depth
    
    for i, j in subsets :
        s0 = int(np.rint(100*i/nlayers))
        s1 = int(np.rint(100*(j-1)/nlayers))

        entropy_filename = os.path.join(output_dir, f'entropy_{s0}-{s1}%.func.gii')
        total_entropy_filename = f'{output_dir}/total_entropy_{s0}-{s1}%.npy'
        print(entropy_filename)

        entropy_surf_filenames.append(entropy_filename)
        total_entropy_filenames.append(total_entropy_filenames)

        if not os.path.exists(entropy_filename) or clobber or True: 
            # calculate the entropy of the voxels across the receptor volumes
            print('subset', i, j, prob.shape)
            prob_sub = prob[:,:,i:j]
            assert np.sum(np.isnan(prob_sub) + np.isinf(prob_sub)) == 0

            entropy = np.max(prob_sub,axis=(0,2))
            entropy = -np.sum(prob_sub * np.log2(prob_sub), axis=(0,2))
            #entropy /= np.log2(prob.shape[0])
            #entropy = np.sum(prob_sub, axis=(0,2))

            entropy[medial_wall_mask] = np.nan

            write_gifti(entropy, entropy_filename)    
            plot_receptor_surf([entropy_filename], wm_surf_filename, output_dir, label=f'entropy_{s0}-{s1}%', cmap='RdBu_r', threshold=[2,98])
  
        if not os.path.exists(total_entropy_filename) or clobber or True :
            entropy = nib.load(entropy_filename).darrays[0].data
            total_entropy = -np.sum(prob * np.log2(prob)) / np.log2(prob.shape[1])
            np.save(total_entropy_filename, total_entropy)
        else :
            total_entropy = np.load(total_entropy_filename)[0]
    exit(0)   
    return entropy_surf_filenames, total_entropy_filenames



