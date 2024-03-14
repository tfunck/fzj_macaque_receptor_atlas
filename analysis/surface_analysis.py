"""Surface based analysis for 3D receptors"""

import argparse
import os
import nibabel as nib
import numpy as np
import ants
import glob
import brainbuilder.utils.mesh_utils as mesh_utils 

from surfaces.surface_gradient_analysis import roi_gradients, vertex_gradients


def mebrains_to_yerkes(mebrains_filename, yerkes_filename, align_dir):
    """Align MEBRAINS to Yerkes"""
    mebrains = ants.image_read(mebrains_filename)
    yerkes = ants.image_read(yerkes_filename)

    outputprefix = f"{align_dir}/mebrains_to_yerkes_"

    tfm_file = f"{outputprefix}0GenericAffine.mat"

    if not os.path.exists(tfm_file):
        reg = ants.registration(
            fixed=yerkes, 
            moving=mebrains, 
            type_of_transform='Affine', 
            outprefix=outputprefix
            ) 

    return tfm_file

def transform_receptor_volumes(receptor_volumes, tfm, output_dir):
    """Transform 3D reconstruction to Yerkes"""

    receptor_volumes_transformed = []

    for receptor_volume in receptor_volumes:
        receptor = ants.image_read(receptor_volume)
        receptor_transformed = f"{output_dir}/{os.path.basename(receptor_volume)}"
        receptor_volumes_transformed.append(receptor_transformed)

        if not os.path.exists(receptor_transformed):
            receptor = ants.apply_transforms(
                fixed=receptor, 
                moving=receptor, 
                transformlist=tfm, 
                interpolator='linear'
                )

            receptor.to_filename(receptor_transformed)

    return receptor_volumes_transformed


def project_to_surface(receptor_volumes, wm_surf_filename, gm_surf_filename, output_dir, n=10):
    """Project recosntructions to Yerkes Surface"""
    profile_list = []

    for receptor_volume in receptor_volumes:
        profile_fn = f"{output_dir}/{os.path.basename(receptor_volume).replace('.nii.gz','')}.npy"
        profile_list.append(profile_fn)

        if not os.path.exists(profile_fn) :
            receptor = nib.load(receptor_volume)
            receptor = receptor.get_fdata()

            #z score receptor data
            wm_coords, _ = mesh_utils.load_mesh_ext(wm_surf_filename)
            gm_coords, _ = mesh_utils.load_mesh_ext(gm_surf_filename)

            d_vtr = gm_coords - wm_coords

            profiles = np.zeros([wm_coords.shape[0], n])

            for i, d in enumerate(np.linspace(0, 1, n)):

                coords = gm_coords + d_vtr * d

                surf_receptor_vtr = mesh_utils.volume_filename_to_mesh(coords, receptor_volume, zscore=True)

                profiles[:,i] = surf_receptor_vtr
            
            # Save profiles
            np.save(profile_fn, profiles)

    return profile_list

if __name__ == '__main__':
    print('Surface based analysis for 3D receptors')
    parser = argparse.ArgumentParser(description='Surface-based Gradient Analysis')
    parser.add_argument('-m', dest='mebrains_filename', default='data/volumes/MEBRAINS_T1_masked.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-y', dest='yerkes_filename', type=str, default='data/volumes/MacaqueYerkes19_v1.2_AverageT1w_restore_masked.nii.gz', help='Path to Yerkes volume')
    parser.add_argument('-o', dest='output_dir', type=str, default='outputs/surf', help='Path to output directory')
    parser.add_argument('-n', dest='n', default=10000, type=int, help='Number of random voxels to sample')
    parser.add_argument('-i', dest='input_dir', type=str, default='data/reconstruction/receptor/', help='Path to receptor volumes')
    parser.add_argument('--wm-surf', dest='wm_surf_filename', type=str, default='data/surfaces/MacaqueYerkes19.L.white.10k_fs_LR.surf.gii', help='Path to Yerkes white matter surface')
    parser.add_argument('--gm-surf', dest='gm_surf_filename', type=str, default='data/surfaces/MacaqueYerkes19.L.pial.10k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--surf-atlas', dest='surf_atlas_filename', type=str, default='data/surfaces/L.Markov.monkey.10k_fs_LR.label.gii', help='Path to surface mask')


    args = parser.parse_args()

    receptor_volumes = glob.glob(f'{args.input_dir}/*nii.gz')

    align_dir = f"{args.output_dir}/align/"
    profiles_dir = f"{args.output_dir}/profiles/"
    gradient_dir = f"{args.output_dir}/gradients/"

    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(profiles_dir, exist_ok=True)
    os.makedirs(gradient_dir, exist_ok=True)

    # Align MEBRAINS to Yerkes
    ### FIXME currently volumetric alignment, replace with surface-based
    tfm = mebrains_to_yerkes(args.mebrains_filename, args.yerkes_filename, align_dir)

    # Transform 3D reconstruction to Yerkes
    receptor_volumes = transform_receptor_volumes(receptor_volumes, tfm, align_dir)

    # Project recosntructions to Yerkes Surface
    receptor_surfaces = project_to_surface(receptor_volumes, args.wm_surf_filename, args.gm_surf_filename, profiles_dir)

    # Calculate ROI-based gradients
    roi_gradients(receptor_surfaces, args.surf_atlas_filename, args.gm_surf_filename, gradient_dir)

    # Calculate vertex-based gradients
    vertex_gradients(receptor_surfaces, args.wm_surf_filename, gradient_dir)