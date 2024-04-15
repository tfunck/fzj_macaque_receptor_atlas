"""Surface based analysis for 3D receptors"""

import argparse
import os
import nibabel as nib
import numpy as np
import ants
import glob
import brainbuilder.utils.mesh_utils as mesh_utils 
from brainbuilder.utils.mesh_utils import mesh_to_volume
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from matplotlib_surface_plotting import plot_surf
from surfaces.surface_gradient_analysis import roi_gradients, vertex_gradients
from surfaces.surface_diff import surface_diff
import utils

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


 

def plot_summed_receptor_surf(
        receptor_surfaces, 
        gm_surf_filename, 
        ligands, 
        profiles_dir, 
        label='', 
        cmap='RdBu_r'
        ):  
    rec_list = utils.get_files_from_list( receptor_surfaces, ligands)
    assert len(rec_list) > 0, f'No receptor profiles found for {ligands}'
    plot_receptor_surf(rec_list, gm_surf_filename, profiles_dir, label=label, cmap=cmap)
    return rec_list




def surface_analysis(receptor_volumes, wm_surf_filename, gm_surf_filename, profiles_dir, output_dir):
    """Surface based analysis for 3D receptors"""

    receptor_surfaces = project_to_surface(
        receptor_volumes, 
        wm_surf_filename, 
        gm_surf_filename, 
        profiles_dir,
        n = 10,
        sigma = 0,
        zscore = False,
        clobber = True
        )

    profiles_dir = f"{output_dir}/profiles/"
    diff_gradient_dir = f"{output_dir}/diff_gradients/"

    os.makedirs(profiles_dir,exist_ok=True)
    os.makedirs(diff_gradient_dir,exist_ok=True)

    """
    inh_list = plot_summed_receptor_surf(
        receptor_surfaces, gm_surf_filename, ['flum', 'musc', 'cgp5'], profiles_dir, label='Inh'
        )
    
    ex_list = plot_summed_receptor_surf(
        receptor_surfaces, gm_surf_filename, ['ampa', 'kain', 'mk80'], profiles_dir, label='Ex'
        )
        
    mod_list = plot_summed_receptor_surf(
        receptor_surfaces, gm_surf_filename, ['dpat', 'uk14', 'oxot', 'keta', 'sch2', 'pire'], profiles_dir, label='Mod'
        )

    """
    surface_diff(receptor_surfaces, wm_surf_filename, wm_surf_filename, diff_gradient_dir, label='all')
    for surface_filename in receptor_surfaces:
        label = os.path.basename(surface_filename).replace('.npy','')
        print('Surface Difference:', label)
        surface_diff([surface_filename], wm_surf_filename, wm_surf_filename, diff_gradient_dir, label=label)

 
