
import glob
import os
from volumetric.surf_utils import align_surface, msm_resample_list, surface_modify_sphere
import nibabel as nib



def cell_type_analysis(receptor_surfaces, warped_sphere,  fixed_sphere, output_dir, clobber:bool=False):
    os.makedirs(output_dir, exist_ok=True)
    input_dir = 'data/mapped_volumes/surfaces/'

    moving_feature_surfaces = glob.glob(f'{input_dir}*.func.gii')

    warped_feature_surfaces = msm_resample_list(warped_sphere, fixed_sphere, moving_feature_surfaces, output_dir)

    
    receptor_array =

    cell_dens_array =
