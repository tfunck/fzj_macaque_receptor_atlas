import os
import subprocess
import numpy as np
import nibabel as nib
import shutil

from brainbuilder.interp.surfinterp import interpolate_over_surface
from matplotlib_surface_plotting import plot_surf
from nibabel.freesurfer import read_morph_data, read_geometry
import brainbuilder.utils.mesh_utils as mesh_utils
from brainbuilder.utils.mesh_utils import load_mesh_ext

def msm_align(
        fixed_sphere, 
        fixed_data, 
        moving_sphere, 
        moving_data, 
        output_dir, 
        clobber=False
        ):
    """Align two surfaces using MSM."""
    #fixed_sphere = fix_surf(fixed_sphere, output_dir)
    #moving_sphere = fix_surf(moving_sphere, output_dir)
    
    #moving_sphere = convert_fs_to_gii(moving_sphere, output_dir, clobber=True)
    
    data_out = f'{output_dir}/{os.path.basename(moving_data).replace(".func.gii","_warped")}'

    base = os.path.basename(moving_sphere).replace('.surf.gii','')
    
    out_sphere = f'{data_out}sphere.reg.surf.gii'
    

    if not os.path.exists(out_sphere)  or clobber :
        cmd = f"msm --inmesh={moving_sphere} --indata={moving_data} --refmesh={fixed_sphere} --refdata={fixed_data} --out={data_out} --levels=2 --verbose=1"
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")

        # following example of msmresample works:
        cmd=f"msmresample {out_sphere} {output_dir}/moving_data_rsl -labels {moving_data} -project {fixed_sphere} -adap_bary"
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        #plot_receptor_surf(
        #    [f'{output_dir}/moving_data_rsl.func.gii'], 
        #    fixed_sphere, output_dir,  label='mv_rsl', cmap='nipy_spectral')

        plot_receptor_surf(
            [fixed_data], 
            fixed_sphere, output_dir,  label='fx_orig', cmap='nipy_spectral')

        #plot_receptor_surf(
        #    [f'{data_out}transformed_and_reprojected.func.gii'], 
        #    fixed_sphere, output_dir,  label='mv_warpedproj', cmap='nipy_spectral')

    metrics_rsl_list = msm_resample(out_sphere, fixed_sphere, moving_data, clobber=clobber)

    return out_sphere, data_out, metrics_rsl_list

def msm_resample_list(rsl_mesh, fixed_mesh, labels, output_dir, clobber=False):
    """Apply MSM to labels."""
    labels_rsl_list = []
    for label in labels:
        #input_mesh.sphere.reg.gii output_metric_basename -labels input_metric.func.gii -project target_mesh.surf,gii -adap_bary
        print('Resampling label\n\t', label)
        print('to\n\t', rsl_mesh)
        label_rsl_filename = msm_resample(rsl_mesh, fixed_mesh, label, clobber=clobber)[0]

        labels_rsl_list.append(label_rsl_filename) #FIXME

    return labels_rsl_list

def msm_resample(rsl_mesh, fixed_mesh, label=None, clobber=False):
    output_label_basename = label.replace('.func','').replace('.gii','') + '_rsl'
    output_label = f'{output_label_basename}.func.gii'
    template_label_rsl_filename = f'{output_label_basename}.func.gii'

    cmd = f"msmresample {rsl_mesh} {output_label_basename} -project {fixed_mesh} -adap_bary"

    if not os.path.exists(output_label) or clobber:
        if label is not None :
            cmd += f" -labels {label}"
        print(cmd);
        subprocess.run(cmd, shell=True, executable="/bin/bash")

    n = nib.load(fixed_mesh).get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data.shape[0]

    label_rsl_list = [] 
    darrays = nib.load(output_label).darrays

    for i, darray in enumerate(darrays):
        curr_label_rsl_filename = template_label_rsl_filename.replace('_rsl',f'_{i}_rsl')
        label_rsl_list.append(curr_label_rsl_filename)
        if not os.path.exists(curr_label_rsl_filename) or clobber:
            data = darray.data.astype(np.float32)
            assert data.shape[0] == n, f"Data shape is {data.shape}"
            print('Writing to\n\t', curr_label_rsl_filename)
            write_gifti( data, curr_label_rsl_filename )
    
    return label_rsl_list


def fix_surf(surf_fn, output_dir ):
    """Fix surface by using surface-modify-sphere command."""
    base = os.path.basename(surf_fn).replace('.surf.gii','')
    out_fn = f"{output_dir}/{base}.surf.gii" 
    cmd = f"wb_command -surface-modify-sphere {surf_fn} 100 {out_fn}"

    subprocess.run(cmd, shell=True, executable="/bin/bash")    
    return out_fn


def get_surface_sulcal_depth(surf_filename, output_dir, n=10, clobber=False):
    """Get sulcal depth using mris_inflate."""
    base = os.path.basename(surf_filename).replace('.surf.gii','')

    if 'lh.' == get_fs_prefix(surf_filename):
        prefix='lh'
    else :
        prefix='rh'

    sulc_suffix = f'{base}.sulc'
    temp_sulc_filename = f'{output_dir}/{prefix}.{sulc_suffix}'
    sulc_filename = f'{output_dir}/lh.{sulc_suffix}'
    inflated_filename = f'{output_dir}/lh.{base}.inflated'

    if not os.path.exists(sulc_filename) or clobber:
        cmd = f"mris_inflate -n {n} -sulc {sulc_suffix} {surf_filename} {inflated_filename}"
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        shutil.move(temp_sulc_filename, sulc_filename)
    
    assert os.path.exists(sulc_filename), f"Could not find sulcal depth file {sulc_filename}"
    return sulc_filename, inflated_filename

def resample_label(label_in, sphere_fn, sphere_rsl_fn, output_dir, clobber=False):
    n = nib.load(sphere_rsl_fn).darrays[0].data.shape[0]
    label_out = f'{output_dir}/n-{n}_{os.path.basename(label_in)}'

    if not os.path.exists(label_out) or clobber:
        cmd = f'wb_command -label-resample {label_in} {sphere_fn} {sphere_rsl_fn} BARYCENTRIC {label_out} -largest'
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    assert os.path.exists(label_out), f"Could not find resampled label {label_out}" 

    return label_out

def resample_surface(surface_in, sphere_fn, sphere_rsl_fn, output_dir, n, clobber=False):
    surface_out = f'{output_dir}/n-{n}_{os.path.basename(surface_in)}'

    if not os.path.exists(surface_out) or clobber:
        cmd = f'wb_command -surface-resample {surface_in} {sphere_fn} {sphere_rsl_fn} BARYCENTRIC {surface_out}'
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    assert os.path.exists(surface_out), f"Could not find resampled surface {surface_out}" 

    return surface_out

def remesh_surface(surface_in,  output_dir, n=10000, radius=1, clobber=False):
    # run command line 
    base = os.path.basename(surface_in)
    temp_surface_out=f'{output_dir}/n-{n}_temp_{base}'
    surface_out=f'{output_dir}/n-{n}_{base}'
    if not os.path.exists(surface_out) or clobber:
        #cmd = f'mris_remesh --nvert {n} -i {surface_in} -o /tmp/{base} && mris_convert /tmp/{base} {surface_out}'
        cmd = f'mris_remesh --nvert {n} -i {surface_in} -o {temp_surface_out} && wb_command  -surface-modify-sphere  {temp_surface_out} {radius} {surface_out} -recenter'
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")

    assert os.path.exists(surface_out), f"Could not find resampled surface {surface_out}"
    
    return surface_out

def get_fs_prefix(surf_filename):
    prefix=''
    target_prefix=os.path.basename(surf_filename)[0:3]
    return target_prefix
    

def get_surface_curvature(surf_filename, output_dir ,n=10, clobber=False):
    """Get surface curvature using mris_curvature."""

    target_prefix = get_fs_prefix(surf_filename)
    prefix=''
    if 'lh.' not in target_prefix and 'rh.' not in target_prefix: 
        prefix='unknown.'

    print()
    print(target_prefix)
    print(prefix)
    print()

    base = prefix+os.path.basename(surf_filename)#.replace('.surf.gii','')
    dirname = os.path.dirname(surf_filename)
    curv_filename = f'{dirname}/{base}.H'
    output_filename = f'{output_dir}/{base}.H'
    if not os.path.exists(output_filename) or clobber :
        cmd = f"mris_curvature -w -a {n}  {surf_filename}"
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        shutil.move(curv_filename, output_filename)
    
    assert os.path.exists(output_filename), f"Could not find curvature file {output_filename}"

    return output_filename

def convert_fs_morph_to_gii(input_filename, output_dir, clobber=False)  :
    """Convert FreeSurfer surface to GIFTI."""
    base = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f'{output_dir}/{base}.surf.gii'

    if not os.path.exists(output_filename) or clobber:
        ar = read_morph_data(input_filename).astype(np.float32)
        g = nib.gifti.GiftiImage()
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar))
        nib.save(g, output_filename)
    return output_filename

def convert_fs_to_gii(input_filename, output_dir, clobber=False):
    """Convert FreeSurfer surface to GIFTI."""
    base = '_'.join( os.path.basename(input_filename).split('.')[0:-2])
    output_filename = f'{output_dir}/{base}.surf.gii'

    if not os.path.exists(output_filename) or clobber:
        try :
            ar = read_geometry(input_filename)
            print('Freesurfer')
        except ValueError:
            darrays = nib.load(input_filename).darrays
            ar = [ darrays[0].data, darrays[1].data ]
            print('Gifti')

        coordsys = nib.gifti.GiftiCoordSystem(dataspace='NIFTI_XFORM_TALAIRACH', xformspace='NIFTI_XFORM_TALAIRACH')
        g = nib.gifti.GiftiImage()
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar[0].astype(np.float32), intent='NIFTI_INTENT_POINTSET',coordsys=coordsys))
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar[1].astype(np.int32), intent='NIFTI_INTENT_TRIANGLE', coordsys=None))
        nib.save(g, output_filename)
    return output_filename



def create_zyaxis_file(surf_filename, curv_filename, output_dir, clobber=False):
    """Create a gifti file with the product of z and y coordinates as the data."""
    base = os.path.basename(surf_filename).replace('.surf.gii','')
    output_filename = f'{output_dir}/{base}_zyaxis.func.gii'
    if not os.path.exists(output_filename) or clobber:
        coords, _ = mesh_utils.load_mesh_ext(surf_filename)
        curv = load_gifti(curv_filename)
        zyaxis = coords[:,2] * coords[:,1]
        zyaxis = (zyaxis - zyaxis.min()) / (zyaxis.max() - zyaxis.min())
        write_gifti(zyaxis, output_filename)
    return output_filename

def get_surface_metrics(surf_filename, output_dir, clobber=False):
    base = os.path.basename(surf_filename).replace('.surf.gii','')
    output_file = f'{output_dir}/lh.{base}_metrics.func.gii'
    
    fs_sulc_filename, _ = get_surface_sulcal_depth(surf_filename, output_dir,n=10, clobber=clobber)
    fs_curv_filename = get_surface_curvature(surf_filename, output_dir, n=100, clobber=clobber)

    sulc_filename = convert_fs_morph_to_gii(fs_sulc_filename, output_dir, clobber=clobber)
    curv_filename = convert_fs_morph_to_gii(fs_curv_filename, output_dir, clobber=clobber)
    zyaxis_filename = create_zyaxis_file(surf_filename, curv_filename, output_dir, clobber=clobber)

    if not os.path.exists(output_file) or clobber:
        # merge input metrics
        #wb_command -metric-merge output_name.func.gii -metric metric1.func.gii -metric metric2.func.gii -metric metric3.func.gii
        #cmd = f"wb_command -metric-merge {output_file} -metric {sulc_filename} -metric {curv_filename}"
        cmd = f"wb_command -metric-merge {output_file} -metric {sulc_filename} "
        #cmd = f"wb_command -metric-merge {output_file} -metric {zyaxis_filename} "
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    
    return output_file, {'sulc':sulc_filename, 'curv':curv_filename, 'zyaxis':zyaxis_filename} 

def write_gifti(array, filename, intent='NIFTI_INTENT_SHAPE'):
    gifti_img = nib.gifti.gifti.GiftiImage()
    gifti_array = nib.gifti.GiftiDataArray(array.astype(np.float32), intent=intent)
    gifti_img.add_gifti_data_array(gifti_array)
    print('Mean:', array.mean(), 'Std:', array.std())
    print('Writing to\n\t', filename)
    gifti_img.to_filename(filename)

def load_gifti(filename):
    return nib.load(filename).darrays[0].data

def interpolate_gradient_over_surface(
        decimated_surface_val:np.ndarray,
        surface_file:str,
        sphere_file:str,
        output_dir:str,
        component:int,
        valid_idx:np.ndarray,
        clobber:bool=False):
    '''
    Interpolate gradient over surface. The gradient is calculated based on a decimated surface
    and needs to be interpolated to the original surface
    :param grad: gradient (n_layers, n_points, n_components)
    :param surface_file: surface file name
    :param sphere_file: sphere file name
    :param clobber: clobber existing output file
    :return: np.ndarray
    '''
    # Load surface
    coords, faces = load_mesh_ext(surface_file)

    # Interpolate gradient over surface
    grad_surf_fn = f'{output_dir}/grad_surf_{component}.npy'

    if not os.path.exists(grad_surf_fn) or clobber:
        surface_val = np.zeros(coords.shape[0])
        surface_val[valid_idx] = decimated_surface_val
        interp_surface_mask = np.zeros(coords.shape[0]).astype(bool)
        interp_surface_mask[valid_idx] = 1
        surface_val = interpolate_over_surface(sphere_file, surface_val, order=1, surface_mask=interp_surface_mask)

        np.save(grad_surf_fn, surface_val)
    else :
        surface_val = np.load(grad_surf_fn)
        
    return surface_val

def project_to_surface(
        receptor_volumes, 
        wm_surf_filename, 
        gm_surf_filename, 
        output_dir, 
        n=10, 
        sigma=0, 
        zscore=True,
        agg_func=None,
        bound0=0,
        bound1=None,
        clobber:bool=False
        ):
    """Project recosntructions to Yerkes Surface"""
    os.makedirs(output_dir, exist_ok=True)
    profile_list = []

    if bound1 is None:
        bound1 = n

    for receptor_volume in receptor_volumes:
        profile_fn = f"{output_dir}/{os.path.basename(receptor_volume).replace('.nii.gz','')}.func.gii"
        profile_list.append(profile_fn)

        if not os.path.exists(profile_fn) or clobber :
            receptor_img = nib.load(receptor_volume)
            receptor = receptor_img.get_fdata()
            starts = np.array(receptor_img.affine[:3,3])
            steps = np.array(receptor_img.affine[[0,1,2],[0,1,2]])
            
            nvol_out =np.zeros(receptor.shape)

            wm_coords, _ = mesh_utils.load_mesh_ext(wm_surf_filename, correct_offset=True)
            gm_coords, _ = mesh_utils.load_mesh_ext(gm_surf_filename, correct_offset=True)

            profiles = np.zeros([wm_coords.shape[0], n])

            d_vtr = ( wm_coords - gm_coords ).astype(np.float64)

            for i, d in enumerate(np.linspace(0, 1, n).astype(np.float64)):
                coords = gm_coords + d * d_vtr
                interp_vol, n_vol = mesh_utils.mesh_to_volume(
                    coords,
                    np.ones(gm_coords.shape[0]),
                    receptor.shape,
                    starts,
                    steps,
                )
                nvol_out[ interp_vol > 0 ] =  1 + d

                surf_receptor_vtr = mesh_utils.volume_filename_to_mesh(coords, receptor_volume, sigma=sigma, zscore=False)
                profiles[:,i] = surf_receptor_vtr

            if zscore :
                profiles = (profiles - profiles.mean()) / profiles.std()

            # Save profiles
            if agg_func is not None:
                profiles = agg_func(profiles[:,bound0:bound1], axis=1).reshape(-1,)

            # save profiles as func.gii with nibabel
            write_gifti(profiles, profile_fn)
            print(wm_surf_filename, profile_fn)

            nib.Nifti1Image(nvol_out, receptor_img.affine).to_filename(f"{output_dir}/n_vol.nii.gz")
    return profile_list



def project_and_plot_surf(
        volumes:list,
        wm_surf_filename:str,
        gm_surf_filename:str,
        output_dir:str,
        n:int = 10,
        sigma:float = 0,
        medial_wall_mask=None,
        zscore:float = True,
        agg_func=None,
        cmap='RdBu_r',
        threshold:float = (.02, .98),
        clobber:bool = False,
        ):

    surface_data_list = project_to_surface(
        volumes,
        wm_surf_filename, 
        gm_surf_filename, 
        output_dir,
        n = n,
        sigma = sigma,
        zscore = zscore,
        agg_func=agg_func,
        clobber = clobber
        )


    for surface_filename in surface_data_list :
        label = os.path.basename(surface_filename).replace('.func.gii','')
        """
        plot_receptor_surf( 
            [surface_filename], 
            wm_surf_filename, 
            output_dir, 
            label=label, 
            medial_wall_mask=medial_wall_mask,
            threshold=threshold,
            cmap=cmap
            )
        """
    return  surface_data_list



def plot_receptor_surf(
        receptor_surfaces, 
        cortex_filename, 
        output_dir, 
        medial_wall_mask=None,
        threshold=[2,98],
        label='', 
        cmap='RdBu_r',
        scale=None,
        clobber:bool=False
        ):
    """Plot receptor profiles on the cortical surface"""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/{label}_surf.png" 
    
    if not os.path.exists(filename) or clobber :
        coords, faces = mesh_utils.load_mesh_ext(cortex_filename)
        
        try :
            ndepths=load_gifti(receptor_surfaces[0]).shape[1]
        except IndexError:
            ndepths=1

        receptor_all = np.array([ load_gifti(fn).reshape(-1,1) for fn in receptor_surfaces ])
        receptor = np.mean( receptor_all,axis=(0,2))

        if scale is not None:
            receptor = scale(receptor)


        pvals = np.ones(receptor.shape[0])
        if medial_wall_mask is not None :
            pvals[medial_wall_mask] = np.nan

        #vmin, vmax = np.nanmax(receptor)*threshold[0], np.nanmax(receptor)*threshold[1]
        vmin, vmax = np.percentile(receptor[~np.isnan(receptor)], threshold)
        print('real threshold', threshold)
        print(f'\tWriting {filename}')
        plot_surf(  coords, 
                    faces, 
                    receptor, 
                    rotate=[90, 270], 
                    filename=filename,
                    pvals=pvals,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    cmap_label=label
                    ) 

        if ndepths > 3 :
            bins = np.rint(np.linspace(0, ndepths,4)).astype(int)
            for i, j in zip(bins[0:-1], bins[1:]):
                receptor = np.mean( np.array([ np.load(fn)[:,i:j] for fn in receptor_surfaces ]),axis=(0,2))

                vmin, vmax = np.nanmax(receptor)*threshold[0], np.nanmax(receptor)*threshold[1]

                filename = f"{output_dir}/surf_profiles_{label}_layer-{i/ndepths}.png" 
                
                plot_surf(  coords, 
                            faces, 
                            receptor, 
                            rotate=[90, 270], 
                            filename=filename,
                            pvals=pvals,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                            cmap_label=label
                            )

def surface_modify_sphere(surface_in, output_dir, radius=1, clobber:bool=False):
    surface_out = output_dir+'/'+os.path.basename(surface_in).replace(".surf.gii","_mod.surf.gii")
    if not os.path.exists(surface_out) or clobber :
        cmd=f'wb_command  -surface-modify-sphere  {surface_in} {radius} {surface_out} -recenter'
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        assert os.path.exists(surface_out), f"Could not find resampled surface {surface_out}"
    return surface_out

def align_surface(
        fixed_sphere:str,
        fixed_mid_cortex:str,
        moving_sphere:str,
        moving_mid_cortex:str,
        output_dir:str,
        radius:float=1,
        clobber:bool=False
):
    clobber=True
    os.makedirs(output_dir, exist_ok=True)
    n_fixed_vertices = nib.load(fixed_sphere).darrays[0].data.shape[0]
    # check if moving_sphere is fs or gii
    moving_sphere = convert_fs_to_gii(moving_sphere, output_dir, clobber=clobber)

    moving_sphere_orig = remesh_surface(moving_sphere, output_dir, n_fixed_vertices , radius=radius, clobber=clobber)
    

    moving_sphere = surface_modify_sphere(moving_sphere, output_dir, radius=radius, clobber=clobber)
    fixed_sphere = surface_modify_sphere(fixed_sphere, output_dir, radius=radius, clobber=clobber)

    moving_mid_cortex = resample_surface(moving_mid_cortex, moving_sphere, moving_sphere_orig, output_dir, n_fixed_vertices,  clobber=clobber)
    #moving_wm_cortex = resample_surface(moving_wm_cortex, moving_sphere_orig, moving_sphere, output_dir, n_fixed_vertices, clobber=clobber)
    #moving_gm_cortex = resample_surface(moving_gm_cortex, moving_sphere_orig, moving_sphere, output_dir, n_fixed_vertices, clobber=clobber)

    fixed_metrics, fixed_metrics_dict =  get_surface_metrics(fixed_mid_cortex, output_dir, clobber=clobber) 
    moving_metrics, moving_metrics_dict = get_surface_metrics(moving_mid_cortex, output_dir, clobber=clobber) 

    # Quality control for surface alignment
    for label, metric in fixed_metrics_dict.items():
        plot_receptor_surf([metric], fixed_mid_cortex, output_dir,  label='fx_'+label, cmap='nipy_spectral')
    
    for label, metric in moving_metrics_dict.items():
        plot_receptor_surf([metric], moving_mid_cortex, output_dir,  label='mv_'+label, cmap='nipy_spectral')
    warped_sphere, warped_data, metrics_rsl_list = msm_align(
        fixed_sphere, 
        fixed_metrics, 
        moving_sphere_orig, 
        moving_metrics,
        output_dir, 
        clobber=clobber
    )

    return warped_sphere, warped_data, metrics_rsl_list, fixed_sphere, moving_sphere


def preprocess_surface(
        fixed_wm_cortex,
        fixed_mid_cortex,
        fixed_gm_cortex, 
        fixed_sphere, 
        moving_wm_cortex,
        moving_mid_cortex,
        moving_gm_cortex, 
        moving_sphere, 
        volume_feature_dict,
        output_dir,
        clobber=False
        ):
    """Preprocess surfaces to get receptor volumes on fixed surface."""
    os.makedirs(output_dir,exist_ok=True)

    warped_sphere, warped_data, metrics_rsl_list, fixed_sphere, moving_sphere = align_surface(
        fixed_sphere, fixed_mid_cortex, moving_sphere, moving_mid_cortex, output_dir, clobber=clobber
        )

    for i, metric in enumerate(metrics_rsl_list):
        plot_receptor_surf([metric], fixed_mid_cortex, output_dir, label=f'warped_fx_{i}',  cmap='nipy_spectral')


    warped_feature_surfaces = {}
    for  label, volumes in volume_feature_dict.items():
        zscore=True
        if 'entropy' in label or 'std' in label:
            zscore=False

        moving_feature_surfaces = project_to_surface(
            volumes, 
            moving_wm_cortex, 
            moving_gm_cortex, 
            output_dir, 
            agg_func=np.mean,
            zscore=zscore,
            bound0=0,
            bound1=7,
            clobber=clobber
            )

        for surface_filename in moving_feature_surfaces : 
            surf_label = os.path.basename(surface_filename).replace('.func.gii','')

            cmap = 'RdBu_r'

            if 'entropy' in surf_label:
                threshold=[10,98]
            else :
                threshold=[2,98]
            print(moving_mid_cortex)
            print(surface_filename)
            plot_receptor_surf([surface_filename], moving_mid_cortex, output_dir, label=f'{surf_label}_mv',  cmap=cmap, threshold=threshold)

        warped_feature_surfaces[label] = msm_resample_list(warped_sphere, fixed_sphere, moving_feature_surfaces, output_dir)

        for surface_filename in warped_feature_surfaces[label] : 
            print(label, surface_filename)
            if 'entropy' in surf_label:
                threshold=[10,98]
            else :
                threshold=[2,98]
            surf_label = os.path.basename(surface_filename).replace('.func.gii','')
            plot_receptor_surf([surface_filename], fixed_mid_cortex, output_dir, label=f'{surf_label}_warp',  cmap=cmap, threshold=threshold)

    spheres_dict = {'warped':warped_sphere, 'fixed':fixed_sphere, 'moving':moving_sphere}

    return warped_feature_surfaces, spheres_dict

