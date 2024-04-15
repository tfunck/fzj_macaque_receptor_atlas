import os
import subprocess
import numpy as np
import nibabel as nib
import shutil


from matplotlib_surface_plotting import plot_surf
from nibabel.freesurfer import read_morph_data, read_geometry
import brainbuilder.utils.mesh_utils as mesh_utils 


def msm_align(
        fixed_sphere, 
        fixed_data, 
        moving_sphere, 
        moving_data, 
        output_dir, 
        clobber=False
        ):
    """Align two surfaces using MSM."""
    fixed_sphere = fix_surf(fixed_sphere, output_dir)
    #moving_sphere = fix_surf(moving_sphere, output_dir)
    
    #moving_sphere = convert_fs_to_gii(moving_sphere, output_dir, clobber=True)
    
    data_out = f'{output_dir}/{os.path.basename(moving_data).replace(".func.gii","_warped")}'

    base = os.path.basename(moving_sphere).replace('.surf.gii','')
    
    sphere_out = f'{data_out}sphere.reg.surf.gii'

    if not os.path.exists(sphere_out)  or clobber :
        cmd = f"msm --inmesh={moving_sphere} --indata={moving_data} --refmesh={fixed_sphere} --refdata={fixed_data} --out={data_out} --levels=4 --verbose=1"
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")

    metrics_rsl_list = msm_resample(sphere_out, fixed_sphere, moving_data, output_dir, clobber=True)
    return sphere_out, data_out, metrics_rsl_list

def msm_resample_list(reg_mesh, target_mesh, labels, output_dir):
    """Apply MSM to labels."""
    labels_rsl_list = []
    for label in labels:
        #input_mesh.sphere.reg.gii output_metric_basename -labels input_metric.func.gii -project target_mesh.surf,gii -adap_bary
        print('Resampling label\n\t', label)
        print('to\n\t', reg_mesh)
        label_rsl_filename = msm_resample(reg_mesh, target_mesh, label, output_dir)[0]

        labels_rsl_list.append(label_rsl_filename) #FIXME

    return labels_rsl_list

def msm_resample(reg_mesh, target_mesh, output_dir, label=None, clobber=False):
    output_label_basename = label.replace('.func','').replace('.gii','') + '_rsl'
    output_label = f'{output_label_basename}.func.gii'
    template_label_rsl_filename = f'{output_label_basename}.func.gii'
    cmd = f"msmresample {reg_mesh} {output_label_basename} -project {target_mesh} -adap_bary"
    if label is not None :
        cmd += f" -labels {label}"
    print(cmd);
    subprocess.run(cmd, shell=True, executable="/bin/bash")

    n = nib.load(target_mesh).get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data.shape[0]

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


def fix_surf(surf_fn, output_dir):
    """Fix surface by using surface-modify-sphere command."""
    base = os.path.basename(surf_fn).replace('.surf.gii','')
    sphere_2_fn = f"{output_dir}/{base}.surf.gii" 
    cmd = f"wb_command -surface-modify-sphere {surf_fn} 100 {sphere_2_fn}"
    subprocess.run(cmd, shell=True, executable="/bin/bash")    
    return sphere_2_fn


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

def resample_surface(surface_in, sphere_fn, sphere_rsl_fn, output_dir, n, clobber=False):
    surface_out = f'{output_dir}/n-{n}_{os.path.basename(surface_in)}'

    if not os.path.exists(surface_out) or clobber:
        cmd = f'wb_command -surface-resample {surface_in} {sphere_fn} {sphere_rsl_fn} BARYCENTRIC {surface_out}'
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    assert os.path.exists(surface_out), f"Could not find resampled surface {surface_out}" 
    return surface_out

def remesh_surface(surface_in,  output_dir, n=10000, clobber=False):
    # run command line 
    base = os.path.basename(surface_in)
    temp_surface_out=f'{output_dir}/n-{n}_temp_{base}.surf.gii'
    surface_out=f'{output_dir}/n-{n}_{base}.surf.gii'
    if not os.path.exists(surface_out) or clobber:
        #cmd = f'mris_remesh --nvert {n} -i {surface_in} -o /tmp/{base} && mris_convert /tmp/{base} {surface_out}'
        cmd = f'mris_remesh --nvert {n} -i {surface_in} -o {temp_surface_out} && wb_command -surface-modify-sphere  {temp_surface_out} 1 {surface_out}'
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/bash")

    assert os.path.exists(surface_out), f"Could not find resampled surface {surface_out}"
    
    return surface_out

def get_fs_prefix(surf_filename):
    prefix=''
    target_prefix=os.path.basename(surf_filename)[0:3]
    return target_prefix
    

def get_surface_curvature(surf_filename, output_dir , clobber=False):
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
        cmd = f"mris_curvature -w  {surf_filename}"
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

def convert_fs_to_gii(input_filename, output_dir, clobber=True):
    """Convert FreeSurfer surface to GIFTI."""
    base = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f'{output_dir}/{base}.surf.gii'

    if not os.path.exists(output_filename) or True:
        ar = read_geometry(input_filename)
        coordsys = nib.gifti.GiftiCoordSystem(dataspace='NIFTI_XFORM_TALAIRACH', xformspace='NIFTI_XFORM_TALAIRACH')
        g = nib.gifti.GiftiImage()
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar[0].astype(np.float32), intent='NIFTI_INTENT_POINTSET',coordsys=coordsys))
        g.add_gifti_data_array(nib.gifti.GiftiDataArray(ar[1].astype(np.int32), intent='NIFTI_INTENT_TRIANGLE', coordsys=None))
        nib.save(g, output_filename)

    return output_filename


def get_surface_metrics(surf_filename, output_dir, clobber=False):
    base = os.path.basename(surf_filename).replace('.surf.gii','')
    output_file = f'{output_dir}/lh.{base}_metrics.func.gii'
    
    fs_sulc_filename, _ = get_surface_sulcal_depth(surf_filename, output_dir,n=3, clobber=clobber)
    fs_curv_filename = get_surface_curvature(surf_filename, output_dir, clobber=clobber)

    sulc_filename = convert_fs_morph_to_gii(fs_sulc_filename, output_dir, clobber=clobber)
    curv_filename = convert_fs_morph_to_gii(fs_curv_filename, output_dir, clobber=clobber)

    if not os.path.exists(output_file) or clobber:
        # merge input metrics
        #wb_command -metric-merge output_name.func.gii -metric metric1.func.gii -metric metric2.func.gii -metric metric3.func.gii
        #cmd = f"wb_command -metric-merge {output_file} -metric {sulc_filename} -metric {curv_filename}"
        cmd = f"wb_command -metric-merge {output_file} -metric {curv_filename} "
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    
    return output_file, {'sulc':sulc_filename, 'curv':curv_filename} 

def write_gifti(array, filename, intent='NIFTI_INTENT_SHAPE'):
    gifti_img = nib.gifti.gifti.GiftiImage()
    gifti_array = nib.gifti.GiftiDataArray(array.astype(np.float32), intent=intent)
    gifti_img.add_gifti_data_array(gifti_array)
    print('Mean:', array.mean(), 'Std:', array.std())
    print('Writing to\n\t', filename)
    gifti_img.to_filename(filename)

def load_gifti(filename):
    return nib.load(filename).darrays[0].data

def project_to_surface(
        receptor_volumes, 
        wm_surf_filename, 
        gm_surf_filename, 
        output_dir, 
        n=10, 
        sigma=0, 
        zscore=True,
        agg_func=None,
        clobber:bool=False
        ):
    """Project recosntructions to Yerkes Surface"""
    os.makedirs(output_dir, exist_ok=True)
    profile_list = []

    for receptor_volume in receptor_volumes:
        profile_fn = f"{output_dir}/{os.path.basename(receptor_volume).replace('.nii.gz','')}.gii"
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
                profiles = agg_func(profiles, axis=1).reshape(-1,)

            # save profiles as func.gii with nibabel
            write_gifti(profiles, profile_fn)

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
        plot_receptor_surf( 
            [surface_filename], 
            wm_surf_filename, 
            output_dir, 
            label=label, 
            medial_wall_mask=medial_wall_mask,
            threshold=threshold,
            cmap=cmap
            )

    return  surface_data_list



def plot_receptor_surf(
        receptor_surfaces, 
        cortex_filename, 
        output_dir, 
        medial_wall_mask=None,
        threshold=[2,98],
        label='', 
        cmap='RdBu_r',
        scale=None
        ):
    """Plot receptor profiles on the cortical surface"""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/surf_profiles_{label}.png" 
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
    vmin, vmax = np.percentile(receptor, threshold)
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

def preprocess_surface(
        fixed_wm_cortex,
        fixed_mid_cortex,
        fixed_gm_cortex, 
        fixed_sphere, 
        moving_wm_cortex,
        moving_mid_cortex,
        moving_gm_cortex, 
        moving_sphere, 
        receptor_volumes,
        output_dir,
        clobber=False
        ):
    """Preprocess surfaces to get receptor volumes on fixed surface."""
    n_fixed_vertices = nib.load(fixed_sphere).darrays[0].data.shape[0]
    clobber = True
    moving_sphere_orig = convert_fs_to_gii(moving_sphere, output_dir, clobber=clobber)
    moving_sphere = remesh_surface(moving_sphere, output_dir, n_fixed_vertices , clobber=clobber)
    moving_mid_cortex = resample_surface(moving_mid_cortex, moving_sphere_orig, moving_sphere, output_dir, n_fixed_vertices, clobber=clobber)

    fixed_metrics, fixed_metrics_dict =  get_surface_metrics(fixed_mid_cortex, output_dir, clobber=clobber) 
    moving_metrics, moving_metrics_dict = get_surface_metrics(moving_mid_cortex, output_dir, clobber=clobber) 

    # Quality control for surface alignment
    
    for label, metric in fixed_metrics_dict.items():
        plot_receptor_surf([metric], fixed_mid_cortex, output_dir,  label='fx_'+label, cmap='viridis')
    
    for label, metric in moving_metrics_dict.items():
        plot_receptor_surf([metric], moving_mid_cortex, output_dir,  label='mv_'+label, cmap='viridis')

    warped_sphere, warped_data, metrics_rsl_list = msm_align(
        fixed_sphere, 
        fixed_metrics, 
        moving_sphere, 
        moving_metrics,
        output_dir, 
        clobber=clobber
    )

    warped_moving_mid_cortex = msm_resample()

    darrays = nib.load(warped_data).darrays
    for i, darray in enumerate(darrays):
        curr_label_rsl_filename = f'{output_dir}/warped_{i}.func.gii'
        if not os.path.exists(curr_label_rsl_filename) or clobber:
            metric = darray.data.astype(np.float32)
            print('Writing to\n\t', curr_label_rsl_filename)
            plot_receptor_surf([metric], warped_moving_mid_cortex, output_dir, label=f'warped_mv_{i}',  cmap='viridis')
    
    for i, metric in enumerate(metrics_rsl_list):
        plot_receptor_surf([metric], fixed_mid_cortex, output_dir, label=f'warped_fx_{i}',  cmap='viridis')
    exit(0)

    moving_receptor_surfaces = project_to_surface(
        receptor_volumes, 
        moving_wm_cortex, 
        moving_gm_cortex, 
        output_dir, 
        agg_func=np.mean,
        clobber=True
        )
    
    warped_receptor_surfaces = msm_resample_list(warped_sphere, fixed_sphere, moving_receptor_surfaces, output_dir)
    exit(0)

    return warped_receptor_surfaces

