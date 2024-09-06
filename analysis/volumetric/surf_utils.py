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
        fixed_mask,
        moving_sphere, 
        moving_data, 
        moving_mask,
        output_dir, 
        levels=3,
        trans=None,
        verbose=True,
        clobber=False
        ):
    """Align two surfaces using MSM."""
    os.makedirs(output_dir,exist_ok=True)
    TFM_STR='transformed_and_reprojected.func.gii'

    data_out = f"{output_dir}/{'.'.join(os.path.basename(moving_data).split('.')[0:-2])}_warped_"
    
    data_out_rsl = f'{data_out}{TFM_STR}'

    out_sphere = f'{data_out}sphere.reg.surf.gii'

    if not os.path.exists(out_sphere) or not os.path.exists(data_out_rsl) or clobber :
        cmd = f"msm --inmesh={moving_sphere} --indata={moving_data} --inweight={moving_mask} "
        cmd += f"  --refmesh={fixed_sphere} --refdata={fixed_data} --refweight={fixed_mask} " 
        if trans is not None :
            cmd += f' --trans={trans}'

        cmd += f" --out={data_out} --levels={levels} --verbose=0"       

        if verbose :
            print()
            print('Inputs:')
            print(f'\tMesh {moving_sphere}')
            print(f'\tMask: {moving_mask}')
            print(f'\tData: {moving_data}')
            print('Reference:')
            print(f'\tMesh {fixed_sphere}')
            print(f'\tMask: {fixed_mask}')
            print(f'\tData: {fixed_data}')
            print(f'\tOptions:')
            print(f'\ttrans: {trans}')
            print('Out')
            print(f'\tOutput Data: {data_out_rsl}')
            print(f'\tOutput Sphere: {out_sphere}')
            print()
            print(cmd);
        subprocess.run(cmd, shell=True, executable="/bin/bash")
        assert os.path.exists(data_out_rsl)
        print('\nwb_view', fixed_sphere, fixed_data, data_out_rsl, '\n' )

        plot_receptor_surf(
            [fixed_data], fixed_sphere, output_dir, label='fx_orig', cmap='nipy_spectral', clobber=True
        )
        plot_receptor_surf(
            [data_out_rsl], fixed_sphere, output_dir, label='mv_rsl', cmap='nipy_spectral', clobber=True
        )
    

    return out_sphere, data_out_rsl

def msm_resample_list(rsl_mesh, fixed_mesh, labels, output_dir, clobber=False):
    """Apply MSM to labels."""
    labels_rsl_list = []
    for i, label in enumerate(labels):
        if i % 50 == 0 : 
            print(f'\nCompleted: {(i/len(labels))*100:.2f}%\n')
        label_rsl_filename = msm_resample(rsl_mesh, fixed_mesh, label, output_dir=output_dir, clobber=clobber)
        labels_rsl_list.append(label_rsl_filename) #FIXME

    return labels_rsl_list

def msm_resample(rsl_mesh, fixed_mesh, label=None, output_dir:str='', write_darrays:bool=False, clobber=False):
    output_label_basename = os.path.basename(label).replace('.func','').replace('.gii','') + '_rsl'
    if output_dir == '':
        output_dir = os.path.dirname(label)
    output_label = f'{output_dir}/{output_label_basename}'
    output_label_ext = f'{output_label}.func.gii'
    template_label_rsl_filename = f'{output_label_basename}.func.gii'

    cmd = f"msmresample {rsl_mesh} {output_label} -project {fixed_mesh} -adap_bary"

    if not os.path.exists(output_label_ext) or clobber:
        if label is not None :
            cmd += f" -labels {label}"
        print(cmd);
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    else :
        pass

    return output_label_ext
    n = nib.load(fixed_mesh).get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data.shape[0]

    if write_darrays:
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


def get_surface_sulcal_depth(surf_filename, output_dir, n=10, dist=0.1, clobber=False):
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
        cmd = f"mris_inflate  -dist {dist} -n {n} -sulc {sulc_suffix} {surf_filename} {inflated_filename}"
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
    surface_out=f'{output_dir}/n-{n}_{base}'
    if not os.path.exists(surface_out) or clobber:
        n_moving_vertices = load_mesh_ext(surface_in)[0].shape[0]
        #cmd = f'mris_remesh --nvert {n} -i {surface_in} -o /tmp/{base} && mris_convert /tmp/{base} {surface_out}'
        #not sure about this->cmd = f'mris_remesh --nvert {n} -i {surface_in} -o {temp_surface_out} && wb_command  -surface-modify-sphere  {temp_surface_out} {radius} {surface_out} -recenter'
        cmd = f'wb_command  -surface-modify-sphere  {surface_in} {radius} {surface_out} -recenter'
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

def convert_fs_morph_to_gii(input_filename, mask_filename, output_dir, clobber=False)  :
    """Convert FreeSurfer surface to GIFTI."""
    base = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f'{output_dir}/{base}_sulc.shape.gii'

    if not os.path.exists(output_filename) or clobber:
        ar = read_morph_data(input_filename).astype(np.float32)

        mask = nib.load(mask_filename).darrays[0].data
        #ar[mask==0] = -1.5*np.abs(np.min(ar))

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



def create_xyz_axis_file(surf_filename, mask_filename, output_dir, axis, clobber=False):
    """Create a gifti file with the product of z and y coordinates as the data."""
    base = os.path.basename(surf_filename).replace('.surf.gii','')
    output_filename = f'{output_dir}/{base}_axis-{axis}.func.gii'

    if not os.path.exists(output_filename) or clobber:
        mask = nib.load(mask_filename).darrays[0].data
        coords, _ = mesh_utils.load_mesh_ext(surf_filename)
        zyaxis = coords[:,axis] 
        zyaxis = (zyaxis - zyaxis.min()) / (zyaxis.max() - zyaxis.min())
        zyaxis[mask==0] = -3 * np.abs(np.min(zyaxis))
        print('MINIMUM', -3 * np.abs(np.min(zyaxis)))
        write_gifti(zyaxis, output_filename)
    return output_filename

def normalize_func_gii(fn):
    """read a .func.gii file and z-score it, then save as out_fn """
    ar = nib.load(fn).darrays[0].data
    print('MEAN', np.mean(ar), 'STD', np.std(ar));
    ar = (ar - np.mean(ar)) / np.std(ar)
    
    data = nib.gifti.GiftiDataArray(data=ar, intent='NIFTI_INTENT_SHAPE', datatype='NIFTI_TYPE_FLOAT32')
    img = nib.gifti.GiftiImage(darrays=[data])

    img.to_filename(fn)


def mask_func_gii(fn, mask_fn):
    """Mask a .func.gii file with a mask file."""
    mask = nib.load(mask_fn).darrays[0].data
    ar = nib.load(fn).darrays[0].data
    ar[mask==0] = -1.5*np.abs(np.min(ar))
    data = nib.gifti.GiftiDataArray(data=ar, intent='NIFTI_INTENT_SHAPE', datatype='NIFTI_TYPE_FLOAT32')
    img = nib.gifti.GiftiImage(darrays=[data])
    img.to_filename(fn)

def get_surface_metrics(surf_filename, mask_filename, output_dir, metrics=['sulc'], n_sulc=10, dist=0.1, n_curv=100, clobber=False):
    base = os.path.basename(surf_filename).replace('.surf.gii','')
    output_file = f'{output_dir}/lh.{base}_metrics.func.gii'

    metrics_dict = {}

    if 'sulc' in metrics : 
        fs_sulc_filename, _ = get_surface_sulcal_depth(surf_filename, output_dir, n=n_sulc, dist=dist, clobber=clobber)
        sulc_filename = convert_fs_morph_to_gii(fs_sulc_filename, mask_filename, output_dir, clobber=clobber)
        mask_func_gii(sulc_filename, mask_filename)
        metrics_dict['sulc'] = sulc_filename
    
    if 'curv' in metrics:
        fs_curv_filename = get_surface_curvature(surf_filename, output_dir, n=n_curv, clobber=clobber)
        curv_filename = convert_fs_morph_to_gii(fs_curv_filename, mask_filename, output_dir, clobber=clobber)
        mask_func_gii(curv_filename, mask_filename)
        metrics_dict['curv'] = curv_filename

    if 'mask' in metrics :
        metrics_dict['mask'] = mask_filename 

    if 'x' in metrics :
        axis_filename = create_xyz_axis_file(surf_filename, mask_filename, output_dir, 0, clobber=clobber)
        metrics_dict['x'] = axis_filename

    if 'y' in metrics :
        axis_filename = create_xyz_axis_file(surf_filename,  mask_filename, output_dir, 1, clobber=clobber)
        metrics_dict['y'] = axis_filename

    if 'z' in metrics :
        axis_filename = create_xyz_axis_file(surf_filename,  mask_filename, output_dir, 2, clobber=clobber)
        metrics_dict['z'] = axis_filename

    if not os.path.exists(output_file) or clobber:
        # merge input metrics
        metric_string=''
        for metric in metrics_dict.values() :
            metric_string += f' -metric {metric} '

        cmd = f"wb_command -metric-merge {output_file} {metric_string} "

        subprocess.run(cmd, shell=True, executable="/bin/bash")
    
    return output_file 

def write_gifti(array, filename, intent='NIFTI_INTENT_NORMAL'):
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
        invert=0,
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
        print(receptor_volume); 
        profile_fn = f"{output_dir}/{os.path.basename(receptor_volume).replace('.nii.gz','')}.func.gii"
        profile_list.append(profile_fn)

        if not os.path.exists(profile_fn) or clobber :
            receptor_img = nib.load(receptor_volume)
            print(receptor_volume)
            receptor = receptor_img.get_fdata()

            starts = np.array(receptor_img.affine[:3,3])
            steps = np.array(receptor_img.affine[[0,1,2],[0,1,2]])
            
            nvol_out = np.zeros(receptor.shape)

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
                # replace nan with 0 
                surf_receptor_vtr[np.isnan(surf_receptor_vtr)] = 0
                profiles[:,i] = surf_receptor_vtr

            if zscore :
                profiles = (profiles - profiles.mean()) / profiles.std()

            # Save profiles
            if agg_func is not None:
                profiles = agg_func(profiles[:,bound0:bound1], axis=1).reshape(-1,)

            if invert == 1:
                profiles = -profiles
            elif invert == 2:
                print('Inverting')
                profiles = (~ profiles.astype(np.bool_)).astype(np.uint8)

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
        invert=0,
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
        invert=invert,
        agg_func=agg_func,
        clobber = clobber
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
        scale=None,
        clobber:bool=False
        ):
    """Plot receptor profiles on the cortical surface"""
    os.makedirs(output_dir, exist_ok=True)
    print('Receptor surfaces', receptor_surfaces)

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
        fixed_mask:str,
        moving_sphere:str,
        moving_mid_cortex:str,
        moving_mask:str,
        output_dir:str,
        radius:float=1,
        visualize:bool=False,
        clobber:bool=False
):
    """Align two surfaces using MSM."""
    output_init_dir = output_dir+'/init/'
    output_mid_dir = output_dir+'/mid/'
    output_final_dir = output_dir+'/final/'
    os.makedirs(output_final_dir, exist_ok=True)
    os.makedirs(output_mid_dir, exist_ok=True)
    os.makedirs(output_init_dir, exist_ok=True)

    # check if moving_sphere is fs or gii
    moving_sphere = convert_fs_to_gii(moving_sphere, output_dir, clobber=clobber)

    n_fixed_vertices = nib.load(fixed_sphere).darrays[0].data.shape[0]
    moving_sphere_orig = convert_fs_to_gii(moving_sphere, output_dir, clobber=clobber)
    moving_sphere = remesh_surface(moving_sphere, output_dir, n_fixed_vertices , clobber=clobber)
    moving_mid_cortex = resample_surface(moving_mid_cortex, moving_sphere_orig, moving_sphere, output_dir, n_fixed_vertices, clobber=clobber)

    #moving_sphere_orig = remesh_surface(moving_sphere, output_dir, radius=radius, clobber=clobber)
    moving_sphere = surface_modify_sphere(moving_sphere, output_dir, radius=radius, clobber=clobber)
    fixed_sphere = surface_modify_sphere(fixed_sphere, output_dir, radius=radius, clobber=clobber)

    init_moving_metrics = get_surface_metrics(moving_mid_cortex, moving_mask, output_mid_dir, metrics=['x', 'y', 'z'], clobber=clobber )
    init_fixed_metrics= get_surface_metrics(fixed_mid_cortex, fixed_mask, output_mid_dir, metrics=['x', 'y', 'z'], clobber=clobber )

    moving_metrics = get_surface_metrics(moving_mid_cortex, moving_mask, output_final_dir, [ 'sulc'], n_sulc=10, n_curv=30, clobber=clobber) 
    print('\nwb_view', moving_sphere, moving_mid_cortex, moving_metrics, '\n' ); 
    fixed_metrics= get_surface_metrics(fixed_mid_cortex, fixed_mask, output_final_dir, [ 'sulc'], n_sulc=10, n_curv=10, clobber=clobber) 
    print('\nwb_view', fixed_sphere, fixed_mid_cortex, fixed_metrics, '\n' );

    # Quality control for surface alignment
    #for label, metric in fixed_metrics_dict.items():
    #    plot_receptor_surf([metric], fixed_mid_cortex, output_dir, label='fx_'+label, cmap='nipy_spectral', clobber=clobber)
    #    plot_receptor_surf([metric], fixed_sphere, output_dir, label='fx_sphere_'+label, cmap='nipy_spectral', clobber=clobber)
    
    #for label, metric in moving_metrics_dict.items():
    #    plot_receptor_surf([metric], moving_mid_cortex, output_dir, label='mv_'+label, cmap='nipy_spectral', clobber=clobber)
    #    plot_receptor_surf([metric], moving_sphere, output_dir, label='mv_sphere_'+label, cmap='nipy_spectral', clobber=clobber)
    warped_sphere_init, data_rsl_init = msm_align(
        fixed_sphere,
        init_fixed_metrics, 
        fixed_mask,
        moving_sphere, 
        init_moving_metrics,
        moving_mask,
        output_init_dir, 
        levels=2,
        clobber=clobber
    )

    warped_sphere, data_rsl_final  = msm_align(
        fixed_sphere,
        fixed_metrics, 
        fixed_mask,
        moving_sphere, 
        moving_metrics,
        moving_mask,
        output_final_dir, 
        levels=2,
        trans=warped_sphere_init,
        clobber=clobber
    )
    if visualize:
        subprocess.run(f'wb_view {fixed_mid_cortex} {fixed_metrics} {data_rsl_final}', shell=True, executable="/bin/bash")

    return warped_sphere, fixed_sphere, moving_sphere


def preprocess_surface(
        fixed_mid_cortex,
        fixed_sphere, 
        fixed_mask,
        moving_wm_cortex,
        moving_mid_cortex,
        moving_gm_cortex, 
        moving_sphere,
        moving_mask, 
        volume_feature_dict,
        output_dir,
        clobber=False
        ):
    """Preprocess surfaces to get receptor volumes on fixed surface."""
    os.makedirs(output_dir,exist_ok=True)
    warped_sphere, fixed_sphere, moving_sphere = align_surface(
        fixed_sphere, fixed_mid_cortex, fixed_mask, 
        moving_sphere, moving_mid_cortex, moving_mask, 
        output_dir, clobber=clobber
        )

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

