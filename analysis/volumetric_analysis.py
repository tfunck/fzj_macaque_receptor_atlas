
import argparse
import nibabel as nib
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess

import brainbuilder.utils.mesh_utils as mesh_utils 

from volumetric.surf_utils import project_to_surface, project_and_plot_surf, load_gifti, write_gifti
from surfaces.surface_diff import surface_diff

from nipype.interfaces.ants import ApplyTransforms

from matplotlib_surface_plotting import plot_surf
from skimage.transform import resize
from scipy.stats import spearmanr, pearsonr

from volumetric.entropy_analysis import entropy_analaysis
from volumetric.volumetric_ratio_analysis import ratio_analysis
from volumetric.volumetric_gradient_analysis import volumetric_gradient_analysis
from volumetric.surf_utils import preprocess_surface
from surface_analysis import surface_analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
import utils


def resize_mask_to_receptor_volume(mask_file, receptor_file, output_dir, order=0):
    """Resize mask to the same dimensions as the receptor volume."""
    base = os.path.basename(mask_file).replace('.nii.gz','_rsl.nii.gz')
    mask_rsl_file = f"{output_dir}/{base}"

    if not os.path.exists(mask_rsl_file):
        mask = nib.load(mask_file)
        mask_vol = mask.get_fdata()

        receptor = nib.load(receptor_file)

        assert np.unique(mask_vol).size > 1, 'Mask is empty'
        print(receptor.shape, mask_vol.min(), mask_vol.max())
        mask_rsl = resize(mask_vol.astype(np.float32), receptor.shape, order=order)

        assert np.unique(mask_rsl).size > 1, 'Resized mask is empty'

        nib.Nifti1Image(mask_rsl, receptor.affine).to_filename(mask_rsl_file)

    return mask_rsl_file


def t1t2_analysis(mask_rsl_file, hist_volumes, t1t2_filename, output_dir):
    import nibabel as nib
    import matplotlib.pyplot as plt
    
    
    mask_vol = nib.load(mask_rsl_file).get_fdata()

    myelin_filename = utils.get_files_from_list(hist_volumes, ['myelin'])[0]

    myelin_img = nib.load(myelin_filename)
    myelin_vol = myelin_img.get_fdata()
    
    t1t2_rsl_file = resize_mask_to_receptor_volume( t1t2_filename, myelin_filename, output_dir, order=3)
    t1t2_vol = nib.load(t1t2_rsl_file).get_fdata()

    idx = (mask_vol > 0) & (t1t2_vol > 0) & (myelin_vol > 0)

    x=myelin_vol[idx]
    y=t1t2_vol[idx]   

    plt.scatter(x,y)
    plt.savefig(f'{output_dir}/t1t2_vs_myelin.png')

    # fit linear regression model for x and y
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x.reshape(-1,1), y)
    # get model residuals
    residuals = y - model.predict(x.reshape(-1,1))

    res_vol = np.zeros_like(myelin_vol)
    res_vol[idx] = residuals

    nib.Nifti1Image(res_vol, myelin_img.affine).to_filename(f'{output_dir}/t1t2_residuals.nii.gz')
    
def plot_pairwise_correlation(comparison_volumes, mask_vol, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    mask_vol = nib.load(mask_vol).get_fdata()

    n = len(comparison_volumes)
    corr_matrix = np.eye(n)

    #   0 1 2 3
    # 0 1
    # 1   1
    # 2     1
    # 3       1
    plt.cla(); plt.clf(); plt.close()

    for i, x_filename in enumerate(comparison_volumes[0:-1]):
        x = nib.load(x_filename).get_fdata()

        for j, y_filename in enumerate(comparison_volumes[i+1:]):
            y = nib.load(y_filename).get_fdata()
            k=j+i+1
            print(i,k)

            xmin = np.min(x) 
            ymin = np.min(y) 

            idx = (mask_vol > 0) & (x > xmin) & (y > ymin)
            
            x0 = x[idx]
            y0 = y[idx]

            r, p = spearmanr(x0, y0)
            #r, p = pearsonr(x, y)
            xlabel=os.path.basename(x_filename).replace('.nii.gz','')
            ylabel=os.path.basename(y_filename).replace('.nii.gz','')
            print(f'{xlabel} vs {ylabel} : r={r}, p={p}')
            corr_matrix[i,k] = r
            corr_matrix[k,i] = r
            plt.scatter(x0,y0, alpha=0.25)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(f'{output_dir}/{xlabel}_vs_{ylabel}.png')
            plt.cla(); plt.clf(); plt.close()


    plt.cla(); plt.clf(); plt.close()
    plt.imshow(corr_matrix, cmap='viridis')
    plt.colorbar()
    plt.savefig(f'{output_dir}/correlation_matrix.png')

    np.save(f'{output_dir}/correlation_matrix.npy', corr_matrix)

def vif_analysis(comparison_volumes, mask_filenmae, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n=len(comparison_volumes)

    mask_vol= nib.load(mask_filenmae).get_fdata()

    idx = np.where(mask_vol > 0)
    features= np.zeros((idx[0].size, n))
    df = pd.DataFrame({'volume': [os.path.basename(fn).replace('.nii.gz','') for fn in comparison_volumes], 'vif':np.zeros(n) })
    for i, filename in enumerate(comparison_volumes):
        x = nib.load(filename).get_fdata()
        features[:,i] = x[idx]

    print(np.mean(features, axis=0))

    df['vif'] = [variance_inflation_factor(features, i) for i in range(n)]
    print(df)
    df.to_csv(f'{output_dir}/vif.csv', index=False)


            
def volumetric_distance_gradients(receptor_volumes, diff_gradient_dir):
    """Calculate the differential gradient of the receptor volumes."""
    os.makedirs(diff_gradient_dir, exist_ok=True)

    for filename in receptor_volumes :
        receptor_img = nib.load(filename)
        receptor_data = receptor_img.get_fdata()
    
        gradient = np.sqrt(np.sum(np.power(np.gradient(receptor_data),2), axis=0))

        nib.Nifti1Image(gradient, receptor_img.affine).to_filename(f'{diff_gradient_dir}/{os.path.basename(filename)}')


def entropy_vs_gradient(
        gradient_filename, 
        entropy_filename, 
        wm_surf_filename, 
        gm_surf_filename, 
        output_dir,
        ndepths=10):
     
    [entropy_surf_filename, gradient_surf_filename] = project_to_surface(
        [entropy_filename, gradient_filename], 
        wm_surf_filename, 
        gm_surf_filename, 
        profiles_dir,
        n = ndepths,
        sigma = 0,
        zscore = False,
        clobber = True
        )

    entropy_surf = load_gifti(entropy_surf_filename)
    gradient_surf = load_gifti(gradient_surf_filename)

    bins = np.rint(np.linspace(0, ndepths,4)).astype(int) # split into 3 layers, 4 bins
    for i, j in zip(bins[0:-1], bins[1:]):
        laminar_entropy = entropy_surf[ :,i:j]
        laminar_gradient = gradient_surf[ :,i:j]


        plt.scatter(laminar_entropy, laminar_gradient, alpha=0.01)
        plt.xlabel('Entropy')
        plt.ylabel('Gradient')
        plt.savefig(f'{output_dir}/entropy_vs_gradient_{j}.png')
        plt.cla()
        plt.clf()



def align(
        fixed_filename, 
        moving_filename, 
        moving_mask_filename, 
        files_to_transform, 
        output_dir,
        moving_prefix='MEBRAINS', 
        fixed_prefix='Yerkes',
        clobber=False
        ):
    import nipype.interfaces.ants as ants

    os.makedirs(output_dir, exist_ok=True)

    prefix='{0}/{1}_{2}_'.format(output_dir, moving_prefix, fixed_prefix)
    fwd_filename = f'{prefix}Composite.h5'
    inv_filename = f'{prefix}InverseComposite.h5'
    output_files = [ f'{output_dir}/{os.path.basename(filename)}' for filename in files_to_transform ]

    if  not os.path.exists(fwd_filename) or not os.path.exists(inv_filename) or clobber:
        print('Registering')
        print(fwd_filename)
        print(inv_filename)
        exit(0)
        moving_prefix = os.path.basename(moving_filename).replace('.nii.gz','') 
        fixed_prefix = os.path.basename(fixed_filename).replace('.nii.gz','')
        
        reg0 = ants.registration.Registration(
            fixed_image=fixed_filename, 
            moving_image=moving_filename, 
            output_transform_prefix='{0}_affine_'.format(prefix),
            output_warped_image='{0}_linear.nii.gz'.format(prefix),
            initial_moving_transform_com=0,
            winsorize_lower_quantile=0.05,
            winsorize_upper_quantile=0.95,
            interpolation='Linear',
            use_histogram_matching=[True],
            transforms=['Affine'],
            transform_parameters=[(0.1,)],
            metric=['MI'],
            metric_weight=[1],
            radius_or_number_of_bins=[32],
            sampling_strategy=['Random'],
            sampling_percentage=[0.5],
            number_of_iterations=[[1600, 800, 400, 200]],
            convergence_threshold=[1e-6],
            convergence_window_size=[10],
            shrink_factors=[[8, 4, 2, 1]],
            smoothing_sigmas=[[3, 2, 1, 0]],
            sigma_units=['vox'],
            verbose=True
            )
        
        reg0.run()
        reg0.outputs = reg0._list_outputs()

        reg = ants.registration.Registration(
            fixed_image=fixed_filename, 
            moving_image=moving_filename, 
            initial_moving_transform = reg0.outputs['composite_transform'],
            write_composite_transform=True,
            verbose=True, 
            dimension=3,
            float=False,
            output_transform_prefix='{0}'.format(prefix),
            output_warped_image='{0}linear+SyN.nii.gz'.format(prefix),
            initial_moving_transform_com=0,
            winsorize_lower_quantile=0.05,
            winsorize_upper_quantile=0.95,
            interpolation='Linear',
            use_histogram_matching=[True],
            transforms=['SyN'],
            transform_parameters=[(0.1,)],
            metric=['CC'],
            metric_weight=[1],
            radius_or_number_of_bins=[4],
            sampling_strategy=['Random'],
            sampling_percentage=[0.5],
            number_of_iterations=[[800, 400, 200, 100]],
            convergence_threshold=[1e-6],
            convergence_window_size=[10],
            shrink_factors=[[8, 4, 2, 1]],
            smoothing_sigmas=[[3, 2, 1, 0]],
            sigma_units=['vox']
            )

        reg.run()
        reg.outputs = reg._list_outputs()

    for output_filename, filename in zip(output_files, files_to_transform):
        if not os.path.exists(output_filename) or clobber :
            tfm = ApplyTransforms(
                input_image=filename,
                reference_image=fixed_filename,
                output_image=output_filename,
                transforms=fwd_filename,
                interpolation='Linear',
            )
            tfm.run()


    return output_files

def apply_surface_atlas(surf_files, atlas_file, output_dir, descriptor):
    # load surface atlas
    atlas = nib.load(atlas_file).darrays[0].data

    df = pd.DataFrame({})

    n=atlas.shape[0]

    for surf_file in surf_files:
        values = load_gifti(surf_file).reshape(-1,)
        print(surf_file)
        print('Mean:', np.mean(values), 'Values:', np.std(values) )
        #n = np.bincount(atlas)
        #totals = np.bincount(atlas, weights=surf)
        #mean = totals / n
        print(atlas.shape, values.shape)
        
        label= os.path.basename(surf_file).replace('.nii.gz','').replace('macaque_','').replace('.npy','').replace('.gii','')
        row = pd.DataFrame({'receptor':[label]*n, 'label':atlas, 'density':values})
        df = pd.concat([df,row])
    
    # remove 0 labels
    df = df.loc[df['label'] > 0]

    atlas_coding = { 6: 'visual', 	3: 'auditory', 2: 'somatosensory', 4: 'limbic', 5: 'dorsal attention', 1: 'default mode network'}
    x_coding = { 6: 1, 	3: 2, 2: 3, 4: 4, 5: 5, 1: 6}
    df['x'] = df['label'].map(x_coding)
    df['atlas'] = df['label'].map(atlas_coding)
    #reindex 
    df = df.reset_index(drop=True)

    print(df)
    plt.figure(figsize=(10,10))
    #sns.lineplot(x='atlas', y='density', hue='receptor', data=df)
    sns.color_palette("Set2")
    g = sns.lineplot(data=df, x="x", y="density", hue="receptor", alpha=0.9)
    g.set_xticks(range(1,7))
    g.set_xticklabels(['visual', 'auditory', 'somatosensory', 'limbic', 'dorsal attention', 'default mode network'])
    plt.savefig(f'{output_dir}/atlas_{descriptor}.png')

# get the directory of this file
current_file_path = os.path.abspath(__file__)
wrk_dir = os.path.dirname(current_file_path)+'/../'

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Volumetric Gradient Analysis')
    parser.add_argument('-m', dest='mask_file', default='data/volumes/MEBRAINS_segmentation_NEW_gm_left.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-l', dest='label_file', default='data/volumes/MEBRAINS_pseudo-segmentation-0_gm_left.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-e' , dest='mebrains_filename', default='data/volumes/MEBRAINS_T1_masked.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-y' , dest='yerkes_template_filename', default='data/volumes/MacaqueYerkes19_v1.2_AverageT1w_restore_masked.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-a' , dest='yerkes_atlas_filename', default='data/surfaces/L.BezginTo7Networks.32k_fs_LR.label.gii', type=str, help='Path to mask file')
    parser.add_argument('-i', dest='input_dir', type=str, default='data/reconstruction/', help='Path to receptor volumes')
    parser.add_argument('-o', dest='output_dir', type=str, default=f'{wrk_dir}/outputs/volumetric', help='Path to output directory')
    parser.add_argument('-n', dest='n', default=10000, type=int, help='Number of random voxels to sample')
    parser.add_argument('-w', dest='medial_wall_filename', default=f'{wrk_dir}/data/volumes/medial_wall_mask_morph_0.4mm.nii.gz', type=str, help='Path to medial wall mask')
    # MEBRAINs surfaces
    parser.add_argument('--wm-surf', dest='wm_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/lh.MEBRAINS_0.5mm_1.0.surf.gii', help='Path to MEBRAINS white matter surface')
    parser.add_argument('--mid-surf', dest='mid_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/lh.MEBRAINS_0.5mm_0.5.surf.gii', help='Path to MEBRAINS white matter surface')
    parser.add_argument('--gm-surf', dest='gm_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/lh.MEBRAINS_0.5mm_0.0.surf.gii', help='Path to MEBRAINS pial matter surface')
    parser.add_argument('--sphere-surf', dest='sphere_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/lh.MEBRAINS_0.125mm_0.5_0.125mm_0.5.sphere', help='Path to Yerkes pial matter surface')
    # Yerkes surfaces
    parser.add_argument('--y-gm-surf', dest='yerkes_gm_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19.L.pial.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-mid-surf', dest='yerkes_mid_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19_v1.2.L.midthickness.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-wm-surf', dest='yerkes_wm_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19.L.white.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-sphere-surf', dest='yerkes_sphere_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19_v1.2.L.sphere.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-sulc', dest='yerkes_sulc_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19_v1.2.sulc.32k_fs_LR.dscalar.nii', help='Path to Yerkes pial matter surface')
    #parser.add_argument('--gm-infl', dest='infl_surf_filename', type=str, default='data/surfaces/MacaqueYerkes19.L.inflated.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    clobber=False

    args = parser.parse_args()
    t1t2_filename = 'data/volumes/MEBRAINS_T1T2_masked.nii.gz'

    os.makedirs(args.output_dir, exist_ok=True)
    receptor_dir = f"{args.input_dir}/receptor/"
    hist_dir = f"{args.input_dir}/hist/"
    entropy_dir = args.output_dir +'/entropy'
    ratio_dir = f'{args.output_dir}/ratios/'
    corr_dir = f'{args.output_dir}/correlations/'
    profiles_dir = f'{args.output_dir}/profiles/'
    yerkes_profiles_dir = f'{args.output_dir}/profiles_yerkes/'
    align_dir = f"{args.output_dir}/align/"
    diff_gradient_dir = f"{args.output_dir}/diff_gradients/"
    grad_dir = f"{args.output_dir}/gradients/"    

    medial_wall_mask_filename = project_and_plot_surf(
        [args.medial_wall_filename], args.wm_surf_filename, args.gm_surf_filename, profiles_dir, clobber=True, zscore=False, agg_func=np.max
        )[0]
    medial_wall_mask = load_gifti(medial_wall_mask_filename).astype(bool).reshape(-1,)
    receptor_volumes = glob.glob(f'{receptor_dir}/*nii.gz')
    hist_volumes = glob.glob(f'{hist_dir}/*nii.gz')

    ### Resize mask to receptor volume
    mask_rsl_file = resize_mask_to_receptor_volume(args.mask_file, receptor_volumes[0], args.output_dir)
    
    if False :
        yerkes_receptor_surfaces = preprocess_surface(
            args.yerkes_wm_surf_filename,
            args.yerkes_mid_surf_filename,
            args.yerkes_gm_surf_filename, 
            args.yerkes_sphere_surf_filename,
            args.wm_surf_filename, 
            args.mid_surf_filename,
            args.gm_surf_filename,
            args.sphere_surf_filename,
            receptor_volumes,
            align_dir
            )
        
    yerkes_receptor_volumes = align(
        args.yerkes_template_filename, args.mebrains_filename, mask_rsl_file, receptor_volumes, align_dir, clobber=False
        )

    yerkes_receptor_surfaces = project_to_surface( 
        yerkes_receptor_volumes, args.yerkes_wm_surf_filename, args.yerkes_gm_surf_filename, profiles_dir, agg_func=np.mean, clobber=False 
        )

    apply_surface_atlas(yerkes_receptor_surfaces, args.yerkes_atlas_filename, args.output_dir, 'receptor')

    #t1t2_analysis(mask_rsl_file, hist_volumes, t1t2_filename, args.output_dir)
    ### Resize MEBRAINS T1/T2 to receptor volume
    #t1t2_rsl_filename = resize_mask_to_receptor_volume( t1t2_filename, receptor_volumes[0], args.output_dir, order=3)

    ### Calculate entropy of receptor volumes
    entropy_file, std_file = entropy_analaysis(mask_rsl_file, receptor_volumes, entropy_dir)

    yerkes_complexity_volumes = align(
        args.yerkes_template_filename, args.mebrains_filename, mask_rsl_file, [entropy_file, std_file], align_dir
        )
    yerkes_entropy_volume = yerkes_complexity_volumes[0]
    yerkes_std_volume = yerkes_complexity_volumes[1]
                                        
    yerkes_entropy_surfaces = project_to_surface( [yerkes_entropy_volume],
                                                     args.yerkes_wm_surf_filename, 
                                                     args.yerkes_gm_surf_filename,
                                                     yerkes_profiles_dir, 
                                                     agg_func=np.mean,
                                                     clobber=True ,
                                                     zscore=False
                                                     )

    yerkes_std_surfaces = project_to_surface( [yerkes_std_volume],
                                                     args.yerkes_wm_surf_filename, 
                                                     args.yerkes_gm_surf_filename,
                                                     yerkes_profiles_dir, 
                                                     agg_func=np.mean,
                                                     clobber=True,
                                                     zscore=False
                                                     )

    apply_surface_atlas(yerkes_entropy_surfaces, args.yerkes_atlas_filename, args.output_dir, 'entropy')
    apply_surface_atlas(yerkes_std_surfaces, args.yerkes_atlas_filename, args.output_dir, 'std')

    ### Calculate PCA gradients
    #gradient_volumes = volumetric_gradient_analysis(mask_rsl_file, receptor_volumes, grad_dir, approach='pca', n=args.n)
    #gradient_volumes = volumetric_gradient_analysis(mask_rsl_file, receptor_volumes, grad_dir, approach='le', n=args.n)
    #gradient_volumes = volumetric_gradient_analysis(mask_rsl_file, receptor_volumes, grad_dir, approach='dm', n=args.n)

    ### Calculate ratios between receptor volumes
    ratio_dict = ratio_analysis(receptor_volumes, mask_rsl_file, ratio_dir, clobber=True )
    summary_volumes = [fn for fn,lab in ratio_dict.items() if lab in ['Inh', 'Glutamate', 'Acetylcholine', 'Noradrenaline', 'Serotonin', 'Dopamine']]
    vif_analysis(summary_volumes, mask_rsl_file, corr_dir)
    plot_pairwise_correlation(summary_volumes, mask_rsl_file, corr_dir)
    #vif_analysis(gradient_volumes, mask_rsl_file, corr_dir)
    #vif_analysis(receptor_volumes, mask_rsl_file, corr_dir)
    #plot_pairwise_correlation(receptor_volumes, mask_rsl_file, corr_dir)
    exit(0)
    

    # Plot entropy on surface
    project_and_plot_surf(
        [entropy_file], 
        args.wm_surf_filename, 
        args.gm_surf_filename, 
        entropy_dir,
        medial_wall_mask = medial_wall_mask,
        threshold=(0.30, 0.98),
        #cmap='nipy_spectral',
        clobber=True
        )
    exit(0) 
    project_and_plot_surf(
        [std_file], 
        args.wm_surf_filename, 
        args.gm_surf_filename, 
        entropy_dir,
        medial_wall_mask = medial_wall_mask,
        threshold=(0.3,0.7),
        #cmap='nipy_spectral',
        clobber=True
        )
    exit(0)

    #receptor_volumes = [ args.label_file ]
    ### Calculate surface differential gradients
    surface_analysis(receptor_volumes, args.wm_surf_filename, args.gm_surf_filename, profiles_dir, args.output_dir)
                     
    exit(0)
    #comparison_volumes = [gradient_volumes, [ str(i) for i, fn in enumerate(gradient_volumes)] ]
    #comparison_volumes[0] += ratios[0]
    #comparison_volumes[1] += ratios[1]
    #comparison_volumes[0].append( entropy)
    #comparison_volumes[1].append('entropy')
    t1t2_tp = (t1t2_rsl_filename,  't1t2')
    entropy_tp = (entropy, 'entropy')
    comparison_volumes = [[t1t2_tp, entropy_tp],]
    
    [entropy, t1t2_rsl_filename] = project_to_surface(
                                        [entropy, t1t2_rsl_filename], 
                                        args.wm_surf_filename, 
                                        args.gm_surf_filename, 
                                        profiles_dir
                                    )
        
    x = np.load(entropy)[:,2:-2]
    y = np.load(t1t2_rsl_filename)[:,2:-2]
    x = np.mean(x, axis=1)
    y = np.mean(y, axis=1)
    print(pearsonr(x,y) )
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x.reshape(-1,1), y)
    # get model residuals
    residuals = np.abs(y - model.predict(x.reshape(-1,1)))

    plt.scatter(x,y,alpha=0.02)
    plt.savefig(f'{corr_dir}/entropy_vs_t1t2_surf.png')

    #coords_l, faces_l = mesh_utils.load_mesh_ext(args.infl_surf_filename)
    coords_l, faces_l = mesh_utils.load_mesh_ext(args.gm_surf_filename)

    plot_surf( coords_l, 
                faces_l, 
                x, 
                rotate=[90, 270], 
                filename=f'{corr_dir}/entropy_surf.png',
                vmax = np.max(x), #np.percentile(y, 99.5), 
                vmin = np.min(x), # np.percentile(y, 0.5), 
                pvals=np.ones_like(x),
                cmap='nipy_spectral',
                cmap_label='Entropy'
                )

    plot_surf( coords_l, 
                faces_l, 
                y, 
                rotate=[90, 270], 
                filename=f'{corr_dir}/t1t2_surf.png',
                #vmax = np.percentile(x, 98), 
                #vmin = np.percentile(x, 2), 
                pvals=np.ones_like(x),
                cmap='nipy_spectral',
                cmap_label='T1/T2'
                )


    plot_surf( coords_l, 
                faces_l, 
                residuals, 
                rotate=[90, 270], 
                filename=f'{corr_dir}/entropy_vs_t1t2_surf_residuals.png',
                #vmax = np.percentile(residuals, 98), 
                #vmin = np.percentile(residuals, 2), 
                pvals=np.ones_like(residuals),
                cmap='nipy_spectral',
                cmap_label='|Residuals|'
                )


    
