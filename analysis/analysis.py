
import argparse
import nibabel as nib
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess

from surf_utils import write_gifti

from surf_utils import project_to_surface, load_gifti, plot_receptor_surf

from skimage.transform import resize

import entropy as entropy

from subject_variability import create_averaged_volumes
from surf_utils import project_to_surface

from receptorSurfaces import ReceptorSurfaces

global ligand_receptor_dict
ligand_receptor_dict={'ampa':'AMPA', 'kain':'Kainate', 'mk80':'NMDA', 'ly34':'mGluR2/3', 'flum':'GABA$_A$ Benz.', 'cgp5':'GABA$_B$', 'musc':'GABA$_A$ Agonist', 'sr95':'GABA$_A$ Antagonist', 'pire':r'Muscarinic M$_1$', 'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_3$','epib':r'Nicotinic $\alpha_4\beta_2$','oxot':r'Muscarinic M$_2$ (oxot)', 'praz':r'$\alpha_1$','uk14':r'$\alpha_2$ (agonist)','rx82':r'$\alpha_2$ (antagonist)', 'dpat':r'5-HT$_{1A}$','keta':r'5HT$_2$', 'sch2':r"D$_1$", 'dpmg':'Adenosine 1', 'cellbody':'Cell Body', 'myelin':'Myelin'}

def imshow_volumes(volumes, mask_filename, template_filename, output_dir):
    
    m0=3
    m1=5

    mask = nib.load(mask_filename).get_fdata()
    template = nib.load(template_filename).get_fdata()
    from scipy.ndimage import center_of_mass
    fig = plt.figure(figsize=(m1*10,m0*10))
    fig.patch.set_facecolor('black')
    #set background to black
    fig.patch.set_facecolor('black')
    for i, fn in enumerate(volumes):
        vol = nib.load(fn).get_fdata()
        vol[mask<1]=0
        x,_,_ = center_of_mass(vol)
        x = int(x)
        
        t1_section=template[int(x),:,:]
        section=vol[int(x),:,:]
        section = np.rot90(section, 1)
        t1_section = np.rot90(t1_section,1)

        ligand = os.path.basename(fn).replace('macaque_','').replace('.nii.gz','')

        ax=plt.subplot(m0,m1,i+1)
        receptor=ligand_receptor_dict[ligand]
        plt.title(receptor, {'color':'white','fontsize':50})
        v0,v1=np.percentile(section[section>0], [2,100])
        
        plt.imshow(t1_section,cmap='gray')
        plt.imshow(section, vmin=v0,vmax=v1, cmap='nipy_spectral',alpha=0.65)

        ax.spines[['right', 'top']].set_visible(False)
        plt.axis('off')
        ax.set_facecolor('black')
        ax.set_aspect('equal')

    fig.tight_layout() 
    
    plt.savefig(f'{output_dir}/volumes.png', dpi=400)

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







def transform_surface(
        sphere_in:str, 
        sphere_project_to:str, 
        sphere_unproject_from:str,
        output_sphere:str,
        clobber:bool=False):
    """Resample input surface to target surface using wb_command -surface-sphere-project-unproject

    Example 1: You have a Human to Chimpanzee registration, and a Chimpanzee
      to Macaque registration, and want to combine them.  If you use the Human
      sphere registered to Chimpanzee as sphere-in, the Chimpanzee standard
      sphere as project-to, and the Chimpanzee sphere registered to Macaque as
      unproject-from, the output will be the Human sphere in register with the
      Macaque.
      
      human->chimp = sphere-in
      chimp = sphere-to
      chimp->macaq = unproject-from
      human->chimp->macaque = output
      
      ------
    Example 2: MEBRAINS to Yerkes to hcp
      mebrains->yerkes = sphere-in
      yerkes = sphere-to
      yerkes->hcp = output
    """

    #cmd = f"wb_command -surface-sphere-project-unproject {sphere_in} {sphere_project_to} {sphere_unproject_from} {output_sphere}"
    #print(cmd); 
    #if not os.path.exists(output_sphere) or clobber:
    #    subprocess.run(cmd, shell=True,executable='/bin/bash')

    return output_sphere

def prepare_surfaces(mebrains_to_yerkes:str, yerkes_sphere:str, yerkes_to_hcp:str, output_dir:str, clobber:bool=True):

    mebrains_to_hcp_surf_tfm = f'{output_dir}/mebrains2human_surf_tfm.surf.gii'

    transform_surface(mebrains_to_yerkes, yerkes_sphere, yerkes_to_hcp, mebrains_to_hcp_surf_tfm, clobber=clobber)

    return  mebrains_to_hcp_surf_tfm


# get repository root dir
wrk_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess(
        output_dir:str,
        receptor_volumes:list,
        mask_file:str,
        wm_surf_filename:str,
        gm_surf_filename:str,
        mebrains_filename:str,
        mebrains_medial_wall_mask_filename:str, 
        bound0:int=0,
        bound1:int=18,
        nlayers:int=18,
        zscore:bool=False,
        label:str='',
        clobber:bool=False
):

    medial_wall_mask = load_gifti(mebrains_medial_wall_mask_filename).astype(bool).reshape(-1,)

    ### Resize mask to receptor volume
    mask_rsl_file = resize_mask_to_receptor_volume(mask_file, receptor_volumes[0], output_dir)

    mebrains_rsl = resize_mask_to_receptor_volume( mebrains_filename, receptor_volumes[0], output_dir, order=3)

    receptor_surfaces = project_to_surface( 
        receptor_volumes, 
        wm_surf_filename, 
        gm_surf_filename, 
        output_dir, 
        mask=medial_wall_mask, 
        n=nlayers, 
        agg_func=np.median, #np.mean, 
        surf_smooth=0,
        zscore=zscore,
        bound0=bound0, 
        bound1=bound1, 
        clobber=clobber
    )

    #TODO add medial wall mask to receptor surfaces
    #TODO create subsets of receptor surfaces  

    #if viz_receptors :
    #     imshow_volumes(receptor_volumes, mask_rsl_file, mebrains_rsl, args.output_dir )
    return receptor_surfaces, mebrains_rsl, mask_rsl_file, medial_wall_mask

def averaged_surfaces(receptor_volumes, subsets, n_layers, medial_wall_mask, output_dir, clobber:bool=False):

    df_list = []

    for sub_id, subject_receptor_volumes in enumerate(receptor_volumes):
        
        for bound0, bound1 in subsets:
            p0 = np.round(bound0/n_layers,2)
            p1 = np.round(bound1/n_layers,2)
            label = f'{p0:.2f}-{p1:.2f}'

            layer_output_dir = f'{args.output_dir}/{label}/'
            os.makedirs(layer_output_dir, exist_ok=True)

            receptor_surfaces = project_to_surface( 
                subject_receptor_volumes, 
                args.wm_surf_filename, 
                args.gm_surf_filename, 
                layer_output_dir, 
                mask = nib.load(medial_wall_mask).darrays[0].data[:].astype(bool).reshape(-1,), 
                n = n_layers, 
                agg_func = np.mean, 
                zscore = False,
                surf_smooth = 1,
                bound0=bound0, 
                bound1=bound1, 
                clobber=clobber
            )

            receptor_labels = [ l for f in subject_receptor_volumes for l in f.split('_') if 'acq-' in l]

            n = len(receptor_surfaces)

            bound_label = f'{p0:.2f}-{p1:.2f}'
            
            tdf = pd.DataFrame([[sub_id]*n, [bound_label]*n, receptor_labels, receptor_surfaces]).T 
            tdf.columns=['subject', 'depth', 'receptor', 'surface']
            df_list.append( tdf )

    df = pd.concat(df_list)
    df = df.loc[ (df['receptor'] != 'cellbody') & (df['receptor'] != 'myelin')]

    average_surface_dir = f'{output_dir}/averaged_surfaces/'

    os.makedirs(average_surface_dir, exist_ok=True)

    for (depth, receptor), group in df.groupby(['depth', 'receptor']):
        n = 0

        print('Depth:', depth, 'Receptor:', receptor) 

        output_surface_filename = f'{average_surface_dir}/{receptor}_{depth}.func.gii'

        if os.path.exists(output_surface_filename) and not clobber or True:
            mean_list=[]
            std_list=[]
            ar = None
            for fn in group['surface']:
                print(fn)
                ar0 = nib.load(fn).darrays[0].data[:]
                
                idx = ~np.isnan(ar0)
                mu = np.mean(ar0[idx])
                sig = np.std(ar0[idx])

                mean_list.append(mu)
                
                std_list.append(sig)

                ar0_norm = ar0 #(ar0 - mu)/sig
                if ar is None:
                    ar = ar0_norm
                else:
                    ar += ar0_norm

                n += 1 

            global_mean = np.mean(mean_list)
            global_std = np.mean(std_list)
            print(global_mean, global_std)
            ar = (ar/n) #*global_std + global_mean
            write_gifti(ar, output_surface_filename)

            plot_receptor_surf(
                [output_surface_filename], args.wm_surf_filename, average_surface_dir, label=f'{receptor}_{depth}_avg', cmap='RdBu_r', threshold=[0,100]
            )


        




if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Volumetric Gradient Analysis')
    parser.add_argument('-m', dest='mask_file', default='data/volumes/MEBRAINS_segmentation_NEW_gm_left.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-l', dest='label_file', default='data/volumes/MEBRAINS_pseudo-segmentation-0_gm_left.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-e' , dest='mebrains_filename', default='data/volumes/MEBRAINS_T1_masked.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-y' , dest='yerkes_template_filename', default='data/volumes/MacaqueYerkes19_v1.2_AverageT1w_restore_masked.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-a' , dest='yerkes_atlas_filename', default='data/surfaces/L.BezginTo7Networks.32k_fs_LR.label.gii', type=str, help='Path to mask file')
    #parser.add_argument('-a' , dest='yerkes_atlas_filename', default='data/surfaces/L.Yeo2011_7Networks_N1000.human-to-monkey.10k_fs_LR.label.gii', type=str, help='Path to mask file')
    parser.add_argument('-n', dest='n', default=10000, type=int, help='Number of random voxels to sample')
    parser.add_argument('-w', dest='medial_wall_volume_filename', default=f'{wrk_dir}/data/volumes/medial_wall_mask_morph_0.4mm.nii.gz', type=str, help='Path to medial wall mask')
    # MEBRAINs surfaces
    parser.add_argument('--wm-surf', dest='wm_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/lh.MEBRAINS_0.5mm_1.0.surf.gii', help='Path to MEBRAINS white matter surface')
    parser.add_argument('--mid-surf', dest='mid_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/lh.MEBRAINS_0.5mm_0.5.surf.gii', help='Path to MEBRAINS white matter surface')
    parser.add_argument('--gm-surf', dest='gm_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/lh.MEBRAINS_0.5mm_0.0.surf.gii', help='Path to MEBRAINS pial matter surface')
    parser.add_argument('--sphere-surf', dest='sphere_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/lh.MEBRAINS_0.5mm_sphere.surf.gii', help='Path to Yerkes pial matter surface')
    # Yerkes surfaces
    parser.add_argument('--y-gm-surf', dest='yerkes_gm_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19.L.pial.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-mid-surf', dest='yerkes_mid_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19_v1.2.L.midthickness.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-wm-surf', dest='yerkes_wm_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19.L.white.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-sphere-surf', dest='yerkes_sphere_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19_v1.2.L.sphere.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-sphere-10k-surf', dest='yerkes_10k_sphere_surf_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19.L.sphere.10k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    parser.add_argument('--y-sulc', dest='yerkes_sulc_filename', type=str, default=f'{wrk_dir}/data/surfaces/MacaqueYerkes19_v1.2.sulc.32k_fs_LR.dscalar.nii', help='Path to Yerkes pial matter surface')
    #parser.add_argument('--gm-infl', dest='infl_surf_filename', type=str, default='data/surfaces/MacaqueYerkes19.L.inflated.32k_fs_LR.surf.gii', help='Path to Yerkes pial matter surface')
    clobber=False

    args = parser.parse_args()
    t1t2_filename = 'data/volumes/MEBRAINS_T1T2_masked.nii.gz'
    mebrains_medial_wall_mask_filename = 'data/surfaces/lh.MEBRAINS_medial_wall_mask.func.gii' 
    yerkes_medial_wall_mask_filename = 'data/surfaces/Yerkes19.L_medial_wall_mask.func.gii'

    # 
    resolution = 0.25
    args.output_dir = f'outputs/{resolution}mm/'
    args.input_dir = 'data/reconstruction/version_6/macaque/volumes/raw/'
    parser.add_argument('-i', dest='input_dir', type=str, default='data/reconstruction/', help='Path to receptor volumes')

    # Human HCP surface
    human_sphere='data/surfaces/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'

    # D99 surfaces
    d99_wm_surf_filename='data/surfaces/D99_L_AVG_T1_v2.L.WM.167625.surf.wm'
    d99_mid_surf_filename='data/surfaces/D99_L_AVG_T1_v2.L.MID.167625.surf.gii'
    d99_gm_surf_filename='data/surfaces/D99_L_AVG_T1_v2.L.PIAL.167625.surf.gii'
    d99_sphere_surf_filename='data/surfaces/D99_L_AVG_T1_v2.L.MID.SPHERE.SIX.167625.surf.gii'
    d99_medial_wall_mask_filename='data/surfaces/D99_L_AVG_T1_v2.L.cortex.func.gii'

    # tfm spheres
    mebrains_to_yerkes='~/projects/neuromaps-nhp-prep/output/aligned/mebrains_to_yerkes19/1/lh.MacaqueYerkes19.L.mid.32k_fs_LR_metrics_curv_sulc_warped_sphere.reg.surf.gii'
    yerkes_to_hcp='~/projects/neuromaps-nhp-prep/data/macaque-to-human/L.macaque-to-human.sphere.reg.32k_fs_LR.surf.gii'

    os.makedirs(args.output_dir, exist_ok=True)
    receptor_dir = f"{args.input_dir}/"
    hist_dir = f"{args.input_dir}/hist/"
    
    profiles_dir = f'{args.output_dir}/profiles/'
    yerkes_profiles_dir = f'{args.output_dir}/profiles_yerkes/'
    align_dir = f"{args.output_dir}/align/"
    diff_gradient_dir = f"{args.output_dir}/diff_gradients/"
    grad_dir = f"{args.output_dir}/gradients/"    
    segment_dir = f"{args.output_dir}/segment/"    

    n_layers = 18
    b0 = n_layers*1/3
    b1 = n_layers*2/3
    subsets = ((1,b0), (b0,b1), (b1,n_layers-1), (1,n_layers-1))



    mebrains_to_hcp = prepare_surfaces(
        mebrains_to_yerkes, args.yerkes_sphere_surf_filename, yerkes_to_hcp, args.output_dir+'/surfaces/', clobber=True
        )
     
    #receptor_volumes = create_averaged_volumes(args.input_dir, args.output_dir+'/volume_averages/', clobber=clobber)

    # exclude cellbody and myelin
    #receptor_volumes = [ f for f in receptor_volumes if not 'cellbody' in f and 'myelin' not in f]

    receptor_volumes_list = []
    for subj in ['11530', '11539', '11543'] :
        receptor_volumes = glob.glob(f'{receptor_dir}/sub-{subj}_hemi-L_acq-*_0.25mm_l15_cortex_unfilled.nii.gz')
        receptor_volumes_list.append(receptor_volumes)

    averaged_surfaces(receptor_volumes_list, subsets, n_layers, mebrains_medial_wall_mask_filename, args.output_dir, clobber=clobber)

    for bound0, bound1 in subsets:
        p0 = np.round(bound0/n_layers,2)
        p1 = np.round(bound1/n_layers,2)
        label = f'{p0:.2f}-{p1:.2f}'

        layer_output_dir = f'{args.output_dir}/{label}/'
        os.makedirs(layer_output_dir, exist_ok=True)

        receptor_surfaces = glob.glob(f'{args.output_dir}/averaged_surfaces/*_{label}.func.gii')

        #receptor_volumes, receptor_surfaces, mebrains_rsl, mask_rsl_file, medial_wall_mask = preprocess(
        #    layer_output_dir,
        #    receptor_volumes,
        #    args.mask_file,
        #    args.wm_surf_filename,
        #    args.gm_surf_filename,
        #    args.mebrains_filename,
        #    mebrains_medial_wall_mask_filename,
        #    bound0 = p0,
        #    bound1 = p1,
        #    nlayers = n_layers,
        #    label = label,
        #    clobber = clobber
        #)

        ReceptorSurfaces(
            receptor_surfaces, 
            args.wm_surf_filename, 
            args.sphere_surf_filename, 
            layer_output_dir,
            human_sphere = human_sphere,
            #macaque_to_human_tfm_sphere = mebrains_to_hcp,
            label=label, 
            clobber=True
            ).run()





