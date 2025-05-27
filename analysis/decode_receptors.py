import glob
import os
import subprocess
import pandas as pd
import nibabel as nib
import numpy as np

from neurosynth_decode import neurosynth_decode, get_surface_features
from utils import ligand_receptor_dict

mebrains_to_hcp = "/media/windows/projects/fzj_macaque_receptor_atlas/outputs/0.25mm/surfaces/mebrains2human_surf_tfm.surf.gii"

mebrains_to_yerkes = "../neuromaps-nhp-prep/output/aligned/mebrains_to_yerkes19/MacaqueYerkes19.L.mid.32k_fs_LR_sphere.sphere.gii"
mebrains_to_yerkes = "mebrains_to_yerkes19/1/lh.lh.MEBRAINS_0.5mm_0.5_metrics_curv_sulc_warped_sphere.reg.surf.gii"
mebrains_to_yerkes = "../neuromaps-nhp-prep/output/aligned/mebrains_to_yerkes19/1/lh.lh.MEBRAINS.mid_metrics_curv_sulc_warped_sphere.reg.surf.gii"


yerkes_to_hcp = "/home/thomas-funck/projects/neuromaps-nhp-prep/data/macaque-to-human/L.macaque-to-human.sphere.reg.32k_fs_LR.surf.gii"
yerkes_sphere = "../neuromaps-nhp-prep/output/aligned/mebrains_to_yerkes19/MacaqueYerkes19.L.mid.32k_fs_LR_sphere.sphere.gii"
yerkes_sphere = "mebrains_to_yerkes19/spheres/MacaqueYerkes19.L.mid.32k_fs_LR_sphere.sphere.gii"

mebrains_sphere = 'data/surfaces/lh.MEBRAINS_0.5mm_sphere.surf.gii'


hcp_sphere = "S1200.L.sphere.32k_fs_LR.surf.gii"
hcp_cortex = 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'


def transform_labels(labels, output_dir, clobber=False):
    labels_hcp = []
    labels_yerkes = []

    for label in labels:

        outstr = output_dir+'/'+os.path.basename(label).replace('.func.gii', '')
        label_mebrains_to_hcp = f"{outstr}_hcp.func.gii"
        label_mebrains_to_yerkes19 = f"{outstr}_yerkes19.func.gii"

        if not os.path.exists(label_mebrains_to_yerkes19) or clobber:
            print(label_mebrains_to_yerkes19)
            subprocess.run(f"msmresample {mebrains_to_yerkes} {outstr}_yerkes19 -project {yerkes_sphere} -labels {label}", shell=True, executable='/bin/bash') 

        labels_yerkes.append(label_mebrains_to_yerkes19)

        if not os.path.exists(label_mebrains_to_hcp) or clobber:
            print(label_mebrains_to_hcp)
            subprocess.run(f"msmresample {mebrains_to_hcp} {outstr}_hcp -project {hcp_sphere} -labels {label}", shell=True, executable='/bin/bash') 

        assert os.path.exists(label_mebrains_to_hcp), f"File not found: {label_mebrains_to_hcp}"

        labels_hcp.append(label_mebrains_to_hcp)

    return labels_hcp, labels_yerkes

global network_atlas_filename
network_atlas_filename='data/surfaces/L.BezginTo7Networks.32k_fs_LR.label.gii'


def parse_source(x):
    if 'acq-' in x :
        return x.split('_')[0].replace('acq-','')
    elif '_PC' in x :
        return x.split('_')[1]
    else :
        return x

def apply_ligand_receptor(x):
    try :
        return ligand_receptor_dict[x]
    except KeyError:
        return x

def get_network_mean(labels:list, network_atlas_filename:str, output_dir:str, clobber:bool=False):

    atlas_coding = { 1:'DMN', 2:'Somatomotor', 3:'Auditory', 4:'Limbic', 5:'DoralAtt', 6:'Visual', 7:'Insular-opercular'}
    
    x_coding = { 6:1, 3:2, 2:3,  4:4, 7:5, 5:6, 1:7 }
    
    network_atlas = nib.load(network_atlas_filename).darrays[0].data[:]

    df_csv = f'{output_dir}/networks.csv'

    if not os.path.exists(df_csv) or clobber : 

        df_list = []

        for label_filename in labels :
            label = nib.load(label_filename).darrays[0].data[:]

            idx = ~np.isnan(label) 
            label = (label - np.mean(label[idx])) / np.std(label[idx])

            source = apply_ligand_receptor( parse_source( os.path.basename(label_filename).replace('.func.gii','') ) )

            for i in np.unique(network_atlas)[1:] :
                
                mu = np.mean( label[ (network_atlas==i) & ~np.isnan(label) ])
                
                if np.isnan(mu) : 
                    print(f"mu is nan for {source} {i}")
                    continue

                x = x_coding[i]

                network = atlas_coding[i]
                
                row = pd.DataFrame({'source':[source], 'network':[network], 'x':[x], 'id':[i], 'mean': [mu] })        
                
                print(row)

                df_list.append(row)

        df = pd.concat(df_list)

        df.to_csv(df_csv)
    else :
        df = pd.read_csv(df_csv)

    return df

def decode_across_depths(sphere_fn:str, depths:list, depth_labels:list, output_dir:str, clobber:bool=True): 

    corr_list = []
    network_list = []

    depths = [ '0.06-0.94', '0.06-0.33', '0.33-0.67', '0.67-0.94']

    for i, (depth, labels) in enumerate(zip(depths,depth_labels)) :

        output_depth_dir = f"{output_dir}/{depth}/"

        os.makedirs(output_depth_dir, exist_ok=True)

        labels_hcp, labels_yerkes = transform_labels(labels, output_depth_dir, clobber=clobber)

        clobber = True
        corr_depth = neurosynth_decode(labels_hcp, hcp_cortex, sphere_fn, f'{output_dir}/{depth}', feature_surface_files=feature_surface_files, clobber=clobber)
        corr_depth['depth'] = depth

        network_depth_df = get_network_mean(labels_yerkes, network_atlas_filename, f'{output_dir}/{depth}/', clobber=clobber)
        network_depth_df['depth'] = depth

        corr_list.append(corr_depth)

        network_list.append(network_depth_df)

    corr_df = pd.concat(corr_list)

    network_df = pd.concat(network_list)

    return corr_df, network_df


if __name__ == "__main__":
    
    output_dir = "/home/thomas-funck/projects/fzj_macaque_receptor_atlas/outputs/receptor_decoding/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(mebrains_to_hcp) or True :
        cmd = f'wb_command -surface-sphere-project-unproject {mebrains_to_yerkes} {yerkes_sphere} {yerkes_to_hcp} {mebrains_to_hcp}'
        subprocess.run(cmd, shell=True, executable='/bin/bash') 

    clobber = False
    corr_list = []
    cca_list = []

    feature_surface_files, _ = get_surface_features(hcp_cortex, output_dir+'/neurosynth/', source='local', input_dir='neuroquery/', clobber=clobber) 

    depths = [ '0.06-0.94', '0.06-0.33', '0.33-0.67', '0.67-0.94']
    
    receptor_labels=[]
    entropy_labels=[]
    ei_labels=[]
    pca_labels=[]
    for depth in depths :
        #receptor_labels.append( [ f for f in glob.glob(f"outputs/0.25mm/{depth}/*smoothed.func.gii") if not 'cellbody' in f and 'myelin' not in f] )
        receptor_labels.append( [ f for f in glob.glob(f"outputs/0.25mm/averaged_surfaces/acq-*{depth}*.func.gii") if not 'cellbody' in f and 'myelin' not in f] )
        entropy_labels.append( [ f for f in glob.glob(f"outputs/0.25mm/{depth}/entropy/entropy.func.gii") ] ) 
        ei_labels.append([ f for f in glob.glob(f"outputs/0.25mm/{depth}/ratios/macaque_ratio_ex_inh.func.gii") ] )
        pca_labels.append([ f for f in glob.glob(f"outputs/0.25mm/{depth}/pca/*surf_PC*.surf.gii") ] )

    receptor_df, receptor_network_df = decode_across_depths(hcp_sphere, depths, receptor_labels, output_dir+'/receptors/', clobber=clobber)

    entropy_df, entropy_network_df = decode_across_depths(mebrains_sphere, depths, entropy_labels, output_dir+'/entropy/', clobber=clobber)

    ei_df, ei_network_df = decode_across_depths(mebrains_sphere, depths, ei_labels, output_dir+'/ei/', clobber=clobber)

    pca_df, pca_network_df = decode_across_depths(mebrains_sphere, depths, pca_labels, output_dir+'/pca/', clobber=clobber)
    #cca_df = pd.concat(cca_list)

    receptor_df['source'] = receptor_df['source'].apply(lambda x: x.split('_')[0].replace('acq-', ''))
    receptor_df['Receptor'] = receptor_df['source'].apply(lambda x: ligand_receptor_dict[x])

    receptor_df.to_csv(f"{output_dir}/neurosynth_decoding_receptors.csv", index=False)
    entropy_df.to_csv(f"{output_dir}/neurosynth_decoding_entropy.csv", index=False)
    ei_df.to_csv(f"{output_dir}/neurosynth_decoding_ei.csv", index=False)
    pca_df.to_csv(f"{output_dir}/neurosynth_decoding_pca.csv", index=False)

    receptor_network_df.to_csv(f"{output_dir}/network_decoding_receptors.csv", index=False)
    entropy_network_df.to_csv(f"{output_dir}/network_decoding_entropy.csv", index=False)
    ei_network_df.to_csv(f"{output_dir}/network_decoding_ei.csv", index=False)
    pca_network_df.to_csv(f"{output_dir}/network_decoding_pca.csv", index=False)

 

    
             