import nibabel as nib
import subprocess
import os
import pandas as pd
import numpy as np
import cca_analysis as cca

import brainbuilder.utils.mesh_utils as mesh_utils

from brainstat.context.meta_analysis import _fetch_precomputed
from brainstat._utils import data_directories
from glob import glob

from utils import get_category
from surf_utils import write_gifti

global default_labels
default_labels = ['affective', 'attention', 'memory', 'inhibition', 'motor', 'visual', 'perception', 'cognitive', 'control', 'social', 'cognition', 
             'reward', 'decision making', 'multisensory', 'visuospatial', 'eye movements', 'action', 
            'auditory', 'pain', 'emotion',  'visual stream', 'association', 'executive'
            ]

def shell(cmd):
    print(cmd)
    subprocess.run([
        cmd
    ], shell=True, executable='/bin/bash' )

from scipy.stats import spearmanr, pearsonr, kendalltau

corr_method_dict={'pearson' : pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}

from matplotlib_surface_plotting import plot_surf
from matplotlib import pyplot as plt


from brainsmash.workbench.geo import cortex
from brainsmash.mapgen.memmap import txt2memmap
from brainsmash.mapgen.stats import pearsonr, pairwise_r
from brainsmash.mapgen.stats import nonparp
from brainsmash.mapgen.base import Base


#surface = "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
#cortex(surface=surface, outfile="./outputs/LeftDenseGeodesicDistmat.txt", euclid=False)
#output_files = txt2memmap(dist_mat_fin, output_dir, maskfile=None, delimiter=' ')

def brain_smash(x:np.array, y:np.array, dist_mat_fin:str, output_dir='outputs/', clobber:bool=False):
    # instantiate class and generate 1000 surrogates
    print('\ttxt2memmap')
    dist_map_npy = f'outputs/distmat.npy'
    if not os.path.exists(dist_map_npy) or clobber :
        dist_map = txt2memmap(dist_mat_fin, output_dir, maskfile=None, delimiter=' ')['distmat']
    else :
        dist_map = np.load(dist_map_npy)

    print('\tCreate generator')
    gen = Base(y, dist_map)  # note: can pass numpy arrays as well as filenames
    print('\tCreate Surrogates')
    surrogate_maps = gen(n=1)

    print('\tCorrelate')
    surrogate_brainmap_corrs = pearsonr(x, surrogate_maps).flatten()

    print('\tPermute and correlate')
    naive_surrogates = np.array([np.random.permutation(y) for _ in range(1000)])
    naive_brainmap_corrs = pearsonr(x, naive_surrogates).flatten()

    test_stat = pearsonr(x, y)[0]

    print('\tCompute p-values')
    naive_p = nonparp(test_stat, naive_brainmap_corrs)
    p = nonparp(test_stat, surrogate_brainmap_corrs)

    print("Spatially naive p-value:", naive_p )
    print("SA-corrected p-value:", p )

    return test_stat, p


def spin_test(coords, x, y, corr, n_rand=1000)->float:
    from brainspace.null_models import SpinPermutations
    
    sp = SpinPermutations(n_rep=n_rand)
    sp.fit(coords)

    x_rotated = np.vstack(sp.randomize(x))
    y_rotated = np.vstack(sp.randomize(y))

    x_idx = np.random.choice(x_rotated.shape[0], x_rotated.shape[0], replace=False)
    y_idx = np.random.choice(y_rotated.shape[0], y_rotated.shape[0], replace=False)

    x_rotated = x_rotated[x_idx]
    y_rotated = y_rotated[y_idx]

    surf_lh = 'data/surfaces/lh.MEBRAINS_0.5mm_1.0.surf.gii'
    surf_lh = 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
    n=5

    coords = nib.load(surf_lh).darrays[0].data[:]
    faces = nib.load(surf_lh).darrays[1].data[:]

    idx = ~np.isnan(x) & ~np.isnan(y) #& (x != 0) & (y != 0)
    r1 = corr(x[idx], y[idx])[0]
    counter=0

    r_nulls = np.zeros(n_rand+1)
    for i, (x_rot, y_rot) in enumerate(zip(x_rotated, y_rotated)):
        idx = ~np.isnan(x_rot) & ~np.isnan(y_rot) #& (x_rot != 0) & (y_rot != 0)
        r0, _ = corr(x_rot[idx], y_rot[idx])

        r_nulls[i] = r0

        if counter > n :
            plt.title(f"Spin {i}\t{np.round(r0,3)}") 
            plot_surf(coords, faces, x_rot, rotate=[90, 270], filename=f'/tmp/rot_{i}_x.png', pvals=np.ones_like(x_rotated[i])) 
            plt.clf(); plt.cla(); plt.close()

            plt.title(f"Spin {i}\t{np.round(r0,3)}") 
            plot_surf(coords, faces, y_rot, rotate=[90, 270], filename=f'/tmp/rot_{i}_y.png', pvals=np.ones_like(x_rotated[i])) 
            plt.clf(); plt.cla(); plt.close()
            counter += 1

    print('R Observed', r1) 
    print('R Nulls', r_nulls)
    p = np.sum(np.abs(r_nulls) > np.abs(r1))  / n_rand 
    print(f"p={p}")
    if r1 > 0.3:
        exit()
    return r1, p 

        
    





def calculate_correlations(receptor_fn:str, feature_surface_files:list, sphere_fn:str, output_dir:str, corr_method='spearman', clobber:bool=False):

    out_df_fn = f'{output_dir}/{os.path.basename(receptor_fn).replace(".func.gii", "")}.csv'

    if not os.path.exists(out_df_fn) or clobber :
        receptor_array = nib.load(receptor_fn).darrays[0].data[:]
        source = os.path.basename(receptor_fn).replace('.func.gii', '')

        try :
            corr = corr_method_dict[corr_method]
        except KeyError:
            raise ValueError(f"Method {corr_method} not implemented. Choose from {list(corr_method_dict.keys())}")

        df_list = []
        for f in feature_surface_files:
            feature_array = nib.load(f).darrays[0].data[:]
            
            #corr = pearsonr 
            #r, p = spin_test(coords, receptor_array, feature_array, corr, n_rand=1000)
            r, p = brain_smash(receptor_array, feature_array, "./outputs/LeftDenseGeodesicDistmat.txt", clobber=False)

            assert not np.isnan(r), f"r is nan for {f}"
        
            category = os.path.basename(f).replace('.func.gii', '')
            print(f"{source}\t{category}\tr2={np.round(r,2)}\tp={np.round(p,3)}")

            df_list.append(pd.DataFrame({'source':source, 'category':[category], 'r':[r], 'p':[p]}))

        df = pd.concat(df_list)
        df.sort_values('r', ascending=False, inplace=True)
        df.to_csv(out_df_fn)
    else :
        df = pd.read_csv(out_df_fn)
    
    return df


def get_surface_features(cortex_surf_fn:str, output_dir:str, labels:list=default_labels, source:str='neurosynth', input_dir:str=None, clobber:bool=False):
    os.makedirs(output_dir, exist_ok=True)

    # load stat map volumes
    if source == 'neurosynth':
        data_dir = data_directories["NEUROSYNTH_DATA_DIR"]
        data_dir.mkdir(exist_ok=True, parents=True)
        feature_volume_files = tuple(_fetch_precomputed(data_dir, database="neurosynth"))
    elif source == 'local':
        assert input_dir is not None, "Local directory not provided"
        feature_volume_files = glob(f'{input_dir}/*nii.gz')
        assert len(feature_volume_files) > 0, f"No feature files found in {input_dir}"

    coords = nib.load(cortex_surf_fn).darrays[0].data[:]

    def vol2surf(x): 
        out_fn = output_dir+os.path.basename(x).replace('.nii.gz', '.func.gii').replace(' ', '_')
        if not os.path.exists(out_fn) or clobber:
            vtr = mesh_utils.volume_filename_to_mesh(coords, x)
            
            assert np.isnan(vtr).any() == False, f"NaN values in {x}"

            write_gifti(vtr, out_fn)
        return out_fn

    def check_feature(f):
        return any([l in f for l in labels])

    feature_tuple = [ (vol2surf( str(f) ), f) for f in feature_volume_files if check_feature(str(f)) ]
    feature_surf_files = [ f for f, _ in feature_tuple]
    feature_volume_files = [ f for _, f in feature_tuple]

    return feature_surf_files, feature_volume_files


metacategories = {
    'emotion': ['emotion', 'affect'],
    'attention': ['attention'],
    'visual': ['visual',  'visuo'],
    'auditory': ['auditory'],
    'memory': ['memory'],
    'sensorimotor': ['motor', 'mov', 'sensory'],
    'social': ['social'],
    'cognition' : ['cognitive', 'cognition', 'executive', 'inhibition' ],
    'reward' : ['reward'],
    'pain' : ['pain'],
}
def categorize(x):
    for k, v in metacategories.items():
        if any([i in x for i in v]):
            return k
    return 'other'


def neurosynth_cca(source_file_list:list, cortex_surf_fn:str, output_dir:str, n_components:int=3, feature_surface_files:list=None, clobber:bool=False):
    """Run CCA analysis on Neurosynth data."""

    if feature_surface_files is None:
        feature_surface_files, _ = get_surface_features(cortex_surf_fn, output_dir+'/neurosynth/', clobber=False) 
    
    categories_list = [ get_category(f) for f in feature_surface_files ]

    metacategories_list = [ categorize(c) for c in categories_list ]

    df = pd.DataFrame({ 'activation_maps':feature_surface_files, 'category':categories_list, 'metacategory':metacategories_list})

    df_list = []
    total_df = pd.DataFrame()
    for metacategory, df_metacategory in df.groupby('metacategory'):
        target_file_list = list(df_metacategory['activation_maps'].values)

        row_fn = f'{output_dir}/{metacategory}.csv'

        n_components = min(len(source_file_list), len(target_file_list))

        if not os.path.exists(row_fn) or clobber or True:
            X_weights = cca.cca_analysis(
                source_file_list, target_file_list, output_dir, zscore_y = False, kind_x = 'receptor', kind_y = 'fmri_activation', label=metacategory, n_components = n_components
                )

            row = pd.DataFrame({'Total CCA Weight': X_weights['Total']});
            #row.columns = X_weights['Receptor'] 
            row['metacategory'] = metacategory
            row['Receptor'] = X_weights['Receptor']
            print(row); 
            row.to_csv(row_fn)
        else :
            row = pd.read_csv(row_fn)      

        df_list.append(row)

    total_df = pd.concat(df_list)
    print(total_df)

    return total_df

def neurosynth_decode(
        source_file_list:list, cortex_surf_fn:str, sphere_fn:str, output_dir:str, corr_method='spearman', labels:list=None, feature_surface_files:list=None, clobber:bool=False
        ):
    os.makedirs(output_dir, exist_ok=True) 
    df_all_csv = f'{output_dir}/all_results.csv'

    if not os.path.exists(df_all_csv) or clobber :
        df_list = []

        if feature_surface_files is None:
            feature_surface_files, _ = get_surface_features(cortex_surf_fn, output_dir+'/neurosynth/', clobber=False) 

        assert len(source_file_list) > 0, "No source files found"
        
        labels = default_labels if labels is None else labels

        for source_feature_fn in source_file_list :
            
            df = calculate_correlations(source_feature_fn, feature_surface_files, sphere_fn, output_dir, corr_method=corr_method, clobber=clobber) 

            df['metacategory'] = df['category'].apply(lambda x : categorize(x) )

            df_list.append(df)

        df_all = pd.concat(df_list)

        df_all.to_csv(df_all_csv)

    else : 
        df_all = pd.read_csv(df_all_csv)

    return df_all

