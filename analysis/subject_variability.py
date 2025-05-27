
from glob import glob
import numpy as np
import nibabel as nib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import statsmodels.api as sm
import seaborn as sns


# 'sch2':r"D$_1$", 
ligand_receptor_dict = {'ampa':'AMPA', 'kain':'Kainate', 'dpmg':'Adenosine 1', 'mk80':'NMDA', 'ly34':'mGluR2/3', 'flum':'GABA$_A$ Benz.', 'cgp5':'GABA$_B$', 'musc':'GABA$_A$ Agonist', 'sr95':'GABA$_A$ Antagonist', 'pire':r'Muscarinic M$_1$', 'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_3$','epib':r'Nicotinic $\alpha_4\beta_2$','oxot':r'Muscarinic M$_2$ (oxot)', 'praz':r'$\alpha_1$','uk14':r'$\alpha_2$ (agonist)','rx82':r'$\alpha_2$ (antagonist)', 'dpat':r'5-HT$_{1A}$','keta':r'5HT$_2$',  'cellbody':'Cell Body', 'myelin':'Myelin'}

def get_subject_data(subjects, hemispheres, data_dir):
    data_dict = {}
    ligands = []
    for subject in subjects:
        data_dict[subject] = {}
        for hemisphere in hemispheres:
            data_dict[subject][hemisphere] = {}
            for ligand in ligand_receptor_dict.keys():
                
                filename = glob(f'{data_dir}/*{subject}*hemi-{hemisphere}*{ligand}*nii.gz')
                if len(filename) == 1:
                
                    data_dict[subject][hemisphere][ligand] = filename[0]
                    ligands.append(ligand)
    ligands = np.unique(ligands)

    return data_dict, ligands

def save_residuals(data, mod, output_dir, clobber:bool=False):

    output_filename = f'{output_dir}/11539_residuals_{ligand}.nii.gz'

    if os.path.exists(output_filename) and not clobber:
    
        residuals = np.abs(data[:,1] - mod.predict(data[:,0].reshape(-1,1)).flatten())

        residuals_volume = np.zeros(img.shape).flatten()
        residuals_volume[idx] = residuals
        residuals_volume = residuals_volume.reshape(img.shape)

        nib.Nifti1Image(residuals_volume, img.affine).to_filename(output_filename)


def hemispheric_symmetry(data_dict , output_dir):
    dict_11539_r = data_dict[11539]["R"]
    dict_11539_l = data_dict[11539]["L"]

    df = pd.DataFrame(columns=['ligand', 'slope', 'intercept', 'r2'])
    plt.figure(figsize=(20,20))
    c=1
    for i, ligand in enumerate(ligand_receptor_dict.keys()):
        if ligand in dict_11539_r.keys() and ligand in dict_11539_l.keys():
            img = nib.load(dict_11539_r[ligand])
            data_r = img.get_fdata()
            data_l = nib.load(dict_11539_l[ligand]).get_fdata()

            # flip data_l along y axis
            data_l = np.flip(data_l, axis=0)
            nib.Nifti1Image(data_l, img.affine).to_filename(f'outputs/11539_L_{ligand}.nii.gz')

            # Calculate regression of right and left hemisphere
            data_r = data_r.flatten()
            data_l = data_l.flatten()

            idx = (~np.isnan(data_r)) & (data_r > 0) & (~np.isnan(data_l)) & (data_l > 0)
            data_r = data_r[idx]
            data_l = data_l[idx]

            data = np.concatenate( [ data_r.reshape(-1,1), data_l.reshape(-1,1) ] , axis=1)
            
            receptor = ligand_receptor_dict[ligand]
            # Linear Regression
            #mod = sm.OLS(data[:,0], data[:,1])
            #res = mod.fit()
            mod = LinearRegression().fit(data[:,0].reshape(-1,1), data[:,1].reshape(-1,1))

            r2 = np.round( mod.score(data[:,0].reshape(-1,1), data[:,1].reshape(-1,1)), 2)
            coef = round(mod.coef_[0][0],2)
            intercept = round(mod.intercept_[0],2)

            save_residuals(data, mod, output_dir) 

            print(f'{i}\t{ligand}: slope={coef} inter={intercept} r2={r2}')
            plt.subplot(3,5,c)
            sns.regplot(x=data[:,0], y=data[:,1], scatter_kws = {'alpha':0.001})
            plt.title(f'{ligand_receptor_dict[ligand]}')
            plt.xlabel('Right Hemisphere')
            plt.ylabel('Left Hemisphere')
            plt.text(0.1,0.9,f'slope={coef}\nintercept={intercept}\n$R^2$={r2}', transform=plt.gca().transAxes)
            c += 1 

            df = pd.concat([df, pd.DataFrame({'receptor':[receptor], 'ligand':[ligand], 'slope':[coef], 'intercept':[intercept], 'r2':[r2]}) ])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/subject_variability.png')

    df.sort_values('slope', ascending=False, inplace=True)
    df.to_csv(f'{output_dir}/subject_variability.csv', index=False)

    sns.barplot(x='receptor', y='slope', data=df)
    plt.xticks(rotation=90)
    plt.savefig(f'{output_dir}/subject_variability_barplot.png')


def save_volume(output_filename, base_dim, affine, valid_idx, array):
    output_volume = np.zeros(base_dim).flatten()
    output_volume[valid_idx] = array
    output_volume = output_volume.reshape(base_dim)
    nib.Nifti1Image(output_volume, affine).to_filename(output_filename)
    return output_volume

def average_volumes(intersubject, voxel_var, valid_idx, base_dim, affine, output_filename, global_mean=0, global_std=1, r=3):
    """average the columns of intersubect only if they fall with r < 3 x voxel_var"""
    n = np.ones( intersubject.shape )

    voxel_var = np.repeat(voxel_var[:,np.newaxis], intersubject.shape[1], axis=1)
    valid_var_idx = np.abs(intersubject) < r * voxel_var

    #x = np.mean(intersubject, axis=1, where=valid_var_idx)
    x = np.sum(n, axis=1, where=valid_var_idx)

    #x = x * global_std + global_mean

    save_volume(output_filename, base_dim, affine, valid_idx, x)
    

def intersubject_variability(data_dict, ligands, output_dir): 

    img = nib.load(data_dict[11539]['L']['ampa'])
    base_dim = img.shape
    flat_dim = np.product(base_dim)

    valid_idx = np.zeros( flat_dim ).flatten()

    subjects = list(data_dict.keys())

    for sub in data_dict.keys():
        for ligand in data_dict[sub]['L'].keys():
            filename = data_dict[sub]['L'][ligand]
            valid_idx += nib.load(filename).get_fdata().flatten() > 0
    valid_idx = valid_idx > 0

    df = pd.DataFrame()
    df_diff = pd.DataFrame()
    for ligand in ligands :
        intersubject = None 
        n=0
        global_mean_list = []
        global_std_list = []
        hemi_list = []
        for hemi in hemispheres:
            for sub in data_dict.keys():
                if ligand not in data_dict[sub][hemi].keys():
                    continue
                n += 1
                img = nib.load(data_dict[sub][hemi][ligand])
                data = img.get_fdata()
                if hemi == 'R' :
                    data = np.flip(data, axis=0)
                
                data= data.flatten()
            
                data = data[valid_idx]

                global_mean_list.append(np.mean(data))
                global_std_list.append(np.std(data))

                #data = (data - np.mean(data))/np.std(data)

                data = data.reshape(-1,1)

                if intersubject is None:
                    intersubject = data
                else :
                    intersubject = np.concatenate([intersubject, data], axis=1)

                hemi_list += [ (sub, hemi) ]

        for i, (sub0, hemi0) in enumerate(hemi_list):
            for j, (sub1, hemi1) in enumerate(hemi_list):
                 if i >= j:
                     continue

                 r = pearsonr(intersubject[:,i], intersubject[:,j])[0]
                 mse = np.mean((intersubject[:,i] - intersubject[:,j])**2)
                 row0 = pd.DataFrame({'ligand':[ligand],'sub0':[sub0], 'hemi0':[hemi0], 'sub2':[sub1], 'hemi1':[hemi1], 'r2':[r], 'mse':[mse]}) 
                 #row1 = pd.DataFrame({'ligand':[ligand],'sub1':subhemi1, 'sub2':subhemi0, 'r2':[r], 'mse':[mse]})
                 df_diff = pd.concat([df_diff, row0])

        std_filename = f'{output_dir}/intersubject_{ligand}_n-{n}_std_.nii.gz'
        mad_filename = f'{output_dir}/intersubject_{ligand}_n-{n}_mad.nii.gz'
        ligand_filename = f'{output_dir}/intersubject_{ligand}_n-{n}.nii.gz'

        voxel_std = np.std(intersubject, axis=1)
        # calculate the median absolute deviation over axis 1 of intersubject
        voxel_mad = np.median(np.abs(intersubject - np.median(intersubject, axis=1).reshape(-1,1)), axis=1)
    
        std_vol = save_volume(std_filename, base_dim, img.affine, valid_idx, voxel_std)
        mad_vol = save_volume(mad_filename, base_dim, img.affine, valid_idx, voxel_mad)
        global_mean = np.mean(global_mean_list)
        global_std = np.mean(global_std_list)
        average_volumes(intersubject, voxel_mad, valid_idx, base_dim, img.affine, ligand_filename, global_mean=global_mean, global_std=global_std )

        row = pd.DataFrame({'ligand':[ligand],'std':[np.mean(voxel_std)], 'mad':[np.mean(voxel_mad)], 'n':[n]})
        df = pd.concat([df, row])

    df_l = df_diff.loc[ (df_diff['hemi0']=='L') & (df_diff['hemi1']=='L') ]
    df_r = df_diff.loc[ (df_diff['hemi0']=='R') | (df_diff['hemi1']=='R') ]

    print(df.shape, df_l.shape, df_r.shape)
    #plot histograms for df_l and df_r
    plt.figure(figsize=(10,10))
    sns.histplot(df_l['r2'], bins=8, alpha=0.25, label='Left Hemisphere', element="step")
    sns.histplot(df_r['r2'], bins=8,  alpha=0.25, label='Right Hemisphere', element="step")
    plt.legend()
    plt.savefig(f'{output_dir}/intersubject_variability_histogram.png')

    #  Kullback–Leibler (KL) divergence between df_l and df_r
    # bin the r2 values and calculate the KL divergence
    bin_max = df_diff['r2'].max()
    bin_min = df_diff['r2'].min()
    bins = np.linspace(bin_min,bin_max,6)
    p_l, _ = np.histogram(df_l['r2'], bins=bins)
    p_r, _ = np.histogram(df_r['r2'], bins=bins)
    p_l = p_l/p_l.sum()
    p_r = p_r/p_r.sum()
    print(p_l, p_r)
    kl = np.sum(p_l * np.log(p_l/p_r))
    print(f'KL divergence between Left and Right Hemisphere: {kl}')


def create_averaged_volumes(data_dir, output_dir, subjects = [11530, 11539, 11543 ], hemispheres = ["L", "R"], clobber:bool=False):
    
    os.makedirs(output_dir, exist_ok=True)

    data_dict, ligand_list = get_subject_data(subjects, hemispheres, data_dir)

    assert len(ligand_list) > 0, 'No ligands found'

    subjects = [11530, 11539, 11543 ]
    hemispheres = ["L", "R"]
    output_list = []

    for ligand in ligand_list :
        
        output_fn = f'{output_dir}/macaque_{ligand}.nii.gz'
        print('Average Volume:', output_fn)
        output_list.append(output_fn)

        if not os.path.exists(output_fn) or clobber :

            img = nib.load(data_dict[11539]['L']['ampa'])

            avg_vol = np.zeros( img.shape )
            n = np.zeros_like(avg_vol)

            for subject in subjects:
                for hemisphere in hemispheres:
                    try :
                        filename = data_dict[subject][hemisphere][ligand]
                    except KeyError :
                        print('Skipping:', subject, hemisphere, ligand)
                        continue

                    img = nib.load(filename)
                    data = img.get_fdata()

                    avg_vol += data
                    n[data>0] += 1

            idx = n > 0 
            avg_vol[idx] = avg_vol[idx] / n[idx]

            nib.Nifti1Image(avg_vol, img.affine).to_filename(output_fn)
    
    return output_list


if __name__ == '__main__':



    data_dir = 'data/reconstruction/subjects/smoothed/'
    output_dir = 'outputs/subject_level/'

    os.makedirs(output_dir, exist_ok=True)


    #hemispheric_symmetry(data_dict , output_dir)

    receptor_volumes, data_dict = create_averaged_volumes( ligands, output_dir+'/averages/')

    intersubject_variability(data_dict, ligands, output_dir)

    






