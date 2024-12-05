
import glob
import os
from surf_utils import align_surface, msm_resample_list, surface_modify_sphere
import nibabel as nib
import numpy as np
from pca import surf_pca
from surf_utils import plot_receptor_surf
import cca_analysis as cca 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas as pd
from surf_utils import write_gifti

from utils import ligand_receptor_dict


def file_lists_to_array(file_list):
    print(file_list[0])
    ar = np.array([nib.load(f).darrays[0].data for f in file_list])

    if len(ar.shape) == 3:
        ar = np.mean(ar, axis=3)
    print(ar.shape)
    exit(0)


import statsmodels.api as sm
def receptor_cell_regr(X, Y, output_dir:str, surface_filename:str, mask, avg_labels, labels, clobber:bool=False):
    # Iterate over Y columns
    df = pd.DataFrame()
    for receptor_name in Y.columns: 
        y = Y[receptor_name].values
        print(receptor_name)
        reg = sm.OLS(y,X)
        fit = reg.fit()
        print(fit.summary())

        # print r2
        #r2 = reg.score(X.values, receptor)
        # adjusted r2
        r2 = fit.rsquared_adj
        residuals = fit.resid 
        residuals_full = np.zeros(mask.shape)

        # plot heteroscedasticity and qq plot
        plt.clf(); plt.cla()
        sm.graphics.plot_regress_exog(fit, 0)
        plt.savefig(f'{output_dir}/regr_{receptor_name}.png')



        if avg_labels is None:
            residuals_full[mask] = residuals
        else:
            residuals_partial = np.zeros(labels.shape[0])
            for i, label in enumerate(avg_labels) :
                residuals_partial[label==labels] = residuals[i]
            residuals_full[mask] = residuals_partial

        residuals_full[~mask] = np.nan  
        
        for i in range(X.shape[1]):
            j = i % 4 + 1
            # strip number from column name, i.e., GABA1 -> GABA or GLUT2 -> GLUT
            cell = X.columns[i].strip('1234567890')
            row_dict = {'receptor':[receptor_name], 'r2':[r2], 'cell':[cell], 'comp':[j] }
            row_dict[f'coef'] = [fit.params[i]]
            row_dict[f'p'] = [fit.pvalues[i]]
            df = pd.concat([df, pd.DataFrame(row_dict)])

        # save residuals as fung.gii
        residuals_gii_filename = f'{output_dir}/residuals_{receptor_name}.func.gii'
        write_gifti(residuals_full, residuals_gii_filename)

        plot_receptor_surf([residuals_gii_filename], surface_filename, output_dir, label=f'residuals_{receptor_name}', cmap='RdBu_r', clobber=True)
    
    return df

from sklearn.decomposition import PCA
def pca(X, mask, label, surface_filename, output_dir, n=5):
    corr = np.corrcoef(X)
    pca = PCA(n_components=n)
    pca.fit(corr)
    pca_components = pca.components_.T

    for i in range(n):
        comp = pca_components[:,i]
        comp_full = np.zeros(mask.shape)
        comp_full[mask] = comp
        comp_full[~mask] = np.nan
        comp_filename = f'{output_dir}/surf_pca_{label}_{i+1}.func.gii'
        write_gifti(comp_full, comp_filename)
        print(comp_filename)
        plot_receptor_surf([comp_filename], surface_filename, output_dir, label=f'pca_{label}_{i+1}', cmap='RdBu_r', clobber=True)
        plt.close(); plt.clf(); plt.cla()
    print('Explained Variance:', pca.explained_variance_ratio_)
    print('Total Explained Variace:', pca.explained_variance_ratio_.sum())

    return pca_components


def cell_type_analysis(self, clobber:bool=False):

    output_dir = self.output_dir + '/cell_type_analysis/'   

    os.makedirs(output_dir, exist_ok=True)
    input_dir = 'data/mapped_volumes/surfaces/'
    feature_dir = f'{output_dir}/features/'
    
    receptor_surfaces, sphere_filename, output_dir,


    os.makedirs(feature_dir, exist_ok=True)

    cell_dens_surfaces = glob.glob(f'{input_dir}*L.func.gii')

    #for cell_dens_surface in cell_dens_surfaces:
    #    label = '.'.join(os.path.basename(cell_dens_surface).split('.')[0:-2])
    #    plot_receptor_surf([cell_dens_surface], surface_filename, feature_dir, label=f'{label}',  cmap='RdBu_r', threshold=[0,100])

    #surf_pca(
    #    cell_dens_surfaces, cortex_mask, surface_filename, sphere_filename, output_dir, n=10000, clobber=True
    #    )

    # Canonical Correlation Analysis between receptors and cell densities
    #cca_analysis(receptor_surfaces, cell_dens_surfaces, output_dir,  kind_x='receptor', kind_y='cell_density')

    df_csv = f'{output_dir}/gaba_glut_r2.csv'
    if not os.path.exists(df_csv) or clobber or True:
        mask = nib.load(cortex_mask_filename).darrays[0].data.astype(bool)

        X0, X0_mask = cca.preprocess(cell_dens_surfaces, kind='cell_density')
        Y0, Y0_mask = cca.preprocess(self.receptor_surfaces, kind='receptor')

        mask = mask * X0_mask #* Y0_mask

        X0 = X0[mask]
        Y0 = Y0[mask]

        X0 = np.round(X0,3)

        # Label duplicate rows in Y based ont the values in the rows
        if False :
            X0['label'] = X0.apply(lambda x: hash(tuple(x)), axis=1)
            Y0['label'] = X0['label']
            labels = X0['label'].values

            # Remove duplicate rows in Y
            X = X0.drop_duplicates(subset='label')
            Y = Y0.groupby('label').mean()

            # drop label column
            
            avg_labels = X['label'].values
            X = X.drop(columns='label')
        else :
            avg_labels = labels = None
            Y = Y0
            X = X0

        gaba_receptor_list = ['cgp5','musc','flum'] 
        glut_receptor_list = ['ampa','kain','mk80']

        gaba_cols = np.array([ 'GABA' in col for col in X.columns ])
        glu_cols = np.array([ 'GLU' in col for col in X.columns ]    )

        cell = X.iloc[:, gaba_cols + glu_cols ]


        receptor = Y[gaba_receptor_list+glut_receptor_list]
        rec_gaba = Y[gaba_receptor_list]
        rec_glut = Y[glut_receptor_list]
        from cca_analysis import cca_analysis

        # Apply PCA to the cell density data
        n = 2
        cell_comp = pca(cell, mask, 'cell_gaba', self.cortical_surface, output_dir, n=n)
        cell_comp = pd.DataFrame(cell_comp, columns=[f'COMP{i+1}' for i in range(n)] )
        #cell_glut_comp = pca(cell, mask, 'cell_glut', surface_filename, output_dir, n=n)
        #rec_gaba_comp = pca(rec_gaba, mask, 'rec_gaba', surface_filename, output_dir, n=n)
        #rec_glut_comp = pca(rec_glut, mask, 'rec_glut', surface_filename, output_dir, n=n)

        # get r2 between 1st component of cell and gaba receptors

        from scipy.stats import pearsonr, spearmanr
        df_list=[]
        for col in rec_gaba.columns:
            print(col)
            rho, p = spearmanr(cell_comp['COMP1'], rec_gaba[col])

            receptor = ligand_receptor_dict[col]
            df_list.append(pd.DataFrame({'receptor':[receptor], 'r2':[np.abs(rho)], 'p':[p],  'type':['GABA']}))

        for col in rec_glut.columns:
            print(col)
            rho, p = spearmanr(cell_comp['COMP1'], rec_glut[col])
            receptor = ligand_receptor_dict[col]
            df_list.append(pd.DataFrame({'receptor':[receptor], 'r2':[np.abs(rho)], 'p':[p],  'type':['GLUT']}))
        df = pd.concat(df_list)
        df.sort_values(by='r2', inplace=True, ascending=False)
        print(df) 
        plt.clf(); plt.cla(); plt.close()
        plt.figure(figsize=(16, 12))
        # set font size
        sns.set_theme(font_scale=1.5)
        sns.barplot(data=df, x='receptor', y='r2', hue='type', palette=sns.color_palette(['seagreen', 'salmon']) )
        # set y label to Spearman's Rho
        plt.ylabel('Spearman\'s Rho')
        plt.xlabel('Receptor')
        # set legend title to Receptor Type
        plt.legend(title='Receptor Type')

        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gaba_glut_r2.png')
        exit(0)

        # concatenate components 
        #columns = [ f'GABA{i+1}' for i in range(n)] + [f'GLUT{i}' for i in range(n)]
        #cell_comp = pd.DataFrame( np.concatenate([cell_gaba_comp, cell_glut_comp], axis=1), columns=columns )
        rec_gaba_comp = pd.DataFrame(rec_gaba_comp, columns=[f'GABA{i+1}' for i in range(n)] )


        df = receptor_cell_regr(
            cell_comp, rec_gaba_comp, output_dir, surface_filename, mask, avg_labels, labels, clobber=clobber
            )
        df.to_csv(df_csv)
    else:
        df = pd.read_csv(df_csv)


    df['receptor'] = df['receptor'].apply(lambda x: ligand_receptor_dict[x])    
    y = 'Regr. Coefficient'
    x = 'Cell Type - PCA Component'
    # rename coef column to Regr. Coefficient
    df = df.rename(columns={'coef':y, 'comp':x})

    print(df)
    plt.clf(); plt.cla()
    df2 = df.drop(columns='cell').groupby(['receptor']).mean()
    df2['Type'] = df2.index.map(cca.get_receptor_type)
    df2.sort_values(by='Type', inplace=True)
    print(df2); 
    sns.barplot(data=df2, x='receptor', y='r2', hue='Type', palette=sns.color_palette(['seagreen', 'salmon']) )
    # rotate x labels
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gaba_glut_r2.png')


    plt.clf(); plt.cla()
    sns.catplot(data=df, x=x, y=y, hue='cell', col='receptor', col_wrap=3, kind='bar', palette=sns.color_palette(['seagreen', 'salmon']) )
    plt.savefig(f'{output_dir}/gaba_glut_coef.png')
    exit(0) 

    plt.figure(figsize=(18, 12))
    plt.subplot(1, 2, 1)
    sns.regplot(x=receptor_gaba, y=cell_gaba)
    sns.despine()
    plt.title('GABA')
    plt.xlabel('Receptor')
    plt.ylabel('Cell Density')
    plt.subplot(1, 2, 2)
    sns.regplot(x=receptor_glut, y=cell_glut)
    sns.despine()
    plt.xlabel('Receptor')
    plt.ylabel('Cell Density')
    plt.title('GLU')
    plt.savefig(f'{output_dir}/receptor_vs_cell_density.png')
    print(receptor_gaba)
    print(cell_gaba)
    print(f'{output_dir}/receptor_vs_cell_density.png')
    exit(0)
    X['Receptor'] = X['Receptor'].apply(lambda x: cca.ligand_receptor_dict[x])

    X['Type'] = X['Receptor'].apply(cca.get_receptor_type) 
    Y['Type'] = Y['Morph.'].apply(cca.get_morph_type)

