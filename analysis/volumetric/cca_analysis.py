

from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
import os

global ligand_receptor_dict
ligand_receptor_dict={'ampa':'AMPA', 'kain':'Kainate', 'mk80':'NMDA', 'ly34':'mGluR2/3', 'flum':'GABA$_A$ Benz.', 'cgp5':'GABA$_B$', 'musc':'GABA$_A$ Agonist', 'sr95':'GABA$_A$ Antagonist', 'pire':r'Muscarinic M$_1$', 'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_3$','epib':r'Nicotinic $\alpha_4\beta_2$','oxot':r'Muscarinic M$_2$ (oxot)', 'praz':r'$\alpha_1$','uk14':r'$\alpha_2$ (agonist)','rx82':r'$\alpha_2$ (antagonist)', 'dpat':r'5-HT$_{1A}$','keta':r'5HT$_2$', 'sch2':r"D$_1$", 'dpmg':'Adenosine 1', 'cellbody':'Cell Body', 'myelin':'Myelin'}

def permute_cca(X, Y, n_components=3, n_permutations=1000):
    """
    Perform a permutation test for canonical correlation analysis (CCA)
    between two datasets X and Y.

    Parameters:
    - X: DataFrame, features for the first dataset.
    - Y: DataFrame, features for the second dataset.
    - n_components: int, number of canonical components to calculate.
    - n_permutations: int, number of permutations to use for the significance test.

    Returns:
    - p_values: array, p-values corresponding to the test of each canonical correlation.
    """

    def calculate_canonical_correlations(X, Y, n_components):
        cca = CCA(n_components=n_components)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        correlations = [pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(X_c.shape[1])]
        return correlations

    # Calculate the observed canonical correlations
    observed_correlations = calculate_canonical_correlations(X, Y, n_components)

    # Permutation test
    permuted_correlations = np.zeros((n_permutations, len(observed_correlations)))
    for i in range(n_permutations):
        # Shuffle Y and recalculate canonical correlations
        Y_permuted = Y[np.random.permutation(np.arange(Y.shape[0]).astype(int))]
        permuted_correlations[i] = calculate_canonical_correlations(X, Y_permuted, n_components)

    # Calculate p-values for each canonical correlation
    p_values = np.mean(permuted_correlations >= np.array([observed_correlations]), axis=0)

    return p_values


def file_list_to_dataframe(file_list, kind=None):
    df = pd.DataFrame()
    for i, file in enumerate(file_list):
        vtr = nib.load(file).darrays[0].data
        if len(vtr.shape) == 2:
            vtr = np.mean(vtr, axis=1)
        if kind == 'receptor':
            file = os.path.basename(file).split('_')[1]
        elif kind == 'cell_density':
            file = '_'.join(os.path.basename(file).split('_')[0:4])
        df[file] = vtr 

    return df

def preprocess(A, kind=None):
    if isinstance(A, list):
        df = file_list_to_dataframe(A, kind=kind)
    elif isinstance(A, type(pd.DataFrame())):
        # Check if the dataframe has the correct format
        # if the data frame is has columns average, label and annot, then it is in vertical form
        # and needs to be reformatted to horizontal form where each annot is a column and each row is a label 
        if 'avg' in A.columns and 'label' in A.columns and 'annot' in A.columns:
            df = A.pivot(index='label', columns='annot', values='avg')
        else :
            df = A
    else :
        raise ValueError("Input must be a list of file paths or a pandas dataframe.")

    idx = np.sum(df,axis=1)
    mask = ( idx  > 0 ) & ( ~ np.isnan(idx) )

    # zscore df by column
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    

    return df, mask


def get_receptor_type(string):
    if 'GABA' in string:
        return 'Inhibibitory'
    elif 'NMDA' in string or 'AMPA' in string or 'Kainate' in string:
        return 'Excitatory.'
    else:
        return 'Mod.'

def cca_analysis(X, Y, output_dir,kind_x=None, kind_y=None, n_components=3):

    os.makedirs(output_dir, exist_ok=True)

    X, _ = preprocess(X, kind=kind_x)
    Y, _ = preprocess(Y, kind=kind_y)
    print(X)
    print(Y); 


    assert X.shape[0] == Y.shape[0], f"X and Y must have the same number of samples but {X.shape} {Y.shape}."

    cca = CCA(n_components=n_components)
    cca.fit(X, Y)

    X_c, Y_c = cca.transform(X, Y)

    print("Canonical Correlation Coefficients:")
    plt.figure(figsize=(12, 18))

    output_png = f'{output_dir}/cca.png'
    for i in range(3):
        r, p = pearsonr(X_c[:, i], Y_c[:, i])
        plt.subplot(1, 3, i + 1)
        plt.scatter(x=X_c[:, i], y=Y_c[:, i], edgecolor='k', color='cyan')
        plt.title(f'Canonical Component {i+1}')
        plt.xlabel(f'X Component {i+1}')
        plt.ylabel(f'Y Component {i+1}')
        print( f'r={r:.2f}, p={p:.2f}')
    plt.tight_layout()
    plt.savefig(output_png)

    # create barplot for the weights of each component for X and Y
    plt.figure(figsize=(12, 18))
    output_png = f'{output_dir}/cca_weights.png'

    total_x_components = np.sum(np.abs(cca.x_weights_), axis=1)
    X_weights = pd.DataFrame({'Receptor': X.columns, 'Total':total_x_components, 'Component 1': cca.x_weights_[:, 0], 'Component 2': cca.x_weights_[:, 1], 'Component 3': cca.x_weights_[:, 2]})
    Y_weights = pd.DataFrame({'Morph.': Y.columns, 'Component 1': cca.y_weights_[:, 0], 'Component 2': cca.y_weights_[:, 1], 'Component 3': cca.y_weights_[:, 2]})
    def get_morph_type(string):
        if 'GLU' in string :
            return 'Excit.'
        elif 'GABA' in string:
            return 'Inhib.'
        else:
            return 'Other'
    

    X_weights['Receptor'] = X_weights['Receptor'].apply(lambda x: ligand_receptor_dict[x])

    X_weights['Type'] = X_weights['Receptor'].apply(get_receptor_type) 
    Y_weights['Type'] = Y_weights['Morph.'].apply(get_morph_type)

    plt.figure(figsize=(12, 18))
    for i in range(3):
        X_weights.sort_values(by=f'Component {i+1}', ascending=False, inplace=True)
        Y_weights.sort_values(by=f'Component {i+1}', ascending=False, inplace=True)
        plt.subplot(3, 2, i*2 + 1)
        sns.barplot(x='Receptor', y=f'Component {i+1}', data=X_weights, hue='Type')
        # rotate the x labels
        plt.xticks(rotation=90)
        plt.title(f'X Component {i+1}')
        plt.ylabel('Weight')
        plt.subplot(3, 2, i*2 + 2)
        sns.barplot(x='Morph.', y=f'Component {i+1}', data=Y_weights, hue='Type')
        plt.title(f'Y Component {i+1}')
    plt.title('Canonical Correlation Analysis Weights: Receptor Vs Peak Development Age')
    plt.tight_layout()
    plt.savefig(output_png)


    # Calc the explained variance
    correlations = [pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(n_components)]
    explained_variance = np.array(correlations)**2
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    
    print('Explained Variance:', explained_variance_ratio)

    X_weights.sort_values(by='Total', ascending=False, inplace=True)

    print('P-values')
    #p_values = permute_cca(X.values, Y.values, 3, 10000); 
    #print(p_values)
    return X_weights