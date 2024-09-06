import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from volumetric.surf_utils import interpolate_gradient_over_surface, plot_receptor_surf, write_gifti
from brainbuilder.utils.mesh_utils import load_mesh_ext
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed




def distance_covariance(X, Y):
    n = X.shape[0]
    a = squareform(pdist(X, 'euclidean'))
    b = squareform(pdist(Y, 'euclidean'))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    return np.sqrt((A * B).sum() / (n ** 2))

def distance_correlation(X, Y):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    dCovXY = distance_covariance(X, Y)
    dCovXX = distance_covariance(X, X)
    dCovYY = distance_covariance(Y, Y)
    return dCovXY / np.sqrt(dCovXX * dCovYY)


def pairwise_distance_correlation(samples):
    n_samples = len(samples)
    correlation_matrix = np.zeros((n_samples, n_samples))

    # Parallelize row calculation
    def row_correlation( i, samples):
        n_samples = len(samples)
        row = np.zeros(n_samples)
        if i % 100 == 0 : 
            print('Completed:', 100*i/n_samples,end='\r')

        for j in range(i, n_samples):
            if i == j:
                row[j] = 1.0
            else:
                corr = distance_correlation(samples[i], samples[j])
                row[j] = corr
        return (i,row)

    res = Parallel(n_jobs=-1)(delayed(row_correlation)(i,samples) for i in range(n_samples))

    for i, row in res:
        correlation_matrix[i] = row
    # fill lower triangle
    for i in range(n_samples):
        for j in range(i):
            correlation_matrix[i, j] = correlation_matrix[j, i]

    return correlation_matrix


def calc_diff(features, n=10000):
    # features = (receptor, vertex, depth)
    # calculate difference between features voxels across features

    ar_expanded_1 = features_reduced[:, np.newaxis, :, :]  # Shape (y, 1, x, z)
    ar_expanded_2 = features_reduced[np.newaxis, :, :, :]  # Shape (1, y, x, z)

    diff = np.sum(np.abs(ar_expanded_1 - ar_expanded_2), axis=(2, 3))
    return diff


def save_partial_vector(vector, cortex_mask, idx, surface_filename, sphere_filename, output_dir, label, cmap='RdBu_r', clobber=False):
    full_comp = interpolate_gradient_over_surface(
        vector,
        surface_filename,
        sphere_filename,
        output_dir,
        label,
        idx,
        clobber=clobber
    )
    full_comp[~cortex_mask] = np.nan
    comp_filename = f'{output_dir}/surf_{label}.gii'
    write_gifti(full_comp, comp_filename)    

    if not os.path.exists(comp_filename) or clobber:
        plot_receptor_surf([comp_filename], surface_filename, output_dir, label=f'{label}',  cmap=cmap, threshold=[0,100])
    
    return comp_filename

def surf_pca(
        features_files, cortex_mask, surface_filename, sphere_filename, output_dir, n=10000, clobber=False
        ):
    os.makedirs(output_dir, exist_ok=True)
    # Load features into numpy array from list of gifti files
    features_npy_filename = f'{output_dir}/features.npy'
    corr_filename = f'{output_dir}/corr.npy'
    idx_filename = f'{output_dir}/idx.npy'
    features_filename = f'{output_dir}/features.npy'
    if not os.path.exists(features_npy_filename) or clobber:
        features = np.array([ np.array(nib.load(file).darrays[0].data) for file in features_files ])
        np.save(features_npy_filename, features)
    else :
        features = np.load(features_npy_filename)
    
    features = np.swapaxes(features, 0, 1)

    # z-score features by column
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    print('Features:', features.shape)

    print('Pairwise Distance Correlation')
    if not os.path.exists(corr_filename) or\
        not os.path.exists(idx_filename) or\
        not os.path.exists(features_filename) or\
        clobber :

        n_feature_dims = len(features.shape)

        if n_feature_dims == 3:
            axis=(1,2)
        elif n_feature_dims == 2:
            axis=(1,)
        else:
            raise ValueError('Features shape not understood, should be 2 or 3')

        # select n random features within mask and dont' have nan values
        print(np.sum(features,axis=axis))
        idx1 = ~ np.isnan(np.sum(features,axis=axis))
        idx2 = np.sum(features,axis=axis) > 0
        print(idx1.shape, idx2.shape)
        print(np.sum(idx1), np.sum(idx2))
        valid_idx = np.where(cortex_mask &  idx1 & idx2 )[0]
        print(valid_idx)
        assert len(valid_idx) > 0, 'Not enough valid features'
        idx = np.random.choice(valid_idx, n, replace=False)
        features = features[idx,:]

        assert np.sum(np.isnan(features)) == 0 , 'Nan values in features'

        if len(features.shape) == 3:
            corr = pairwise_distance_correlation(features)
        elif len(features.shape) == 2:
            corr = np.corrcoef(features)
        else :
            raise ValueError('Features shape not understood')

        corr = np.corrcoef(features.T)
        v0, v1 = np.percentile(corr, [5, 95])
        
        plt.figure(figsize=(7, 7))
        plt.imshow(corr, vmin=v0, vmax=v1, cmap='RdBu_r')
        plt.colorbar()
        plt.savefig(f'{output_dir}/corr.png')   

        np.save(corr_filename, corr)
        np.save(idx_filename, idx)
        np.save(features_filename, features)
    else:   
        corr = np.load(corr_filename)
        idx = np.load(idx_filename)
        features = np.load(features_filename)

    assert np.sum(np.isnan(corr)) == 0 , 'Nan values in correlation matrix'

    # Calculate PCA
    print('PCA')
    pca = PCA(n_components=5)
    pca.fit(corr)
    pca_features = pca.transform(corr)
    pca_components = pca.components_.T

    print('Explained Variance:', pca.explained_variance_ratio_)
    print('Total Explained Variace:', pca.explained_variance_ratio_.sum())
    # plot PCA 
    plt.clf(); plt.cla()
    plt.figure(figsize=(7, 7))
    plt.scatter(pca_features[:,0], pca_features[:,1], alpha=0.3)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(f'{output_dir}/pca.png')
    plt.clf(); plt.cla(); plt.close()

    print('Componenets')
    print(pca_components.shape)
    component_list = []
    for i in range(pca_components.shape[1]):
        comp_filename = save_partial_vector(
                pca_components[:,i], cortex_mask, idx, surface_filename, sphere_filename, output_dir, f'PC{i+1}', clobber=clobber
                )
        component_list.append(comp_filename)

    #features = features.reshape(features.shape[0],-1)
    #for eps in np.arange(2,20,2) :
    #    labels = cluster.KMeans(eps).fit(features).labels_
    #    
    #    save_partial_vector(
    #            labels, cortex_mask, idx, surface_filename, sphere_filename, output_dir, f'seg_eps-{eps}', cmap='nipy_spectral', clobber=True
    #            )

    return component_list

     


    
