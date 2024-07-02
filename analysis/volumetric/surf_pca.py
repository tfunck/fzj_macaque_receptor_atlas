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


def save_partial_vector(vector, medial_wall_mask, idx, surface_filename, sphere_filename, output_dir, label, cmap='RdBu_r', clobber=False):
    full_comp = interpolate_gradient_over_surface(
        vector,
        surface_filename,
        sphere_filename,
        output_dir,
        label,
        idx,
        clobber=clobber
    )
    full_comp[medial_wall_mask] = np.nan
    comp_filename = f'{output_dir}/surf_{label}.gii'
    write_gifti(full_comp, comp_filename)    

    plot_receptor_surf([comp_filename], surface_filename, output_dir, label=f'{label}',  cmap=cmap, threshold=[0,100])

def surf_pca(
        features_files, medial_wall_mask, surface_filename, sphere_filename, output_dir, n=10000, clobber=True
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
    
    x,y,z = features.shape
    features = np.swapaxes(features, 0, 1)

    print('Pairwise Distance Correlation')
    if not os.path.exists(corr_filename) or\
        not os.path.exists(idx_filename) or\
        not os.path.exists(features_filename) or\
        clobber :

        valid_idx = np.where(~medial_wall_mask)[0]
        idx = np.random.choice(valid_idx, n, replace=False)
        features = features[idx,:,:]
        corr = pairwise_distance_correlation(features)

        np.save(corr_filename, corr)
        np.save(idx_filename, idx)
        np.save(features_filename, features)
    else:   
        corr = np.load(corr_filename)
        idx = np.load(idx_filename)
        features = np.load(features_filename)

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
    for i in range(pca_components.shape[1]):
        save_partial_vector(
                pca_components[:,i], medial_wall_mask, idx, surface_filename, sphere_filename, output_dir, f'PC{i+1}', clobber=False
                )
    features = features.reshape(features.shape[0],-1)
    for eps in np.arange(2,20,2) :
        labels = cluster.KMeans(eps).fit(features).labels_
        
        save_partial_vector(
                labels, medial_wall_mask, idx, surface_filename, sphere_filename, output_dir, f'seg_eps-{eps}', cmap='nipy_spectral', clobber=True
                )

    return corr

     


    
