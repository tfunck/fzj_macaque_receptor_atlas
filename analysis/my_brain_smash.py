from brainsmash.workbench.geo import cortex
from brainsmash.mapgen.memmap import txt2memmap
from brainsmash.mapgen.stats import pearsonr, pairwise_r
from brainsmash.mapgen.stats import nonparp
from brainsmash.mapgen.base import Base


#surface = "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
#cortex(surface=surface, outfile="./outputs/LeftDenseGeodesicDistmat.txt", euclid=False)
#output_files = txt2memmap(dist_mat_fin, output_dir, maskfile=None, delimiter=' ')

def calculate_correlations(x:np.array, y:np.array, dist_mat_fin:str, output_dir:str, clobber:bool=False):
    # instantiate class and generate 1000 surrogates
    gen = Base(y, dist_mat_fin)  # note: can pass numpy arrays as well as filenames
    surrogate_maps = gen(n=1000)

    surrogate_brainmap_corrs = pearsonr(x, surrogate_maps).flatten()

    naive_surrogates = np.array([np.random.permutation(y) for _ in range(1000)])
    naive_brainmap_corrs = pearsonr(x, naive_surrogates).flatten()

    test_stat = pearsonr(x, y)[0]

    naive_p = nonparp(test_stat, naive_brainmap_corrs)
    p = nonparp(test_stat, surrogate_brainmap_corrs)

    print("Spatially naive p-value:", naive_p )
    print("SA-corrected p-value:", p )

    return p

