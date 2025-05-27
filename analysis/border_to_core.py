import nibabel as nib
import numpy as np
from skimage.morphology import binary_erosion
from surface_analysis import project_to_surface
from brainbuilder.utils.mesh_io import load_mesh_ext


def find_core_border(roi, roi_ngh):
    """Find the border of a binary ROI defined on a surface mesh
    by seeing which indices have neighbors that are not in the ROI."""
    border = []
    core = []
    set_roi = set(np.unique(roi))
    for i, n in enumerate(roi_ngh):
        if len(n) != len(set(n) & set_roi) :
            border.append(i)
        else :
            core.append(i)

    return core, border

def mesh_erode(ngh, atlas, threshold=0.05):
    """Erode a binary ROI defined on a surface mesh"""
    print('Mesh erode')
    print(np.unique(atlas))

    for label in np.unique(atlas)[1:]:
        idx = np.where(atlas==label)
        core, border = find_core_border(ngh, idx)
        n_core = len(core)

        print('Border',len(border))
        ratio = len(core)/n_core
        while  ratio > threshold :
            core, border = find_core_border(ngh, core)
            n_core = len(core)
            ratio = len(core)/n_core
            print(len(border), ratio)
        idx = np.which(atlas==label)
        print(idx); exit(0) 
        core, border = find_core_border(ngh, idx)
        n_core = len(core)

        print('Border',len(border))
        ratio = len(core)/n_core
        while  ratio > threshold :
            core, border = find_core_border(ngh, core)
            n_core = len(core)
            ratio = len(core)/n_core
            print(len(border), ratio)


def border_to_core(coords, ngh, atlas, threshold=0.05):

    atlas_unique = np.unique(atlas)[1:]

    print(atlas_unique)
    for i, label in enumerate(atlas_unique):
        roi = np.zeros_like(atlas).astype(bool)
        roi[atlas == label] = 1
        roi_sum = roi.sum()
        ratio = 1
        while ratio > threshold :
            new_roi = binary_erosion(roi) 
            border = roi & (~ new_roi)

            ratio = new_roi.sum() / roi_sum
            roi = new_roi
            print(label, ratio, roi.sum(), roi_sum)




atlas_fn="data/volumes/MEBRAINS_segmentation_sym_20230626_mirror.nii"
mask_fn="data/volumes/MEBRAINS_segmentation_NEW_gm_left.nii.gz"
gm_surf_fn="data/surfaces/lh.MEBRAINS_0.5mm_0.0.surf.gii"
wm_surf_fn="data/surfaces/lh.MEBRAINS_0.5mm_1.0.surf.gii"

surface_data_list = project_to_surface(
    [atlas_fn],
    wm_surf_fn, 
    gm_surf_fn, 
    'outputs/border_to_core/',
    n = 10,
    agg_func=np.median,
    )
surf_atlas = surface_data_list[0]

coords, ngh = load_mesh_ext(gm_surf_fn)
mesh_erode(ngh, np.load(surf_atlas), threshold=0.05)
#border_to_core(coords, ngh, surf_atlas, threshold=0.05)