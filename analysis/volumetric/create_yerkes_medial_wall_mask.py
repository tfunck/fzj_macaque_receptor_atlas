import nibabel as nib
import numpy as np

fn='~/Downloads/Yerkes19_Parcellations_v2.32k_fs_LR.dlabel.nii'
img = nib.load(fn)
x = img.dataobj[0]
np.unique(x)
n = int(x.shape[0]/2)
print(n)
print(np.bincount(x[0:n].astype(np.int64)) )
print(np.bincount(x[n:].astype(np.int64)) )
mask = np.array(x[0:n])

# write mask to gifti func.gii file
mask[ mask>0 ]  = -1
mask[ mask==0 ] = 1
mask[ mask==-1 ] = 0

mask_array = nib.gifti.GiftiDataArray(data=mask, intent='NIFTI_INTENT_LABEL', datatype='NIFTI_TYPE_FLOAT32')
mask_img = nib.gifti.GiftiImage(darrays=[mask_array])
print('data/surfaces/Yerkes19.L_medial_wall_mask.func.gii')
nib.save(mask_img, 'data/surfaces/Yerkes19.L_medial_wall_mask.func.gii')