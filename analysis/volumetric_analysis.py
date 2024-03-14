
import argparse
import nibabel as nib
import glob
import os
import numpy as np

from skimage.transform import resize

from volumetric.volumetric_ratio_analysis import ratio_analysis
from volumetric.volumetric_gradient_analysis import volumetric_gradient_analysis
import utils


def resize_mask_to_receptor_volume(mask_file, receptor_file, output_dir, order=0):
    """Resize mask to the same dimensions as the receptor volume."""
    base = os.path.basename(mask_file).replace('.nii.gz','_rsl.nii.gz')
    mask_rsl_file = f"{output_dir}/{base}"

    if not os.path.exists(mask_rsl_file):
        mask = nib.load(mask_file)
        mask_vol = mask.get_fdata()

        receptor = nib.load(receptor_file)

        assert np.unique(mask_vol).size > 1, 'Mask is empty'
        print(receptor.shape, mask_vol.min(), mask_vol.max())
        mask_rsl = resize(mask_vol.astype(np.float32), receptor.shape, order=order)

        assert np.unique(mask_rsl).size > 1, 'Resized mask is empty'

        nib.Nifti1Image(mask_rsl, receptor.affine).to_filename(mask_rsl_file)

    return mask_rsl_file


def t1t2_analysis(mask_rsl_file, hist_volumes, t1t2_filename, output_dir):
    import nibabel as nib
    import matplotlib.pyplot as plt
    
    
    mask_vol = nib.load(mask_rsl_file).get_fdata()

    myelin_filename = utils.get_files_from_list(hist_volumes, ['myelin'])[0]

    myelin_img = nib.load(myelin_filename)
    myelin_vol = myelin_img.get_fdata()
    
    t1t2_rsl_file = resize_mask_to_receptor_volume( t1t2_filename, myelin_filename, output_dir, order=3)
    t1t2_vol = nib.load(t1t2_rsl_file).get_fdata()

    idx = (mask_vol > 0) & (t1t2_vol > 0) & (myelin_vol > 0)

    x=myelin_vol[idx]
    y=t1t2_vol[idx]   

    plt.scatter(x,y)
    plt.savefig(f'{output_dir}/t1t2_vs_myelin.png')

    # fit linear regression model for x and y
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x.reshape(-1,1), y)
    # get model residuals
    residuals = y - model.predict(x.reshape(-1,1))

    res_vol = np.zeros_like(myelin_vol)
    res_vol[idx] = residuals

    nib.Nifti1Image(res_vol, myelin_img.affine).to_filename(f'{output_dir}/t1t2_residuals.nii.gz')
    

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Volumetric Gradient Analysis')
    parser.add_argument('-m', dest='mask_file', default='data/volumes/MEBRAINS_segmentation_NEW_gm_left.nii.gz', type=str, help='Path to mask file')
    parser.add_argument('-i', dest='input_dir', type=str, default='data/reconstruction/', help='Path to receptor volumes')
    parser.add_argument('-o', dest='output_dir', type=str, default='outputs/volumetric', help='Path to output directory')
    parser.add_argument('-n', dest='n', default=10000, type=int, help='Number of random voxels to sample')

    args = parser.parse_args()
    t1t2_filename = 'data/volumes/MEBRAINS_T1T2_masked.nii.gz'

    os.makedirs(args.output_dir, exist_ok=True)
    receptor_dir = f"{args.input_dir}/receptor/"
    hist_dir = f"{args.input_dir}/hist/"

    receptor_volumes = glob.glob(f'{receptor_dir}/*nii.gz')
    hist_volumes = glob.glob(f'{hist_dir}/*nii.gz')

    mask_rsl_file = resize_mask_to_receptor_volume(args.mask_file, receptor_volumes[0], args.output_dir)

    volumetric_gradient_analysis(mask_rsl_file, receptor_volumes, args.output_dir, args.n)

    ratio_dir = f'{args.output_dir}/ratios/'
    ratio_analysis(receptor_volumes, mask_rsl_file, ratio_dir )
    #t1t2_analysis(mask_rsl_file, hist_volumes, t1t2_filename, args.output_dir)




