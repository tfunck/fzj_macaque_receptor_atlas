""""""
import os
import numpy as np
import nibabel as nib
import utils
from scipy.ndimage import gaussian_filter


def sum_volumes(volume_list, mask_vol, zscore=False, gauss_sd=0):
    n=0
    idx = mask_vol > 0
    for i, ligand in enumerate(volume_list):
        rec_vol = nib.load(ligand).get_fdata()
        if zscore:
            rec_vol[idx] = (rec_vol[idx] - rec_vol[idx].mean()) / rec_vol[idx].std()
            rec_vol[~idx] = 0

        if i == 0:
            vol = rec_vol 
        else:
            vol += rec_vol
        n += 1
    
    vol = vol / n
    if gauss_sd > 0:
        vol = gaussian_filter(vol,gauss_sd)

    return vol



def create_volume(receptor_files, mask_file, target_strings, output_filename, clobber=False):

    file_list = utils.get_files_from_list( receptor_files, target_strings) 
    
    if not os.path.exists(output_filename) or clobber:
        mask_vol = nib.load(mask_file).get_fdata()
        
        ref_img = nib.load(file_list[0])

        summed_vol = sum_volumes(file_list, mask_vol )

        nib.Nifti1Image(summed_vol, ref_img.affine, ref_img.header).to_filename(output_filename)
    else :
        summed_vol = nib.load(output_filename).get_fdata()

    return summed_vol, file_list

def calc_ratio(vol0,vol1,mask):
    out = np.zeros_like(vol0)
    idx = (mask > 0) & (vol0 > 0) & (vol1 > 0)
    out[idx] = vol0[idx] / vol1[idx]
    out = np.clip(out, 0, 10)
    print(f'Max ratio: {out.max()}, Min ratio: {out.min()}')
    return out


def ratio_analysis(receptor_files, mask_file, output_dir, clobber=False):
    """Calculate ratios of receptor volumes"""
    os.makedirs(output_dir, exist_ok=True)

    exh_filename = f'{output_dir}/macaque_exh.nii.gz'
    inh_filename = f'{output_dir}/macaque_inh.nii.gz'
    mod_filename = f'{output_dir}/macaque_mod.nii.gz'

    exh_inh_filename = f'{output_dir}/macaque_ratio_ex_inh.nii.gz'
    gabaa_gabab_filename = f'{output_dir}/macaque_ratio_gabaa_gabab.nii.gz'
    ampakain_mk80_filename = f'{output_dir}/macaque_ratio_ampakain_mk80.nii.gz'
    inhexh_mod_filename = f'{output_dir}/macaque_ratio_inhexh_mod.nii.gz'

    glut_filename = f'{output_dir}/macaque_glutamate.nii.gz'
    acetyl_filename = f'{output_dir}/macaque_acetylcholine.nii.gz'
    nodrad_filename = f'{output_dir}/macaque_noradrenaline.nii.gz'
    serotonin_filename = f'{output_dir}/macaque_serotonin.nii.gz'
    dopamine_filename = f'{output_dir}/macaque_dopamine.nii.gz'

    output_volumes = [
        exh_filename, inh_filename, mod_filename, 
        gabaa_gabab_filename, exh_inh_filename,  inhexh_mod_filename, ampakain_mk80_filename,
        glut_filename, acetyl_filename, nodrad_filename,  serotonin_filename #, dopamine_filename 
        ]

    cmap_label_list = [
        'Ex',  'Inh', 'Mod', 
        'GABAa/GABAb',  'Ex/Inh', '(Inh+Ex)/Mod', 'Ex. Ion/Metab',
        'Glutamate', 'Acetylcholine', 'Noradrenaline', 'Serotonin' #, 'Dopamine'
        ]

    output_dict = dict(zip(output_volumes, cmap_label_list))


    inh_vol, inh_list = create_volume(receptor_files, mask_file, ['musc', 'cgp5', 'flum'], inh_filename, clobber=clobber)
    exh_vol, exh_list = create_volume(receptor_files, mask_file, ['ampa', 'kain', 'mk80'], exh_filename, clobber=clobber)
    mod_vol, mod_list = create_volume(receptor_files, mask_file, ['dpat', 'uk14', 'oxot', 'keta', 'sch2', 'pire'], mod_filename, clobber=clobber)

    if False in [os.path.exists(file) for file in output_volumes]:

        mask_vol = nib.load(mask_file).get_fdata()

        affine = nib.load(receptor_files[0]).affine

        

        print('Exhibitory / Inhibitory')
        exh_inh_vol = calc_ratio(exh_vol, inh_vol, mask_vol)
        nib.Nifti1Image(exh_inh_vol, affine).to_filename(exh_inh_filename)

        flum_vol = utils.get_volume_from_list(receptor_files, 'flum', zscore=False)
        cgp5_vol = utils.get_volume_from_list(receptor_files, 'cgp5', zscore=False)
        musc_vol = utils.get_volume_from_list(receptor_files, 'musc', zscore=False)
        ampa_vol = utils.get_volume_from_list(receptor_files, 'ampa', zscore=False)
        kain_vol = utils.get_volume_from_list(receptor_files, 'kain', zscore=False)
        mk80_vol = utils.get_volume_from_list(receptor_files, 'mk80', zscore=False)

        gabaa_gabab = calc_ratio((musc_vol+flum_vol)/2, cgp5_vol, mask_vol )
        nib.Nifti1Image(gabaa_gabab, affine).to_filename(gabaa_gabab_filename)
    
        ampakain_mk80 = calc_ratio( (ampa_vol + kain_vol)/2, mk80_vol, mask_vol) 
        nib.Nifti1Image(ampakain_mk80, affine).to_filename(ampakain_mk80_filename)

        print('Inhibitory + Exhibitory / Modulatory')
        inhexh_mod_vol = calc_ratio( (inh_vol + exh_vol), mod_vol, mask_vol)
        nib.Nifti1Image(inhexh_mod_vol, affine).to_filename(inhexh_mod_filename)

    return output_dict, [inh_list, exh_list, mod_list]