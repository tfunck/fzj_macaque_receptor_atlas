""""""
import os
import numpy as np
import nibabel as nib
import utils
from surf_utils import write_gifti, plot_receptor_surf

#receptor_volumes, mask_rsl_file, ratio_dir,
def sum_volumes(surface_list, zscore=False, gauss_sd=0):
    n=0

    for i, ligand in enumerate(surface_list):
        
        rec_vol = nib.load(ligand).darrays[0].data  

        idx = np.isnan(rec_vol)

        if zscore:
            rec_vol[idx] = (rec_vol[idx] - rec_vol[idx].mean()) / rec_vol[idx].std()

        if i == 0:
            output = rec_vol 
        else:
            output += rec_vol
        
        n += 1
    
    output = output / n

    return output



def create_summed_surface(
        receptor_files:list, 
        target_strings:str, 
        output_dir:str, 
        cortical_surface:str, 
        output_filename:str,
        label:str='', 
        clobber:bool=False
    ):

    file_list = utils.get_files_from_list( receptor_files, target_strings) 
    
    if not os.path.exists(output_filename) or clobber:
        
        summed_receptors = sum_volumes(file_list)

        write_gifti(summed_receptors, output_filename)
        
        print(output_filename, cortical_surface); print(output_dir); 
        plot_receptor_surf(
            [output_filename], cortical_surface, output_dir, label=label, cmap='RdBu_r', threshold=[0,100]
            )

    else :
        summed_receptors = nib.load(output_filename).darrays[0].data

    return summed_receptors, file_list

def calc_ratio(vol0, vol1):
    out = np.zeros_like(vol0)

    idx = (vol0 > 0) & (vol1 > 0)

    out[idx] = vol0[idx] / vol1[idx]
    
    out[~idx] = 0

    out = np.clip(out, 0, 10)

    print(f'Max ratio: {out.max()}, Min ratio: {out.min()}')
    return out


def ratio_analysis(self, clobber=False):
    """Calculate ratios of receptor volumes"""

    output_dir = self.output_dir + '/ratios/'

    os.makedirs(output_dir, exist_ok=True)

    receptor_files = self.receptor_surfaces

    exh_filename = f'{output_dir}/macaque_exh.func.gii'
    inh_filename = f'{output_dir}/macaque_inh.func.gii'
    mod_filename = f'{output_dir}/macaque_mod.func.gii'

    exh_inh_filename = f'{output_dir}/macaque_ratio_ex_inh.func.gii'
    gabaa_gabab_filename = f'{output_dir}/macaque_ratio_gabaa_gabab.func.gii'
    ampakain_mk80_filename = f'{output_dir}/macaque_ratio_ampakain_mk80.func.gii'
    inhexh_mod_filename = f'{output_dir}/macaque_ratio_inhexh_mod.func.gii'

    glut_filename = f'{output_dir}/macaque_glutamate.func.gii'
    acetyl_filename = f'{output_dir}/macaque_acetylcholine.func.gii'
    nodrad_filename = f'{output_dir}/macaque_noradrenaline.func.gii'
    serotonin_filename = f'{output_dir}/macaque_serotonin.func.gii'
    dopamine_filename = f'{output_dir}/macaque_dopamine.func.gii'

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

    # ,  'oxot', 'dpat', 'uk14' ]
    inh_vol, inh_list = create_summed_surface(
        receptor_files, ['musc', 'cgp5', 'flum'], output_dir, self.cortical_surface, inh_filename, label='Inh.', clobber=clobber
        )
        
    exh_vol, exh_list = create_summed_surface(
        receptor_files, ['ampa', 'kain', 'mk80', 'keta', 'pire'], output_dir, self.cortical_surface, exh_filename, label='Exh.', clobber=clobber
        )

    mod_vol, mod_list = create_summed_surface(
        receptor_files, ['dpat', 'uk14', 'oxot', 'keta', 'sch2', 'pire'], output_dir, self.cortical_surface, mod_filename, label='Mod.', clobber=clobber
        )

    if False in [os.path.exists(file) for file in output_volumes]:
        print('Exhibitory / Inhibitory')
        exh_inh_vol = calc_ratio(exh_vol, inh_vol)

        write_gifti(exh_inh_vol, exh_inh_filename)

        flum_vol = utils.get_volume_from_list(receptor_files, 'flum', zscore=False)
        cgp5_vol = utils.get_volume_from_list(receptor_files, 'cgp5', zscore=False)
        musc_vol = utils.get_volume_from_list(receptor_files, 'musc', zscore=False)
        ampa_vol = utils.get_volume_from_list(receptor_files, 'ampa', zscore=False)
        kain_vol = utils.get_volume_from_list(receptor_files, 'kain', zscore=False)
        mk80_vol = utils.get_volume_from_list(receptor_files, 'mk80', zscore=False)

        gabaa_gabab = calc_ratio((musc_vol+flum_vol)/2, cgp5_vol)
        write_gifti(gabaa_gabab, gabaa_gabab_filename)
    
        ampakain_mk80 = calc_ratio( (ampa_vol + kain_vol)/2, mk80_vol) 
        write_gifti(ampakain_mk80, ampakain_mk80_filename)

        print('Inhibitory + Exhibitory / Modulatory')
        inhexh_mod_vol = calc_ratio( (inh_vol + exh_vol), mod_vol)
        write_gifti(inhexh_mod_vol, inhexh_mod_filename)

        plot_receptor_surf(
            [exh_inh_filename], self.cortical_surface, output_dir, label='ex_inh', cmap='RdBu_r', threshold=[0,100]
            )

    return output_dict, [inh_list, exh_list, mod_list]