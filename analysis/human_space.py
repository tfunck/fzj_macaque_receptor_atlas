import subprocess
import os
import nibabel as nib

from brainstat.context.meta_analysis import meta_analytic_decoder

def transform_macaque_to_human(self, clobber:bool=False):
    """
    Transform macaque surface data to human
    """

    for label in self.receptor_surfaces :
        outstr = os.path.basename(label).split('.')[0]
        label_rsl_str = f'{self.output_dir}/{outstr}_space-macaque-rsl'
        label_human_str = f'{self.output_dir}/{outstr}_space-human'
        
        if not os.path.exists(f'{label_rsl_str}.func.gii') or self.clobber or True:
            cmd = f'msmresample {self.sphere_surface} {label_rsl_str} -project {self.macaque_to_human_tfm_sphere} -labels {label}'
            print(cmd) ; 
            subprocess.run(cmd, shell=True, executable='/bin/bash')

        assert os.path.exists(f'{label_rsl_str}.func.gii'), f'Error: {label_rsl_str}.func.gii not found'


        if not os.path.exists(f'{label_human_str}.func.gii') or self.clobber:

            cmd = f'msmresample {self.macaque_to_human_tfm_sphere} {label_human_str} -project  {self.human_sphere} -labels {label_rsl_str}.func.gii'
            subprocess.run(cmd, shell=True, executable='/bin/bash') 

        assert os.path.exists(f'{label_human_str}.func.gii'), f'Error: {label_human_str}.func.gii not found'

        self.receptor_surfaces_tfm.append(f'{label_human_str}.func.gii')
    


def decode(self, clobber:bool=False):
    """
    Transform macaque surface data to human and use human functional data to decode
    """

    for fn in self.receptor_surfaces_tfm:
        
        print(fn)
        receptor_array = nib.load(fn).darrays[0].data

        meta_analysis = meta_analytic_decoder("fsaverage5", receptor_array.flatten(),"spearman")
        
        print(meta_analysis)