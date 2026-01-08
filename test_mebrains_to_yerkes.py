import os
import subprocess
import nibabel as nib
from surfalign.surfalign import surfalign


def shell(cmd):
    print(cmd)
    subprocess.run([
        cmd
    ], shell=True, executable='/bin/bash' )

yerkes19_lh = 'data/surfaces/MacaqueYerkes19.L.mid.32k_fs_LR.surf.gii'
mebrains_lh = 'data/surfaces/lh.MEBRAINS_0.5mm_0.5.surf.gii' 
fixed_sphere = 'data/surfaces/MacaqueYerkes19.L.sphere.32k_fs_LR.surf.gii'

mebrains_mask='data/surfaces/lh.MEBRAINS_medial_wall_mask.func.gii'
yerkes_mask='data/surfaces/Yerkes19.L_medial_wall_mask.func.gii'

mebrains_to_yerkes, yerkes_sphere_rsl, mebrains32k = surfalign(
    yerkes19_lh, 
    mebrains_lh, 
    'mebrains_to_yerkes19',
    levels = 2, 
    #fixed_sphere = fixed_sphere, 
    metric_list_heir = [['y','z'], ['curv','sulc']], 
    mov_param = {'n_curv':225, 'n_sulc':20},
	fix_param = {'n_curv':10, 'n_sulc':10},
	fixed_mask = yerkes_mask,
	moving_mask = mebrains_mask, 
    title = 'mebrains_to_yerkes19',
	clobber = False,
	verbose = False
    )

exit(0)

#mebrains_to_yerkes='../neuromaps-nhp-prep/output/aligned/mebrains_to_yerkes19/1/lh.lh.MEBRAINS.mid_metrics_curv_sulc_warped_sphere.reg.surf.gii'
#mebrains32k = '../neuromaps-nhp-prep/output/aligned/mebrains_to_yerkes19/n-32492_lh.MEBRAINS.mid_sphere.sphere.gii'

base_dir = "/home/thomas-funck/projects/neuromaps-nhp-prep/output/"
metric_dir = os.path.join(base_dir, "metrics")

mebrains_to_hcp = "/media/windows/projects/fzj_macaque_receptor_atlas/outputs/0.25mm/surfaces/mebrains2human_surf_tfm.surf.gii"
mebrains_orig = "/home/thomas-funck/projects/fzj_macaque_receptor_atlas/data/surfaces/lh.MEBRAINS_0.5mm_sphere.surf.gii"

yerkes_sphere = "/media/windows/projects/fzj_macaque_receptor_atlas/data/surfaces/MacaqueYerkes19.L.sphere.32k_fs_LR.surf.gii"
#yerkes_sphere = "../neuromaps-nhp-prep/data/yerkes/surfaces/MacaqueYerkes19.L.sphere.32k_fs_LR.surf.gii"
yerkes_sphere = "../neuromaps-nhp-prep/output/aligned/mebrains_to_yerkes19/MacaqueYerkes19.L.mid.32k_fs_LR_sphere.sphere.gii"

yerkes_to_hcp = "/home/thomas-funck/projects/neuromaps-nhp-prep/data/macaque-to-human/L.macaque-to-human.sphere.reg.32k_fs_LR.surf.gii"
yerkes_cortex = "../neuromaps-nhp-prep/data/yerkes/surfaces/MacaqueYerkes19.L.mid.32k_fs_LR.surf.gii"

hcp_sphere = "S1200.L.sphere.32k_fs_LR.surf.gii"
hcp_cortex = 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'


#check if input files exist
for fn in [mebrains_to_yerkes, mebrains_to_hcp, mebrains_orig, mebrains32k, yerkes_sphere, yerkes_to_hcp, hcp_sphere, hcp_cortex]:
	assert os.path.exists(fn), f"File not found: {fn}"
    # Set structure with wb_command
	if not 'sphere' in fn:
		shell(f"wb_command -set-structure {fn} CORTEX_LEFT")

output_dir = "/home/thomas-funck/projects/fzj_macaque_receptor_atlas/temp/mebrains_to_hcp/"
os.makedirs(output_dir, exist_ok=True)

cmd = f'wb_command -surface-sphere-project-unproject {mebrains_to_yerkes} {yerkes_sphere} {yerkes_to_hcp} {mebrains_to_hcp}'
shell(cmd)

clobber = True

def dims(fn):
	print(fn)
	print(nib.load(fn).darrays[0].data.shape)
	
def check_equal_dims(fn1, fn2):
    a = nib.load(fn1).darrays[0].data.shape[0]
    b = nib.load(fn2).darrays[0].data.shape[0]
    assert a == b, f"Dimensions do not match: {a} != {b}, for files \n\t{fn1}\n\t{fn2}"


#labels = ["mebrains_to_yerkes19/metrics/1/lh.lh.MEBRAINS_0.5mm_0.5_metrics_curv_sulc.func.gii"] 
labels = [ "outputs/0.25mm/0.06-0.33/macaque_oxot_smoothed.func.gii"]
for label in labels:
	outstr = output_dir+'/'+os.path.basename(label).replace('.func.gii', '')

	label_mebrains32k = f"{outstr}_mebrains32k.func.gii"
	label_yerkes_rsl = f"{outstr}_yerkes_rsl.func.gii"
	label_yerkes = f"{outstr}_yerkes.func.gii"
	label_yerkes_to_hcp = f"{outstr}_yerkes-to-hcp.func.gii"
	label_mebrains_to_hcp = f"{outstr}_mebrains-to-hcp.func.gii"

	for file in [label_mebrains_to_hcp, label_yerkes, label_yerkes_to_hcp]:
		if os.path.exists(file):
			os.remove(file)

	if not os.path.exists(label_yerkes_rsl) or clobber:
		check_equal_dims(mebrains_orig, label)
		cmd=f'msmresample {mebrains_to_yerkes} {f"{outstr}_yerkes_rsl"} -project {yerkes_sphere_rsl} -labels {label}' 
		shell(cmd)
		assert os.path.exists(label_yerkes_rsl), f"File not found: {label_yerkes_rsl}"
		check_equal_dims(yerkes_sphere_rsl, label_yerkes_rsl)
	
	if not os.path.exists(label_yerkes) or clobber:
		check_equal_dims(mebrains_to_yerkes, label)
		shell(f'msmresample {yerkes_sphere_rsl} {f"{outstr}_yerkes"} -project {yerkes_sphere} -labels {label_yerkes_rsl}')
		assert os.path.exists(label_yerkes), f"File not found: {label_yerkes}"
		check_equal_dims(yerkes_sphere, label_yerkes)
	
	shell(f"wb_command -set-structure {label_yerkes} CORTEX_LEFT")
	#shell(f'wb_view {yerkes_cortex} {label_yerkes}')

	if not os.path.exists(f"{outstr}_yerkes-to-hcp.func.gii"):
		shell(f"msmresample {yerkes_to_hcp} {outstr}_yerkes-to-hcp -project {hcp_sphere} -labels {outstr}_yerkes.func.gii")
	
	#shell(f'wb_view {hcp_cortex} {label_yerkes_to_hcp}')
		
	shell(f"msmresample {mebrains_to_hcp} {outstr}_hcp -project {hcp_sphere} -labels {label}" ) 
	
	shell(f"wb_view {hcp_cortex} {outstr}_hcp.func.gii {label_yerkes_to_hcp}")


#wb_view  ../neuromaps-nhp-prep/data/yerkes/surfaces/MacaqueYerkes19.L.mid.32k_fs_LR.surf.gii\
#	temp/mebrains_to_fsaverage/yerkes_lh.MEBRAINS.mid_axis-2.func.gii.func.gii