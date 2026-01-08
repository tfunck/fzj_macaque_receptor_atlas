import os
import subprocess
import nibabel as nib

base_dir = "/home/thomas-funck/projects/neuromaps-nhp-prep/output/"
metric_dir = os.path.join(base_dir, "metrics")

mebrains_to_yerkes = 'mebrains_to_yerkes19/0//lh.n-32492_lh.MEBRAINS_0.5mm_1.0_metrics_y_z_x_warped_sphere.reg.surf.gii'
mebrains_to_hcp = "/media/windows/projects/fzj_macaque_receptor_atlas/outputs/0.25mm/surfaces/mebrains2human_surf_tfm.surf.gii"
mebrains_orig = "/home/thomas-funck/projects/fzj_macaque_receptor_atlas/data/surfaces/lh.MEBRAINS_0.5mm_sphere.surf.gii"
mebrains_32k = '/media/windows/projects/fzj_macaque_receptor_atlas/mebrains_to_yerkes19/1/lh.n-32492_lh.MEBRAINS_0.5mm_1.0_metrics_curv_warped_sphere.reg.surf.gii'

yerkes_sphere = "/media/windows/projects/fzj_macaque_receptor_atlas/data/surfaces/MacaqueYerkes19.L.sphere.32k_fs_LR.surf.gii"
yerkes_to_hcp = "/home/thomas-funck/projects/neuromaps-nhp-prep/data/macaque-to-human/L.macaque-to-human.sphere.reg.32k_fs_LR.surf.gii"
yerkes_cortex = "../neuromaps-nhp-prep/data/yerkes/surfaces/MacaqueYerkes19.L.mid.32k_fs_LR.surf.gii"

hcp_sphere = "S1200.L.sphere.32k_fs_LR.surf.gii"
hcp_cortex = 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'

output_dir = "/home/thomas-funck/projects/fzj_macaque_receptor_atlas/temp/mebrains_to_hcp/"
os.makedirs(output_dir, exist_ok=True)

labels = "outputs/0.25mm/0.06-0.33/macaque_oxot_smoothed.func.gii"

subprocess.run([
	"wb_command", "-surface-sphere-project-unproject",
	mebrains_to_yerkes,
	yerkes_sphere,
	yerkes_to_hcp,
	mebrains_to_hcp
], shell=True, executable='/bin/bash' )

def dims(fn):
	print(fn)
	print(nib.load(fn).darrays[0].data.shape)

for label in labels.split():
	outstr = os.path.basename(label).replace('.func.gii', '')

	label_yerkes = os.path.join(output_dir, f"{outstr}_yerkes.func.gii")
	label_yerkes_to_hcp = os.path.join(output_dir, f"{outstr}_yerkes-to-hcp.func.gii")
	label_mebrains_to_hcp = os.path.join(output_dir, f"{outstr}_mebrains-to-hcp.func.gii")

	for file in [label_mebrains_to_hcp, label_yerkes, label_yerkes_to_hcp]:
		if os.path.exists(file):
			os.remove(file)

	if not os.path.exists(os.path.join(output_dir, f"{outstr}_mebrains_rsl.func.gii")):
		print("MEBRAINS 100k -> MEBRAIN 32k")
		subprocess.run([
			"msmresample", mebrains_orig,
			os.path.join(output_dir, f"{outstr}_mebrains_rsl"),
			"-project", mebrains_32k,
			"-labels", label
		] shell=True, executable='/bin/bash')

		dims(mebrains_orig) 
		dims( label)
		dims( os.path.join(output_dir, f"{outstr}_mebrains_rsl.func.gii"))

	if not os.path.exists(os.path.join(output_dir, f"{outstr}_yerkes.func.gii")):
		print("MEBRAINS 32k -> Yerkes19 32k")
		subprocess.run([
			"msmresample", mebrains_to_yerkes,
			os.path.join(output_dir, f"{outstr}_yerkes"),
			"-project", yerkes_sphere,
			"-labels", os.path.join(output_dir, f"{outstr}_mebrains_rsl.func.gii")
		])

		dims( mebrains_to_yerkes)
		dims( yerkes_sphere)
		dims( os.path.join(output_dir, f"{outstr}_mebrains_rsl.func.gii"))

	subprocess.run(["wb_view", yerkes_cortex, label_yerkes], shell=True, executable='/bin/bash')
	break

	if not os.path.exists(os.path.join(output_dir, f"{outstr}_yerkes-to-hcp.func.gii")):
		print("Yerkes 32k -> HCP 32k")
		subprocess.run([
			"msmresample", yerkes_to_hcp,
			os.path.join(output_dir, f"{outstr}_yerkes-to-hcp"),
			"-project", hcp_sphere,
			"-labels", os.path.join(output_dir, f"{outstr}_yerkes.func.gii")
		], shell=True, executable='/bin/bash')

		print("Yerkes to HCP dimensions")
		dims( yerkes_to_hcp)
		print("Yerkes dimensions")
		dims( label_yerkes)
		print("HCP dimensions")
		dims( hcp_sphere)

	subprocess.run([
		"msmresample", mebrains_to_hcp,
		os.path.join(output_dir, f"{outstr}_hcp"),
		"-project", hcp_sphere,
		"-labels", label
	],shell=True, executable='/bin/bash')

	subprocess.run([
		"wb_view", hcp_cortex,
		os.path.join(output_dir, f"{outstr}_hcp.func.gii"),
		label_yerkes_to_hcp
	],shell=True, executable='/bin/bash')


#wb_view  ../neuromaps-nhp-prep/data/yerkes/surfaces/MacaqueYerkes19.L.mid.32k_fs_LR.surf.gii\
#	temp/mebrains_to_fsaverage/yerkes_lh.MEBRAINS.mid_axis-2.func.gii.func.gii
