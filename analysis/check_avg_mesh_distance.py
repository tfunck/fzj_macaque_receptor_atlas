import os
import numpy as np

from brainbuilder.utils.mesh_io import load_mesh_ext
from brainbuilder.utils.mesh_utils import get_edges_from_faces


def check_avg_mesh_distance(mesh_filename):

    coords, faces = load_mesh_ext(mesh_filename)
    
    edges = get_edges_from_faces(faces)

    avg_distance = np.mean([np.linalg.norm(coords[edge[0]] - coords[edge[1]]) for edge in edges])

    print(os.path.basename(mesh_filename),':', avg_distance)


if __name__ == "__main__":  

    files = [ 
        'data/surfaces/MacaqueYerkes19.R.midthickness.1k_fs_LR.surf.gii', 
        'data/surfaces/MacaqueYerkes19.R.midthickness.2.5k_fs_LR.surf.gii', 
        'data/surfaces/MacaqueYerkes19.R.midthickness.5k_fs_LR.surf.gii', 
        'data/surfaces/MacaqueYerkes19.R.midthickness.10k_fs_LR.surf.gii', 
        'data/surfaces/MacaqueYerkes19.R.midthickness.32k_fs_LR.surf.gii'
    ]

    for file in files:
        check_avg_mesh_distance(file)   