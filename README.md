# FZJ Macaque Receptor Atlas

3D reconstruction of 2D receptor autoradiographs for 4 macaque hemispheres (3 subjects) using the BrainBuilder pipeline.

Receptors: GABA_A, etc.

# Instructions

```
# Preprocess raw macaque data for recosntruction
python3 reconstruction/prepare_macaque.py
python3 reconstruction/reconstruct_macaque.py

# Build atlas from 4 hemispheres
python3.9 reconstruction/combine_volumes.py

# Analysis of receptor atlases





