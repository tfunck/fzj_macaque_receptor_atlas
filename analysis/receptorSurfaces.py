import entropy as entropy
import pca as pca
import ratios as ratios

import human_space as human_space


class ReceptorSurfaces():

    def __init__(self, 
                 receptor_surfaces:str, 
                 cortical_surface:str, 
                 sphere_surface:str,
                 output_dir:str, 
                 label:str=None,
                 human_sphere:str=None,
                 macaque_to_human_tfm_sphere:str=None,
                 clobber:bool=False):

        self.receptor_surfaces = receptor_surfaces
        self.cortical_surface = cortical_surface
        self.sphere_surface = sphere_surface
        self.output_dir = output_dir
        self.label=label
        self.human_sphere = human_sphere
        self.macaque_to_human_tfm_sphere = macaque_to_human_tfm_sphere
        self.clobber = clobber

        self.receptor_surfaces_tfm = []

        #Analyses 
        self.entropy_analysis = entropy.entropy_analysis 
        self.pca = pca.surf_pca
        self.ratios = ratios.ratio_analysis

        #Human Space Analysis
        self.transform_macaque_to_human = human_space.transform_macaque_to_human
        self.decode = human_space.decode

    def run(self):
        self.entropy_analysis(self, nbins=32, clobber=self.clobber)
        self.pca(self, clobber=self.clobber)
        self.ratios(self, clobber=self.clobber)

        if self.macaque_to_human_tfm_sphere is not None and self.human_sphere is not None:
            self.transform_macaque_to_human(self, clobber=self.clobber)
            #self.decode(self, clobber=self.clobber)