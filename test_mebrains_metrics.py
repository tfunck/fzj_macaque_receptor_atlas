
import os
from surfalign.metrics import extract_metrics
import numpy as np



def test(fn, output_dir):
    params = zip(np.linspace(1, 25, 5), np.linspace(1, 25, 5))
    for n_sulc, n_curv in params:
        metrics_dir = output_dir+f"/{n_curv}_{n_sulc}/"

        os.makedirs(metrics_dir, exist_ok=True)

        metric_filename_list = extract_metrics(
            fn, 
            metrics_dir, 
            metric_list_heir = [['curv','sulc']], 
            params=dict(n_sulc=n_sulc, n_curv=n_curv), 
            title=f'{n_curv}_{n_sulc}',
            clobber=False
            )

mebrains_lh = 'data/surfaces/lh.MEBRAINS_0.5mm_1.0.surf.gii' 
mebrains_32k_lh = "mebrains_to_yerkes19/n-32492_lh.MEBRAINS_0.5mm_1.0.surf.gii" 

#test(mebrains_lh, "test_metrics/100k/")
test(mebrains_32k_lh, "test_metrics/32k/")
