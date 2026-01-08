

def analyze_surface(receptor_surfaces, wm_surf_filename, gm_surf_filanme, output_dir, clobber=False):


    entropy_dir = args.output_dir +'/entropy'
    ratio_dir = f'{args.output_dir}/ratios/'
    corr_dir = f'{args.output_dir}/correlations/'

    ########################
    ### Entropy Analysis ###
    ########################
    if run_entropy :
        entropy_file, total_entropy_filenames = entropy_analysis(
            mask_rsl_file, 
            medial_wall_mask, 
            receptor_surfaces, 
            wm_surf_filename,
            gm_surf_filename, 
            entropy_dir, 
            nlayers=n_layers,
            clobber=clobber
            )
        #volume_feature_dict['entropy']=entropy_file
        #volume_feature_dict['mean'] = [mean_file]
        #volume_feature_dict['std'] = [std_file]
    exit(0)
    #########################
    ### Gradient Analysis ###
    #########################
    if run_gradient :
        surf_pca(receptor_surfaces, medial_wall_mask, args.wm_surf_filename, args.sphere_surf_filename, grad_dir, n=10000, clobber=clobber)
        #gradient_volumes = volumetric_gradient_analysis(mask_rsl_file, receptor_volumes, grad_dir, approach='pca', n=args.n, clobber=clobber)
        #volume_feature_dict['gradient'] = [ gradient_volumes ]

    #################################################
    ### Calculate ratios between receptor volumes ###
    #################################################
    if run_ratio :
        ratio_dict, [inh_list, exh_list, mod_list] = ratio_analysis(receptor_volumes, mask_rsl_file, ratio_dir, clobber=clobber )
        #ratio_volumes = [fn for fn,lab in ratio_dict.items() if lab in ['Ex', 'Inh', 'Mod', 'GABAa/GABAb',  'Ex/Inh', '(Inh+Ex)/Mod']]
        ratio_volumes = [fn for fn,lab in ratio_dict.items() if lab in [  'Ex/Inh', '(Inh+Ex)/Mod']]
        volume_feature_dict['ratio'] = ratio_volumes 

    ##################################
    ### Calculate surface aligment ###
    ##################################
    ### MEBRAINS -> Yerkes
    yerkes_surface_feature_dict, spheres_dict = preprocess_surface(
        args.yerkes_mid_surf_filename,
        args.yerkes_sphere_surf_filename,
        yerkes_medial_wall_mask_filename,
        args.wm_surf_filename, 
        args.mid_surf_filename,
        args.gm_surf_filename,
        args.sphere_surf_filename,
        mebrains_medial_wall_mask_filename,
        volume_feature_dict,
        align_dir,
        clobber=clobber
        )
    ### MEBRAINS -> D99 
    if False :
        d99_volume_feature_dict = {'cell_density': glob.glob("data/mapped_volumes/image_by_cell_default_res/*nii.gz")}
        yerkes_cell_surface_feature_dict, spheres_dict = preprocess_surface(
            args.yerkes_mid_surf_filename,
            args.yerkes_sphere_surf_filename,
            yerkes_medial_wall_mask_filename,
            d99_wm_surf_filename,
            d99_mid_surf_filename,
            d99_gm_surf_filename, 
            d99_sphere_surf_filename,
            d99_medial_wall_mask_filename,
            d99_volume_feature_dict,
            align_dir+'/d99/',
            clobber=clobber
            )
    
    ##########################
    ### Cell Type Analysis ###
    ##########################
    if run_cell_analysis :
        cell_type_analysis(
                yerkes_surface_feature_dict['receptor'],
                yerkes_surface_feature_dict['medial_wall_mask'][0],
                args.yerkes_wm_surf_filename, 
                args.yerkes_sphere_surf_filename,
                args.output_dir+'/cell_type', 
                clobber=clobber 
                )
        
    ##############################
    ### Developmental Analysis ###
    ##############################
    if run_developmental_analysis :
        pass
    
    ############################
    ### Extract atlas values ###
    ############################
    if run_atlas_analysis :
        for descriptor, yerkes_feature_surfaces in yerkes_surface_feature_dict.items():
            apply_surface_atlas(yerkes_feature_surfaces, yerkes_atlas_filename, args.output_dir, descriptor)

    exit(0)
 
    #yerkes_atlas_filename = resample_label(
    #    args.yerkes_atlas_filename,
    #    args.yerkes_10k_sphere_surf_filename,
    #    args.yerkes_sphere_surf_filename,
    #    align_dir,
    #    clobber=True
    #)
        
    #Volumetric Alignment
    #receptor_volumes = align(args.yerkes_template_filename, args.mebrains_filename, mask_rsl_file, receptor_volumes, align_dir)
    #receptor_surfaces = project_to_surface( receptor_volumes, args.wm_surf_filename, args.gm_surf_filename, profiles_dir, agg_func=np.mean, clobber=False )   
    #vif_analysis(receptor_volumes, mask_rsl_file, corr_dir)
    #plot_pairwise_correlation(receptor_volumes, mask_rsl_file, corr_dir)
    #complexity_volumes = align(args.yerkes_template_filename, args.mebrains_filename, mask_rsl_file, [entropy_file, std_file], align_dir)

    #t1t2_analysis(mask_rsl_file, hist_volumes, t1t2_filename, args.output_dir)
    ### Resize MEBRAINS T1/T2 to receptor volume
    #t1t2_rsl_filename = resize_mask_to_receptor_volume( t1t2_filename, receptor_volumes[0], args.output_dir, order=3)

    # Plot entropy on surface
    project_and_plot_surf(
        [entropy_file], 
        args.wm_surf_filename, 
        args.gm_surf_filename, 
        entropy_dir,
        medial_wall_mask = medial_wall_mask,
        threshold=(0.30, 0.98),
        #cmap='nipy_spectral',
        clobber=True
        )
    exit(0) 
    project_and_plot_surf(
        [std_file], 
        args.wm_surf_filename, 
        args.gm_surf_filename, 
        entropy_dir,
        medial_wall_mask = medial_wall_mask,
        threshold=(0.3,0.7),
        #cmap='nipy_spectral',
        clobber=True
        )
    #receptor_volumes = [ args.label_file ]
    ### Calculate surface differential gradients
    surface_analysis(receptor_volumes, args.wm_surf_filename, args.gm_surf_filename, profiles_dir, args.output_dir)
                     
    #apply_surface_atlas(yerkes_receptor_surfaces, args.yerkes_atlas_filename, args.output_dir, 'receptor')
    #comparison_volumes = [gradient_volumes, [ str(i) for i, fn in enumerate(gradient_volumes)] ]
    #comparison_volumes[0] += ratios[0]
    #comparison_volumes[1] += ratios[1]
    #comparison_volumes[0].append( entropy)
    #comparison_volumes[1].append('entropy')
    t1t2_tp = (t1t2_rsl_filename,  't1t2')
    entropy_tp = (entropy, 'entropy')
    comparison_volumes = [[t1t2_tp, entropy_tp],]
    
    [entropy, t1t2_rsl_filename] = project_to_surface(
                                        [entropy, t1t2_rsl_filename], 
                                        args.wm_surf_filename, 
                                        args.gm_surf_filename, 
                                        profiles_dir
                                    )
        
    x = np.load(entropy)[:,2:-2]
    y = np.load(t1t2_rsl_filename)[:,2:-2]
    x = np.mean(x, axis=1)
    y = np.mean(y, axis=1)
    print(pearsonr(x,y) )
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x.reshape(-1,1), y)
    # get model residuals
    residuals = np.abs(y - model.predict(x.reshape(-1,1)))

    plt.scatter(x,y,alpha=0.02)
    plt.savefig(f'{corr_dir}/entropy_vs_t1t2_surf.png')

    #coords_l, faces_l = mesh_utils.load_mesh_ext(args.infl_surf_filename)
    coords_l, faces_l = mesh_utils.load_mesh_ext(args.gm_surf_filename)

    plot_surf( coords_l, 
                faces_l, 
                x, 
                rotate=[90, 270], 
                filename=f'{corr_dir}/entropy_surf.png',
                vmax = np.max(x), #np.percentile(y, 99.5), 
                vmin = np.min(x), # np.percentile(y, 0.5), 
                pvals=np.ones_like(x),
                cmap='nipy_spectral',
                cmap_label='Entropy'
                )

    plot_surf( coords_l, 
                faces_l, 
                y, 
                rotate=[90, 270], 
                filename=f'{corr_dir}/t1t2_surf.png',
                #vmax = np.percentile(x, 98), 
                #vmin = np.percentile(x, 2), 
                pvals=np.ones_like(x),
                cmap='nipy_spectral',
                cmap_label='T1/T2'
                )


    plot_surf( coords_l, 
                faces_l, 
                residuals, 
                rotate=[90, 270], 
                filename=f'{corr_dir}/entropy_vs_t1t2_surf_residuals.png',
                #vmax = np.percentile(residuals, 98), 
                #vmin = np.percentile(residuals, 2), 
                pvals=np.ones_like(residuals),
                cmap='nipy_spectral',
                cmap_label='|Residuals|'
                )


    

def project_and_plot_surf(
        volumes:list,
        wm_surf_filename:str,
        gm_surf_filename:str,
        output_dir:str,
        n:int = 10,
        sigma:float = 0,
        medial_wall_mask=None,
        zscore:float = True,
        agg_func=None,
        invert=0,
        cmap='RdBu_r',
        threshold:float = (.02, .98),
        clobber:bool = False,
        ):

    surface_data_list = project_to_surface(
        volumes,
        wm_surf_filename, 
        gm_surf_filename, 
        output_dir,
        n = n,
        sigma = sigma,
        zscore = zscore,
        invert=invert,
        agg_func=agg_func,
        clobber = clobber
        )


    return  surface_data_list

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
    
def plot_pairwise_correlation(comparison_volumes, mask_vol, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    mask_vol = nib.load(mask_vol).get_fdata()

    n = len(comparison_volumes)
    corr_matrix = np.eye(n)

    #   0 1 2 3
    # 0 1
    # 1   1
    # 2     1
    # 3       1

    for i, x_filename in enumerate(comparison_volumes[0:-1]):
        x = nib.load(x_filename).get_fdata()

        for j, y_filename in enumerate(comparison_volumes[i+1:]):
            y = nib.load(y_filename).get_fdata()
            k=j+i+1
            print(i,k)

            xmin = np.min(x) 
            ymin = np.min(y) 

            idx = (mask_vol > 0) & (x > xmin) & (y > ymin)
            
            x0 = x[idx]
            y0 = y[idx]

            r, p = spearmanr(x0, y0)
            #r, p = pearsonr(x, y)
            print(f'{x_filename} vs {y_filename} : r={r}, p={p}')
            corr_matrix[i,k] = r
            corr_matrix[k,i] = r

    plt.cla(); plt.clf(); plt.close()
    plt.imshow(corr_matrix, cmap='viridis')
    plt.colorbar()
    plt.savefig(f'{output_dir}/correlation_matrix.png')

    np.save(f'{output_dir}/correlation_matrix.npy', corr_matrix)




def vif_analysis(comparison_volumes, mask_filenmae, output_dir):
    n=len(comparison_volumes)

    mask_vol= nib.load(mask_filenmae).get_fdata()

    idx = np.where(mask_vol > 0)
    features= np.zeros((idx[0].size, n))
    df = pd.DataFrame({'volume': [os.path.basename(fn).replace('.nii.gz','') for fn in comparison_volumes], 'vif':np.zeros(n) })
    for i, filename in enumerate(comparison_volumes):
        x = nib.load(filename).get_fdata()
        features[:,i] = x[idx]

    print(np.mean(features, axis=0))

    df['vif'] = [variance_inflation_factor(features, i) for i in range(n)]
    print(df)
    df.to_csv(f'{output_dir}/vif.csv', index=False)

def align(fixed_filename, moving_filename, moving_mask_filename, files_to_transform, output_dir):
    import ants

    os.makedirs(output_dir, exist_ok=True)

    fwd_filename = f'{output_dir}/forward_transform.h5'
    inv_filename = f'{output_dir}/inverse_transform.h5'

    output_files = [ f'{output_dir}/{os.path.basename(filename)}' for filename in files_to_transform ]

    if  not os.path.exists(fwd_filename) or not os.path.exists(inv_filename):
        print('Registering')
        fixed = ants.image_read(fixed_filename)
        moving = ants.image_read(moving_filename)
        moving_mask = ants.image_read(moving_mask_filename)
        reg0 = ants.registration(
            fixed=fixed, 
            moving=moving, 
            type_of_transform='SyN',
            syn_metric = 'GC',
            reg_iterations=(400, 200, 100),
            mask=moving_mask, 
            write_composite_transform=True,
            verbose=True, 
            output=f'{output_dir}/aligned.nii.gz'
            )
        
        reg = ants.registration(
            fixed=fixed, 
            moving=moving, 
            type_of_transform='SyNOnly',
            syn_metric = 'CC',
            reg_iterations=(20, 10),
            mask=moving_mask, 
            transform_list = reg0['fwdtransforms'],
            write_composite_transform=True,
            verbose=True, 
            output=f'{output_dir}/aligned.nii.gz'
            )
        print(reg.items())
        #  transforms
        ants.write_transform( ants.read_transform(reg['fwdtransforms']), fwd_filename)
        ants.write_transform( ants.read_transform(reg['invtransforms']), inv_filename)
        reg['warpedmovout'].to_filename(f'{output_dir}/aligned.nii.gz')

    for output_filename, filename in zip(output_files, files_to_transform):
        if not os.path.exists(output_filename):
            fixed = ants.image_read(fixed_filename)
            print('Applying transform', output_filename)
            img = ants.image_read(filename)
            img = ants.apply_transforms(fixed=fixed, moving=img, transformlist=fwd_filename)
            img.to_filename(output_filename)
            output_files.append(output_filename)
    
    return output_files

def apply_surface_atlas(surf_files, atlas_file, output_dir, descriptor, use_col=True):
    # load surface atlas
    atlas = nib.load(atlas_file).darrays[0].data

    df = pd.DataFrame({})

    n=atlas.shape[0]

    for surf_file in surf_files:
        print("Surf file:", surf_file)
        values = load_gifti(surf_file).reshape(-1,)
        print(atlas.shape, values.shape)
        
        filename= os.path.basename(surf_file).replace('.nii.gz','').replace('macaque_','').replace('.npy','')
        row = pd.DataFrame({'receptor':[filename]*n, 'label':atlas, 'density':values})
        df = pd.concat([df,row])
    
    # remove 0 labels
    df = df.loc[df['label'] > 0]

    #atlas_coding = { 1: 'Visual', 2: 'Somatomotor', 3: 'Dorsal Attention', 4: 'Ventral Attention', 5:'Limbic', 6: 'Frontoparietal', 7: 'DMN'}
    atlas_coding = { 1:'DMN', 2:'Somatomotor', 3:'Auditory', 4:'Limbic', 5:'DoralAtt', 6:'Visual', 7:'Insular-opercular'}
    x_coding = { 6:1, 3:2, 2:3,  4:4, 7:5, 5:6, 1:7 }
    df['x'] = df['label'].map(x_coding)
    df['atlas'] = df['label'].map(atlas_coding)
    print(df);
    df.to_csv(f'{output_dir}/atlas_{descriptor}.csv', index=False) 
    print(f'{output_dir}/atlas_{descriptor}.csv')
    #reindex 
    df = df.reset_index(drop=True)

    # normalize 'density' values based on 'receptor' category
    #df['density'] = (df['density'] - df.groupby('receptor')['density'].transform('mean')) / df.groupby('receptor')['density'].transform('std') 
    #df['density'] = df.groupby('receptor')['density'].transform(lambda x: (x - x.mean()) / x.std()) 

    plt.figure(figsize=(10,10))
    sns.color_palette("Set2")
    print(df['x'].unique())
    g = sns.boxplot(data=df, x="x", y="density", hue="receptor" ) 
    g.set_xticks(range(1,8))
    #g.set_xticklabels(['Visual', 'Somatomotor', 'Dor. Att.', 'Ven. Att.', 'Limbic', 'Frontoparietal', 'DMN'])
    g.set_xticklabels(['Visual', 'Auditory', 'Somatomotor', 'Limbic', 'Insular-opercular',  'DMN', 'DorsalAtt'])
    plt.savefig(f'{output_dir}/atlas_{descriptor}.png')
    plt.clf(); plt.cla()

