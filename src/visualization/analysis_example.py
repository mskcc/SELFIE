import os
import numpy as np
import scipy.io
import scipy.ndimage
import nibabel as nib
from natsort import natsorted
from pathlib import Path
from src.utilities.validation import calculate_mean_and_std, print_metrics, extract_data, \
    extract_metrics, load_metrics, aggregate_metrics_and_csv # calculate_start_slope,
from src.visualization.visualize import plot_temp_enhancement_nufft_grasp_output, plot_contrast_phases, boxplots,\
    make_gif_zoom, make_gif, make_gif_forward_reverse #plot_two_directions

# ADMINISTRATION
# Folder with the Movienet results
foldername = Path(os.environ["RESULTS_DIR"])
# Folder where to store results
savefoldername = Path(os.environ["RESULTS_ANALYSIS_DIR"])
# Example slices to analyze
PURE = False # not applicable, so must be always True for example script
saves_as_niftii = True
# saves_as_npz = False
GYN = True
MOTION = False
SAVE_METRICS = True
PRINT_METRICS = False

if GYN is True: 
    # name of experiments
    runname = 'SELFIE_gyn_example' #'SELFIE_GYN_2Dt_floor' 'SELFIE_GYN_2nd --> without extra 5 patients'
    runname_Movienet = 'Movienet_gyn_example' #'Movienet_GYN_newtestdata_floor' 'Movienet_GYN_newtestdata --> without extra 5 patients'
    # folder where ROIs are stored
    maskfoldername = Path(os.environ["MASK_DIR"])
    # name for legend in plots
    exp_name = 'SELFIE'
    # number of time frames
    Tf = 60
    # Study names of test set
    filenames = ['Study1234']
    filenames_metric = filenames
    filenames_metric_labels = ['volunteer']
    scatter_alpha = 0.5
    # ROIs
    ROIs = [0, 1, 2]
    vmax = 0.35
    # coeff = 30
    tp1 = 6
    tp2 = [13,]
    tp3 = 54
    time_points = [15,]
    clip_low = 0
    clip_high = 2.2
    loc = "lower right"
    res = [4,]
    # Example slices to analyze
    ax_slice_id = [31,]
    # sag_slice_id = [105,]
    # cor_slice_id = [176, 155, 176, 163, 168, 189, 109, 201]
    rotl90 = [0, ]

if MOTION is True: 
    # name of experiments
    runname = 'SELFIE_motion_PURE'
    runname_Movienet = 'SELFIE_motion_PURE' 
    # name for legend in plots
    exp_name = 'SELFIE'
    # number of time frames
    Tf = 10
    # Study names of test set
    filenames = ['EXAMXXXX']
    # Example slices to analyze
    filenames_metric = ['EXAMXXXX']
    filenames_metric_labels = ['patient A']
    scatter_alpha = 0.5
    # Example slices to analyze
    # ROIs
    tp1 = 0
    tp2 = [5,]
    tp3 = 9
    clip_low = 0
    clip_high = 1.8
    time_points = [0,]
    res = [4/1.25,]
    ax_slice_id = [32,]
    # sag_slice_id = [216, 156, 181, 216, 216, 216]
    # cor_slice_id = [171, 171, 161, 171, 171, 171]

outputfolder = savefoldername / f'exp_{runname}'
Path.mkdir(savefoldername, exist_ok=True)
Path.mkdir(outputfolder, exist_ok=True)

if PRINT_METRICS is False:
    for file_id, filename, time_point in zip(range(len(filenames)), filenames, time_points):
        print(filename)
        path = Path(foldername / f'{runname}')
        path_Movienet = Path(foldername / f'{runname_Movienet}')

        SELFIE_files = natsorted([str(x) for x in path.glob(f'{filename}*test_output.mat')])
        Movienet_files = natsorted([str(x) for x in path_Movienet.glob(f'{filename}*test_output.mat')])

        NUFFT_array_all = np.zeros((256, 256, len(SELFIE_files), Tf))
        GRASP_array_all = np.zeros((256, 256, len(SELFIE_files), Tf))
        Movienet_array_all = np.zeros((256, 256, len(SELFIE_files), Tf))
        SELFIE_array_all = np.zeros((256, 256, len(SELFIE_files), Tf))
        
        if filename == "Study3396":
            clip_high_vol = 2.4
        else:
            clip_high_vol = clip_high
        resolution = (1, 1, res[file_id])
        slices = [ax_slice_id[file_id]]

        if saves_as_niftii == True:
            for slice_nr in range(len(SELFIE_files)):
                # Extract study name
                data = scipy.io.loadmat(SELFIE_files[slice_nr])
                data_Movienet = scipy.io.loadmat(Movienet_files[slice_nr])

                rnufft, grasp, output_Movienet, output = extract_data(data, data_Movienet, PURE)
                if GYN is True and PURE is False:
                    if rotl90[file_id] == 1:
                        rnufft = np.rot90(rnufft, -1, axes=(1, 2))
                        grasp = np.rot90(grasp, -1, axes=(1, 2))
                        output = np.rot90(output, -1, axes=(1,2))
                        output_Movienet = np.rot90(output_Movienet, -1, axes=(1, 2))

                NUFFT_array_all[:, :, slice_nr, :] = np.transpose(rnufft, (1, 2, 0))
                GRASP_array_all[:, :, slice_nr, :] = np.transpose(grasp, (1, 2, 0))
                Movienet_array_all[:, :,  slice_nr, :] = np.transpose(output_Movienet,(1, 2, 0))
                SELFIE_array_all[:, :,  slice_nr, :] = np.transpose(output, (1, 2, 0))

            affine = np.eye(4)
            np.fill_diagonal(affine[:3, :3], resolution)
            nufft_nifti_image = nib.Nifti1Image(NUFFT_array_all, affine)
            grasp_nifti_image = nib.Nifti1Image(GRASP_array_all, affine)
            movienet_nifti_image = nib.Nifti1Image(Movienet_array_all, affine)
            selfie_nifti_image = nib.Nifti1Image(SELFIE_array_all, affine)

            # Save the NIfTI image
            nib.save(nufft_nifti_image,f'{savefoldername}/exp_{runname}/{filename}_nufft.nii.gz')
            nib.save(grasp_nifti_image,f'{savefoldername}/exp_{runname}/{filename}_grasp.nii.gz')
            nib.save(movienet_nifti_image,f'{savefoldername}/exp_{runname}/{filename}_movienet.nii.gz')
            nib.save(selfie_nifti_image,f'{savefoldername}/exp_{runname}/{filename}_selfie.nii.gz')

        else:
            # load nifti
            NUFFT_array_all = nib.load(f'{savefoldername}/exp_{runname}/{filename}_nufft.nii.gz').get_fdata()
            GRASP_array_all = nib.load(f'{savefoldername}/exp_{runname}/{filename}_grasp.nii.gz').get_fdata()
            Movienet_array_all = nib.load(f'{savefoldername}/exp_{runname}/{filename}_movienet.nii.gz').get_fdata()
            SELFIE_array_all = nib.load(f'{savefoldername}/exp_{runname}/{filename}_selfie.nii.gz').get_fdata()

        # normalization with 5th and 95th percentile
        NUFFT_array_all = np.fliplr((NUFFT_array_all - np.percentile(NUFFT_array_all, 5)) / (np.percentile(NUFFT_array_all, 95) - np.percentile(NUFFT_array_all, 5)))
        GRASP_array_all = np.fliplr((GRASP_array_all - np.percentile(GRASP_array_all, 5)) / (np.percentile(GRASP_array_all, 95) - np.percentile(GRASP_array_all, 5)))
        Movienet_array_all = np.fliplr((Movienet_array_all - np.percentile(Movienet_array_all, 5)) / (np.percentile(Movienet_array_all, 95) - np.percentile(Movienet_array_all, 5)))
        SELFIE_array_all = np.fliplr((SELFIE_array_all - np.percentile(SELFIE_array_all, 5)) / (np.percentile(SELFIE_array_all, 95) - np.percentile(SELFIE_array_all, 5)))
        
        # slice wise analysis
        for slice_nr, slice_idx in enumerate(slices):
            savename = f'{filename}_slice{slice_idx}'
            percent1 = percent2 = percent3 = 100.0

            rnufft = NUFFT_array_all[:, :, slice_nr, :]
            grasp = GRASP_array_all[:, :, slice_nr, :]
            output_Movienet = Movienet_array_all[:, :, slice_nr, :]
            output = SELFIE_array_all[:, :, slice_nr, :]
            ##########################################
            if GYN is True:
                # Make GIFS
                make_gif(
                    np.clip(grasp, clip_low, clip_high),
                    np.clip(output_Movienet, clip_low, clip_high),
                    np.clip(output, clip_low, clip_high), 
                    outputfolder,
                    savename,
                    Tf,
                    percent1,
                    percent2,
                    percent3)

                orientation = 'top_bottom'
                zoom_box = 'top_right'
                size = 80
                zoom = 2.5
                scaling = 1.15
                if filename == "Study1234":
                    x1 = 390
                    y1 = 330 #230
                
                make_gif_zoom(
                    np.clip(grasp, clip_low, clip_high),
                    np.clip(output_Movienet,clip_low, clip_high),
                    np.clip(output, clip_low, clip_high),
                    outputfolder,
                    savename,
                    x1,
                    y1,
                    Tf,
                    percent1,
                    percent2,
                    percent3,
                    size=size,
                    zoom=zoom,
                    scaling=scaling,
                    orientation=orientation,
                    zoom_box=zoom_box,
                    fps=5)

                #########################################
                #  Temporal behavior analysis
                Path.mkdir(Path(outputfolder / 'ROIs'), exist_ok=True)
                masks = scipy.io.loadmat(maskfoldername / f'ROIs_{filename}_slice_{slice_idx}.mat')['masks']
                masks = np.transpose(np.fliplr(np.transpose(masks, (1, 2, 0))), (2, 0, 1))

                if GYN is True:
                    if rotl90[file_id] == 1:
                        masks = np.flip(np.flip(np.rot90(masks, -1, axes=(1, 2)), axis=2), axis=1)

                for roi_number in ROIs:

                    # calculate norm and set to 1 if the norm is 0
                    norm_nufft = np.linalg.norm(rnufft, axis=2)
                    norm_nufft = np.where(norm_nufft == 0, 1, norm_nufft)
                    norm_grasp = np.linalg.norm(grasp, axis=2)
                    norm_grasp = np.where(norm_grasp == 0, 1, norm_grasp)
                    norm_output_Movienet = np.linalg.norm(output_Movienet, axis=2)
                    norm_output_Movienet = np.where(norm_output_Movienet == 0, 1, norm_output_Movienet)
                    norm_output = np.linalg.norm(output, axis=2)
                    norm_output = np.where(norm_output == 0, 1, norm_output)

                    # normalize the data
                    rnufft_norm = rnufft / norm_nufft[:,:,None]
                    grasp_norm = grasp / norm_grasp[:,:,None]
                    output_norm_Movienet = output_Movienet / norm_output_Movienet[:,:,None]
                    output_norm = output / norm_output[:,:,None]

                    # data in ROI
                    rnufft_roi_all_norm = masks[roi_number, :, :, None] * rnufft_norm
                    grasp_roi_all_norm = masks[roi_number, :, :, None] * grasp_norm
                    output_Movienet_roi_all_norm = masks[roi_number, :, :, None] * output_norm_Movienet
                    output_roi_all_norm = masks[roi_number, :, :, None] * output_norm

                    scipy.io.savemat(
                        outputfolder / f'ROIs/ROI_{roi_number}_{filename}_slice_{slice_idx}.mat',
                        {
                            "rnufft_roi_all_norm": rnufft_roi_all_norm,
                            "grasp_roi_all_norm": grasp_roi_all_norm,
                            "output_Movienet_roi_all_norm": output_Movienet_roi_all_norm,
                            "output_roi_all_norm": output_roi_all_norm
                            }
                        )

                    mean_std_dict = calculate_mean_and_std(
                        rnufft_roi_all_norm,
                        grasp_roi_all_norm,
                        output_Movienet_roi_all_norm,
                        output_roi_all_norm,
                        masks,
                        roi_number,
                        Tf)
                
                    plot_temp_enhancement_nufft_grasp_output(Tf, mean_std_dict, outputfolder, savename, roi_number, vmax=vmax, loc=loc)

                    # start_slope = calculate_start_slope(
                    #     mean_std_dict,
                    #     Tf,
                    #     coeff=coeff,
                    #     plot_path=f'{outputfolder}/ROIs/{filename}_slice_{slice_idx}_ROI_{roi_number}_poly_fit.png')
                    # print(f'{filename}_slice{slice_idx}_ROI_{roi_number}: ', start_slope)
            if MOTION is True:
                    make_gif_forward_reverse(
                    np.clip(grasp, clip_low, clip_high),
                    np.clip(output_Movienet, clip_low, clip_high),
                    np.clip(output, clip_low, clip_high), 
                    outputfolder,
                    savename,
                    Tf,
                    percent1,
                    percent2,
                    percent3)
        # volume analysis
        if SAVE_METRICS == True: 
            extract_metrics(filename, Tf, NUFFT_array_all, GRASP_array_all, Movienet_array_all, SELFIE_array_all, outputfolder)

        plot_contrast_phases(
            np.clip(GRASP_array_all, clip_low, clip_high_vol) / (clip_high_vol - clip_low),
            np.clip(Movienet_array_all, clip_low, clip_high_vol) / (clip_high_vol - clip_low),
            np.clip(SELFIE_array_all, clip_low, clip_high_vol) / (clip_high_vol - clip_low),
            0, # actually ax_slice_id[file_id]-1, but here 0, as weonly have 1 test slice
            tp1,
            tp2[file_id],
            tp3, 
            savefoldername,
            runname,
            filename)

        # plot coronal and saggital view
        # resolution_GRASP = scipy.ndimage.zoom(GRASP_array_all[:, :, :, time_point], resolution)
        # resolution_Movienet = scipy.ndimage.zoom(Movienet_array_all[:, :, :, time_point], resolution)
        # resolution_SELFIE = scipy.ndimage.zoom(SELFIE_array_all[:, :, :, time_point], resolution)

        # if saves_as_npz == True:
        #     np.savez(f'{savefoldername}/exp_{runname}/{filename}_3D.npz', GRASP=resolution_GRASP, Movienet=resolution_Movienet, SELFIE=resolution_SELFIE) 

        # plot_two_directions(
        #     np.clip(resolution_GRASP, clip_low, clip_high) / (clip_high - clip_low),
        #     np.clip(resolution_Movienet, clip_low, clip_high) / (clip_high - clip_low),
        #     np.clip(resolution_SELFIE, clip_low, clip_high) / (clip_high - clip_low),
        #     ax_slice_id[file_id],
        #     sag_slice_id[file_id],
        #     cor_slice_id[file_id],
        #     resolution, 
        #     LIVER,
        #     savefoldername,
        #     runname,
        #     filename)
    
if PRINT_METRICS == True: 
    aggregate_metrics_and_csv(filenames, ['BEW', 'MLC_in', 'MTV'], Tf, outputfolder)

    metric_list = []
    for id, filename in enumerate(filenames):
        metrics = load_metrics(outputfolder, filename)
        metric_list.append(metrics)


    boxplots(metric_list, ['BEW', 'MLC_in', 'MTV'], outputfolder, filenames_metric_labels, filenames_metric, scatter_alpha)
    print_metrics(filenames, outputfolder)