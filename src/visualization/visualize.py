import imageio
import scipy
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line


def plot_temp_enhancement_nufft_grasp_output(Tf, mean_std_dict, outputfolder, savename, roi_number, vmax=0.19, loc='lower right'):
    """
    Temporal enhancement curve plots. 

    :param Tf: number of time frames
    :param mean_std_dict:
    :param outputfolder: foldername where to save plots
    :param savename: name of plot
    :param roi_number: roi mask index
    :param vmax: maximum for y-axis
    :param loc: location of legend

    :return: None, but plot is saved accordingly.
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('classic')
    for name, name_plus, name_minus, color, label in zip(
            ['rnufft_mean', 'grasp_mean', 'output_Movienet_mean', 'output_mean'],
            ['rnufft_plus', 'grasp_plus', 'output_Movienet_plus', 'output_plus'],
            ['rnufft_minus', 'grasp_minus', 'output_Movienet_minus', 'output_minus'],
            ['#8F306D', '#C63810', '#308F82', '#8C9C19'],
            ['NUFFT', 'GRASP', 'DCE-Movienet', 'SELFIE']
            ):
        if name == 'rnufft_mean':
            plt.plot(
                range(Tf),
                mean_std_dict[name],
                '-',
                linewidth=3,
                markersize=2,
                color=color,
                label=label,
                alpha=1)
            plt.fill_between(
                range(Tf),
                mean_std_dict[name_plus], mean_std_dict[name_minus],
                color=color,
                alpha=0.1)
        else:
            # find least squares factor (offset) to scale mean curve with in order to match rnufft_mean curve best
            factor = np.sum(np.array(mean_std_dict[name]) * np.array(mean_std_dict['rnufft_mean'])) / np.sum(
                np.array(mean_std_dict[name]) * np.array(mean_std_dict[name]))

            plt.plot(
                range(Tf),
                factor * np.array(mean_std_dict[name]),
                '-',
                linewidth=3,
                markersize=2,
                color=color,
                label=label,
                alpha=1)

            plt.fill_between(
                range(Tf),
                factor * np.array(mean_std_dict[name_plus]), factor * np.array(mean_std_dict[name_minus]),
                color=color,
                alpha=0.1)

    # plt.legend(loc=loc, fontsize=20)
    plt.ylim((0.02, vmax))
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2f}'.format(x) for x in current_values])
    # plt.gcf().set_facecolor((0.9, 0.9, 0.9, 0.))  # Set bg color to light gray
    # plt.gcf().set_edgecolor("white")
    plt.gca().tick_params(labelsize=25, color='1', labelcolor='1')
    plt.savefig(outputfolder / f'{savename}_mean_ROI_{roi_number}.png', transparent=True)
    plt.close()


def boxplots(metrics, metric_names, outputfolder, names, datasets, scatter_alpha=0.01):
    """
    Creation of boxplots that compares metric results of test datasets between NUFFT, grasp, Movienet, and SELFIE. 
    For each metric in metric_names a boxplot is created, where on the x-axis are the studies (datasets) and the y-axis 
    shows the metric values. For each study, the dots encode one metric value per slice and tf and additionally, 
    descriptive statistics are shown as a boxplot.

    :param metrics: list of metrics_dicts including the metric results for each study
    :param metric_names: list of metric_names to analyze
    :param outputfolder: folder where to save plots
    :param names: list of label names for x-axis
    :param datasets: list of filenames/studynames
    :param scatter_alpha: alpha value for scatterplot

    :return: None, but the plots are saved accordingly. 
    """
    for metric_name in metric_names:
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.subplots_adjust(left=0.07, right=0.98)

        np.random.seed(1234)
        centers = []
        
        # plt.style.use('classic')
        for count, metric, dataset in zip(range(len(names)), metrics, datasets):
            # Example data for two subjects and four methods
            if count == 0:
                timeframes = len(metric[metric_name][dataset]['NUFFT'][list(metric[metric_name][dataset]['NUFFT'].keys())[0]])
            NUFFT_values_1 = np.zeros((len(metric[metric_name][dataset]['NUFFT'].keys()), timeframes))
            Movienet_values_1 = np.zeros((len(metric[metric_name][dataset]['NUFFT'].keys()), timeframes))
            SELFIE_values_1 = np.zeros((len(metric[metric_name][dataset]['NUFFT'].keys()), timeframes))
            GRASP_values_1 = np.zeros((len(metric[metric_name][dataset]['NUFFT'].keys()), timeframes))

            for idx, slice in enumerate(list(metric[metric_name][dataset]['NUFFT'].keys())):
                NUFFT_values_1[idx] = metric[metric_name][dataset]['NUFFT'][slice]
                Movienet_values_1[idx] = metric[metric_name][dataset]['Movienet'][slice]
                SELFIE_values_1[idx] = metric[metric_name][dataset]['SELFIE'][slice]
                GRASP_values_1[idx] = metric[metric_name][dataset]['grasp'][slice]

 
            if  metric_name == "MLC" or metric_name == "MLC_in" or metric_name == "BEW" or metric_name == "MTV":
                colors = ['#8F306D', '#C63810', '#308F82', '#8C9C19']

                # Combine data for calculating stats
                all_values_1 = [NUFFT_values_1.flatten(), GRASP_values_1.flatten(), Movienet_values_1.flatten(), SELFIE_values_1.flatten()]
                range1 = [i*2 + count*8 + count * 1 for i in range(4)]
                centers.append(int(np.median(range1)))

            # Calculate statistics (mean, median, quantiles) for each method
            means_1 = np.mean(all_values_1, axis=1)

            x = [[i for j in range(len(all_values_1[0]))] for i in range1]
            x_rand = [list(np.asarray(x[i]) -0.9 + np.random.normal(scale=0.15, size=len(all_values_1[0]))) for i in range(len(x))]
            ax.scatter(
                x_rand,
                all_values_1,
                zorder=1,
                s=15,
                alpha=scatter_alpha,
                lw=0,
                c=[col for col in colors for j in range(len(all_values_1[0]))])

            # Boxplot for quantiles
            bp = ax.boxplot(
                all_values_1,
                positions=range1,
                widths=0.8,
                patch_artist=True,
                boxprops=dict(facecolor='lightgrey', color='grey'),
                medianprops=dict(color='black'),
                showmeans=False,
                showfliers=False)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(1)

            # Plot means as a separate line (for visual clarity)
            ax.scatter(range1, means_1, color='k', marker='o', alpha=0.8, zorder=5, s=3)

        # Labeling
        ax.set_xticks(centers)
        ax.set_xlim(-2, centers[-1]+5)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("[a.u.]",  fontsize=10)
        if metric_name == "MLC_in":
            ax.set_title(f'MLC', fontsize=20)
        else:
            ax.set_title(f'{metric_name}', fontsize=20)
        ax.set_axisbelow(True) 
        ax.grid(True, linestyle='--')
        plt.savefig(f'{outputfolder}/{metric_name}.pdf', dpi=600)
        plt.close()


def plot_contrast_phases(GRASP_array_all, Movienet_array_all, SELFIE_array_all, axial_slice, tp1, tp2, tp3, savefoldername,
                         runname, filename):
    """
    Plots GRASP, Movienet, and SELFIE reconstructions side-by-side for three time points/contrasts. 

    :param GRASP_array_all: normalized array of GRASP reconstruction
    :param Movienet_array_all: normalized array of Movienet reconstruction
    :param SELFIE_array_all: normalized array of SELFIE reconstruction
    :param axial_slice: axial slice index
    :param tp1: time point 1
    :param tp2: time point 2
    :param tp3: time point 3
    :param savefoldername: parent foldername
    :param runname: foldername in parent foldername where to save plot
    :param filename: name of plot

    :return: None, but figure is saved accordinly. 

    """
    plt.style.use('dark_background')
    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(5.2, 5)
    )

    plt.subplots_adjust(bottom=0.01, left= 0.01, right=0.9, top=0.99, wspace=0.01, hspace=0.01)

    axs[0, 0].imshow(
        GRASP_array_all[:, :, axial_slice, tp1], cmap='gray', aspect='equal', vmin=0, vmax=1)
    axs[0, 0].axis('off') 
    axs[0, 1].imshow(
        Movienet_array_all[:, :, axial_slice, tp1], cmap='gray', aspect='equal', vmin=0, vmax=1)
    axs[0, 1].axis('off') 
    axs[0, 2].imshow(
        SELFIE_array_all[:, :, axial_slice, tp1], cmap='gray', vmin=0, vmax=1)
    axs[0, 2].axis('off')
    axs[1, 0].imshow(
        GRASP_array_all[:, :, axial_slice, tp2], cmap='gray', aspect='equal', vmin=0, vmax=1)
    axs[1, 0].axis('off') 
    axs[1, 1].imshow(
        Movienet_array_all[:, :, axial_slice, tp2], cmap='gray', aspect='equal', vmin=0, vmax=1)
    axs[1, 1].axis('off') 
    axs[1, 2].imshow(
        SELFIE_array_all[:, :, axial_slice, tp2], cmap='gray', vmin=0, vmax=1)
    axs[1, 2].axis('off') 
    axs[2, 0].imshow(
        GRASP_array_all[:, :, axial_slice, tp3], cmap='gray', aspect='equal', vmin=0, vmax=1)
    axs[2, 0].axis('off') 
    axs[2, 1].imshow(
        Movienet_array_all[:, :, axial_slice, tp3], cmap='gray', aspect='equal', vmin=0, vmax=1)
    axs[2, 1].axis('off') 
    im = axs[2, 2].imshow(
        SELFIE_array_all[:, :, axial_slice, tp3], cmap='gray', vmin=0, vmax=1)
    axs[2, 2].axis('off') 


    cbar_ax = fig.add_axes([0.91, 0.03, 0.01, 0.95])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=7) 

    # plt.tight_layout
    plt.savefig(f'{savefoldername}/exp_{runname}/{filename}_pre_art_post.png', dpi=600)
    plt.close()


def plot_two_directions(
        rescaled_GRASP, rescaled_Movienet, rescaled_SELFIE, axial_slice, sag_slice, cor_slice, resolution, LIVER,
        savefoldername, runname, filename):
    """
    Plots results of GRSAP, Movienet, and SELFIE in axial and sagittal orientation for DCE GYN experiment, and 
    in axial and coronal orientation for DCE liver experiment. 

    :param rescaled_GRASP: normalized and rescaled GRASP array (in-plane and slice resolutions are correct)
    :param rescaled_Movienet: normalized and rescaled Movienet array (in-plane and slice resolutions are correct)
    :param rescaled_SELFIE: normalized and rescaled SELFIE array (in-plane and slice resolutions are correct)
    :param axial_slice: axial slice id
    :param sag_slice: sagittal slice id
    :param cor_slice: coronal slice id
    :param resolution: list of resolutions along x-,y-, and z-plane
    :param LIVER: bool if liver (True) or gyn (False) experiment
    :param savefoldername: parent foldername
    :param runname: foldername in parent foldername where to save plot
    :param filename: name of plot

    :return: None, but figure is saved accordinly. 
    """
    plt.style.use('dark_background')
    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(6.8, 5),
        gridspec_kw={'width_ratios': [1,1,1], 'height_ratios': [256, rescaled_SELFIE.shape[2]]}
    )
    plt.subplots_adjust(bottom=0.15, left= 0.01, right=0.99, top=0.9, wspace=0.01, hspace=0.05)
    
    if LIVER == True:
        axs[0, 0].imshow(
            np.fliplr(rescaled_GRASP[:,:,int(axial_slice*resolution[2])]), cmap='gray', aspect='equal')
        axs[0, 0].axis('off') 
        axs[0, 0].set_title("GRASP", size=12, color='white')
        axs[0, 1].imshow(
            np.fliplr(rescaled_Movienet[:,:,int(axial_slice*resolution[2])]), cmap='gray', aspect='equal')
        axs[0, 1].axis('off') 
        axs[0, 1].set_title("DCE-Movienet", size=12, color='white')
        axs[0, 2].imshow(
            np.fliplr(rescaled_SELFIE[:,:,int(axial_slice*resolution[2])]), cmap='gray')
        axs[0, 2].axis('off')
        axs[0, 2].set_title("SELFIE", size=12, color='white')

        im = axs[1, 0].imshow(
            np.fliplr(np.rot90(rescaled_GRASP[cor_slice,:,:])), cmap='gray')
        axs[1, 0].axis('off') 
        axs[1, 1].imshow(
            np.fliplr(np.rot90(rescaled_Movienet[cor_slice,:,:])), cmap='gray')
        axs[1, 1].axis('off') 
        axs[1, 2].imshow(
            np.fliplr(np.rot90(rescaled_SELFIE[cor_slice,:,:])), cmap='gray')
        axs[1, 2].axis('off') 

        cbar_ax = fig.add_axes([0.025, 0.12, 0.95, 0.015])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='intensity [a. u.]')

        # plt.tight_layout
        plt.savefig(f'{savefoldername}/exp_{runname}/{filename}_axial_coronal.png', dpi=300)
        plt.close()


    else: 
        axs[0, 0].imshow(
            rescaled_GRASP[:,:,int(axial_slice*resolution[2])], cmap='gray', aspect='equal')
        axs[0, 0].axis('off') 
        axs[0, 0].set_title("GRASP", size=12, color='white')
        axs[0, 1].imshow(
            rescaled_Movienet[:,:,int(axial_slice*resolution[2])], cmap='gray', aspect='equal')
        axs[0, 1].axis('off') 
        axs[0, 1].set_title("DCE-Movienet", size=12, color='white')
        axs[0, 2].imshow(
            rescaled_SELFIE[:,:,int(axial_slice*resolution[2])], cmap='gray')
        axs[0, 2].axis('off')
        axs[0, 2].set_title("SELFIE", size=12, color='white')

        im = axs[1, 0].imshow(
            np.rot90(rescaled_GRASP[:,sag_slice,:], 1), cmap='gray')
        axs[1, 0].axis('off') 
        axs[1, 1].imshow(
            np.rot90(rescaled_Movienet[:,sag_slice,:], 1), cmap='gray')
        axs[1, 1].axis('off') 
        axs[1, 2].imshow(
            np.rot90(rescaled_SELFIE[:,sag_slice,:], 1), cmap='gray')
        axs[1, 2].axis('off') 

        cbar_ax = fig.add_axes([0.025, 0.12, 0.94, 0.015])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='intensity [a. u.]')
        plt.style.use('dark_background')
        plt.savefig(f'{savefoldername}/exp_{runname}/{filename}_axial_sagittal.png', dpi=300)
        plt.close()


def make_gif(grasp, output_Movienet, output, outputfolder, savename: str, Tf: int, percent1: float, percent2: float, percent3: float):
    """
    Creates a gif of GRASP, Movienet, and SELFIE outputs.

    :param grasp: grasp image of shape [time, x, y]
    :param output_Movienet: output_Movienet image of shape [time, x, y]
    :param output: SELFIE output image of shape [time, x, y]
    :param outputfolder: foldername where to save gif
    :param savename: name of gif
    :param Tf: number of time frames
    :param percent1: percentile to normalize grasp data
    :param percent2: percentile to normalize movienet data
    :param percent3: percentile to normalize selfie data

    :return: None, but gif is saved
    """
    videoarray = np.zeros((output.shape[0], output.shape[1] * 3 + 45, output.shape[2]))
    videoarray[:, :output.shape[1], :] = grasp / np.percentile(grasp, percent1)
    videoarray[:, output.shape[1]+15:output.shape[1]*2+15, :] = output_Movienet/ np.percentile(output_Movienet, percent2)
    videoarray[:, output.shape[1]*2+30:output.shape[1]*3+30, :] = output / np.percentile(output_Movienet, percent3)

    videoarray_normalized = videoarray / np.max(videoarray) 

    # # # Save gif
    gif_file = outputfolder / f'{savename}.gif'
    with imageio.get_writer(gif_file, mode='I', fps=5, dpi=300, loop=0) as writer:
        for k in range(Tf):
            writer.append_data((videoarray_normalized[:, :, k] * 255).astype(np.uint8))


def make_gif_forward_reverse(
        grasp,
        output_Movienet,
        output,
        outputfolder,
        savename: str,
        Tf: int,
        percent1: float,
        percent2: float,
        percent3: float,
        ):
    """
    Creates a gif of GRASP, Movienet, and SELFIE outputs in forward and reverse order.

    :param grasp: grasp image of shape [time, x, y]
    :param output_Movienet: output_Movienet image of shape [time, x, y]
    :param output: SELFIE output image of shape [time, x, y]
    :param outputfolder: foldername where to save gif
    :param savename: name of gif
    :param Tf: number of time frames
    :param percent1: percentile to normalize grasp data
    :param percent2: percentile to normalize movienet data
    :param percent3: percentile to normalize selfie data

    :return: None, but gif is saved
    """
    videoarray = np.zeros((output.shape[0], output.shape[1] * 3 + 45, output.shape[2]))
    videoarray[:, :output.shape[0], :] = grasp / np.percentile(
        grasp, percent1)
    videoarray[:, output.shape[0]+15:output.shape[0]*2+15, :] = output_Movienet / np.percentile(
        output_Movienet, percent2)
    videoarray[:, output.shape[0]*2+30:output.shape[0]*3+30, :] = output / np.percentile(
        output, percent3)

    videoarray_normalized = videoarray / np.max(videoarray)

    # # # Save gif
    gif_file = outputfolder / f'{savename}.gif'
    with imageio.get_writer(gif_file, mode='I', fps=10, dpi=300, loop=0) as writer:
        for k in range(Tf):
            writer.append_data((videoarray_normalized[:, :, k] * 255).astype(np.uint8))
        reverse_Tf = list(range(Tf)[::-1])[1:-1]
        for k in reverse_Tf:
            writer.append_data((videoarray_normalized[:, :, k] * 255).astype(np.uint8))


def make_gif_zoom(
        grasp,
        output_Movienet,
        output,
        outputfolder,
        savename,
        x1,
        y1,
        Tf,
        percent1,
        percent2,
        percent3,
        size=80,
        zoom=2.5,
        scaling=1.15,
        orientation='bottom_top',
        zoom_box='top_left',
        fps=5):
    """
    Creates a gif of GRASP, Movienet, and SELFIE outputs with zoomed windows.

    :param grasp: grasp image of shape [time, x, y]
    :param output_Movienet: output_Movienet image of shape [time, x, y]
    :param output: SELFIE output image of shape [time, x, y]
    :param outputfolder: foldername where to save gif
    :param savename: name of gif
    :param x1: x start position for zoom window 1
    :param y1: y start position for zoom window 1
    :param Tf: number of time frames
    :param percent1: percentile to normalize grasp
    :param percent2: percentile to normalize Movienet
    :param percent3: percentile to normalize SELFIE
    :param size: size of zoomed area
    :param zoom: zoom factor
    :param scaling: scaling factor for max. image intensity
    :param orientation: orientation of diagonal lines, naming convention is 'ROI_zoombox'
    :param zoom_box: position of zoom_box, must be 'top_right' or 'top_left'
    :param fps: frames per second 

    :return: None, but gif is saved
    """
    img_dim = output.shape[0]
    Tf = output.shape[2]

    size = size
    zoom = zoom
    scaling = scaling

    grasp_norm = grasp / np.percentile(grasp, percent1)
    grasp_norm = grasp_norm / scaling
    output_Movienet_norm = output_Movienet / np.percentile(output_Movienet, percent2)
    output_Movienet_norm = output_Movienet_norm / scaling
    output_norm = output / np.percentile(output, percent3)
    output_norm = output_norm / scaling

    videoarray = np.zeros((img_dim, img_dim * 3 + 45, Tf))
    videoarray[:, :output.shape[1], :] = grasp_norm
    videoarray[:, output.shape[1]+15:output.shape[1]*2+15, :] = output_Movienet_norm
    videoarray[:, output.shape[1]*2+30:output.shape[1]*3+30, :] = output_norm

    grasp = np.clip(scipy.ndimage.zoom(grasp_norm, [zoom, zoom, 1]), 0, None)
    output_Movienet = np.clip(scipy.ndimage.zoom(output_Movienet_norm, [zoom, zoom, 1]), 0, None)
    output = np.clip(scipy.ndimage.zoom(output_norm, [zoom, zoom, 1]), 0, None)
    
    #perfrom zoom only if x1 and y1 are != 0
    if x1 != 0 and y1 !=0:
        # put zoom in top right or top left corner
        if zoom_box == 'top_right':
            videoarray[:size, img_dim-size:img_dim, :] = \
                grasp[x1:x1+size, y1:y1+size, :]
            videoarray[:size, img_dim*2+15-size:img_dim*2+15, :] = \
                output_Movienet[x1:x1+size, y1:y1+size, :]
            videoarray[:size, img_dim*2+30+img_dim-size:img_dim*3+30, :] = \
                output[x1:x1+size, y1:y1+size, :]
        if zoom_box == 'top_left':
            videoarray[:size, 0:size, :] = \
                grasp[x1:x1+size, y1:y1+size, :]
            videoarray[:size, img_dim*1+15:img_dim*1+15+size, :] = \
                output_Movienet[x1:x1+size, y1:y1+size, :]
            videoarray[:size, img_dim*2+30:img_dim*2+30+size, :] = \
                output[x1:x1+size, y1:y1+size, :]
   
        for idx, i in enumerate([0, 15, 30]):
            # framing
            if zoom_box == 'top_right':
                idx += 1
                videoarray[:size, img_dim*idx-size+i-1, :] = 1
                videoarray[:size, img_dim*idx+i-1, :] = 1
                videoarray[0, img_dim*idx-size+i:img_dim*idx+i, :] = 1
                videoarray[size, img_dim*idx-size+i:img_dim*idx+i, :] = 1
                idx -=1
            if zoom_box == 'top_left':
                videoarray[:size, img_dim*idx+i, :] = 1
                videoarray[:size, img_dim*idx+i+size, :] = 1
                videoarray[0, img_dim*idx+i:img_dim*idx+i+size, :] = 1
                videoarray[size, img_dim*idx+i:img_dim*idx+i+size, :] = 1

            videoarray[int(x1/zoom), int(y1/zoom)+img_dim*idx+i:int((y1+size)/zoom)+img_dim*idx+i, :] = 1
            videoarray[int((x1+size)/zoom), int(y1/zoom)+img_dim*idx+i:int((y1+size)/zoom)+img_dim*idx+i, :] = 1
            videoarray[int(x1/zoom):int((x1+size)/zoom), int(y1/zoom)-1+img_dim*idx+i, :] = 1
            videoarray[int(x1/zoom):int((x1+size)/zoom), int((y1+size)/zoom)-1+img_dim*idx+i, :] = 1

            # diagonal lines
            # a couple of options are implemented so far
            if orientation == 'left_right':
                rr, cc = line(int(int(x1)/zoom), int((y1)/zoom)+img_dim*idx+i - 1, 0, img_dim*idx+i+size)
                videoarray[rr, cc, :] = 1
                rr, cc, = line(int(int(x1+size)/zoom), int((y1)/zoom)+img_dim*idx+i -1, size, img_dim*idx+i+size)
                videoarray[rr, cc, :] = 1
            elif orientation == 'left_bottom':
                rr, cc = line(int(int(x1)/zoom), int((y1)/zoom)+img_dim*idx+i - 1, size, img_dim*idx+i)
                videoarray[rr, cc, :] = 1
                rr, cc, = line(int(int(x1+size)/zoom), int((y1)/zoom)+img_dim*idx+i -1, size, img_dim*idx+i+size)
                videoarray[rr, cc, :] = 1
            elif orientation == 'top_right':
                rr, cc = line(int(int(x1)/zoom), int((y1)/zoom)+img_dim*idx+i - 1, size, img_dim*idx+i+size)
                videoarray[rr, cc, :] = 1
                rr, cc, = line(int(int(x1)/zoom), int((y1+size)/zoom)+img_dim*idx+i -1, 0, img_dim*idx+i+size)
                videoarray[rr, cc, :] = 1
            elif orientation == 'top_left':
                rr, cc = line(int(int(x1)/zoom), int((y1)/zoom)+img_dim*idx+i - 1, 0, img_dim*(idx+1)+i-size-1)
                videoarray[rr, cc, :] = 1
                rr, cc, = line(int(int(x1)/zoom), int((y1+size)/zoom)+img_dim*idx+i -1, size, img_dim*(idx+1)+i-size-1)
                videoarray[rr, cc, :] = 1
            elif orientation == 'right_left':
                rr, cc = line(int(int(x1)/zoom), int((y1+size)/zoom)+img_dim*idx+i, 0, img_dim*(idx+1)+i-size-1)
                videoarray[rr, cc, :] = 1
                rr, cc, = line(int(int(x1+size)/zoom), int((y1+size)/zoom)+img_dim*idx+i, size, img_dim*(idx+1)+i-size-1)
                videoarray[rr, cc, :] = 1
            elif orientation == 'top_bottom':
                rr, cc = line(int(int(x1)/zoom), int((y1)/zoom)+img_dim*idx+i - 1, size, img_dim*(idx+1)+i - size- 1)
                videoarray[rr, cc, :] = 1
                rr, cc, = line(int(int(x1)/zoom), int((y1+size)/zoom)+img_dim*idx+i -1, size, img_dim*(idx+1)+i - 1)
                videoarray[rr, cc, :] = 1
            else: 
                raise NotImplementedError('So far, only a subset of oriantation options are implemented.')

    videoarray_normalized = videoarray / np.max(videoarray)

    # Save gif
    gif_file = outputfolder / f'{savename}_zoom.gif'
    with imageio.get_writer(gif_file, mode='I', fps=fps, dpi=300, loop=0) as writer:
        for k in range(Tf):
            writer.append_data((videoarray_normalized[:, :, k] * 255).astype(np.uint8))
