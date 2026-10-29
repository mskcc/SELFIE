
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from medimetrics.metrics.line_corr import MLC
from medimetrics.metrics.blur_widths import BlurWidths
from medimetrics.metrics.total_var import MeanTotalVar
from src.visualization.normalize import minmax_binning
from src.utilities.helper import create_mask


def extract_metrics(filename, Tf, NUFFT_array_all, GRASP_array_all, Movienet_array_all, SELFIE_array_all, outputfolder):
    """
    Calculates metric values per slice and time frame and save it in dictionary. 
    
    :param filename: current filename/studyname
    :param Tf: number of time frames
    :param NUFFT_array_all: normalized array of NUFFT reconstruction
    :param GRASP_array_all: normalized array of grasp reconstruction
    :param Movienet_array_all: normalized array of Movienet reconstruction
    :param SELFIE_array_all: normalized array of SELFIE reconstruction
    :param outputfolder: foldername where to save metrics

    :return: None, but dicitionary is saved.
    """
    metrics = dict()
    metrics['MLC'] = dict()
    metrics['BEW'] = dict()
    metrics['MTV'] = dict()
    metrics['MLC_in'] = dict()

    mlc = MLC()
    bew = BlurWidths()
    mtv = MeanTotalVar()

    for key in metrics.keys():
        metrics[key][filename] = dict()

    for key in metrics.keys():
        metrics[key][filename]['NUFFT'] = dict()
        metrics[key][filename]['Movienet'] = dict()
        metrics[key][filename]['SELFIE'] = dict()

        if key == 'BEW' or key == 'MLC' or key == 'MTV' or key == 'MLC_in':
            metrics[key][filename]['grasp'] = dict()

    for slice_nr, slice_idx in enumerate(range(GRASP_array_all.shape[2])):
        print('Extracting metrcis for slice_nr', slice_nr)
        # Extract study name
        slicename = f'slice{slice_idx}'

        rnufft = NUFFT_array_all[:, :, slice_idx, :]
        grasp = GRASP_array_all[:, :, slice_idx, :]
        output_Movienet = Movienet_array_all[:, :, slice_idx, :]
        output = SELFIE_array_all[:, :, slice_idx, :]

        # masking for body region
        grasp_mean = np.mean(grasp, axis=2)

        mask = create_mask(grasp_mean)

        rnufft_bg = rnufft * (1- mask[:,:,None])
        output_bg = output * (1 - mask[:,:,None])
        grasp_bg = grasp * (1 - mask[:,:,None])
        output_Movienet_bg = output_Movienet * (1 - mask[:,:,None])

        rnufft = rnufft * mask[:,:,None]
        output = output * mask[:,:,None]
        grasp = grasp * mask[:,:,None]
        output_Movienet = output_Movienet * mask[:,:,None]

        nufft_bew = []
        grasp_bew = []
        Movienet_bew = []
        SELFIE_bew = []
        
        nufft_mlc = []
        grasp_mlc = []
        Movienet_mlc = []
        SELFIE_mlc = []

        nufft_mlc_in = []
        grasp_mlc_in = []
        Movienet_mlc_in = []
        SELFIE_mlc_in = []

        nufft_mtv = []
        grasp_mtv = []
        Movienet_mtv = []
        SELFIE_mtv = []

        for tf in range(Tf):
            nufft_bew.append(bew.compute(minmax_binning(rnufft)[:,:,tf]))
            grasp_bew.append(bew.compute(minmax_binning(grasp)[:,:,tf]))
            Movienet_bew.append(bew.compute(minmax_binning(output_Movienet)[:,:,tf]))
            SELFIE_bew.append(bew.compute(minmax_binning(output)[:,:,tf]))

            nufft_mlc.append(mlc.compute(minmax_binning(rnufft_bg)[:,:,tf]))
            grasp_mlc.append(mlc.compute(minmax_binning(grasp_bg)[:,:,tf]))
            Movienet_mlc.append(mlc.compute(minmax_binning(output_Movienet_bg)[:,:,tf]))
            SELFIE_mlc.append(mlc.compute(minmax_binning(output_bg)[:,:,tf]))

            nufft_mlc_in.append(mlc.compute(minmax_binning(rnufft)[:,:,tf]))
            grasp_mlc_in.append(mlc.compute(minmax_binning(grasp)[:,:,tf]))
            Movienet_mlc_in.append(mlc.compute(minmax_binning(output_Movienet)[:,:,tf]))
            SELFIE_mlc_in.append(mlc.compute(minmax_binning(output)[:,:,tf]))

            nufft_mtv.append(mtv.compute(minmax_binning(rnufft)[:,:,tf]))
            grasp_mtv.append(mtv.compute(minmax_binning(grasp)[:,:,tf]))
            Movienet_mtv.append(mtv.compute(minmax_binning(output_Movienet)[:,:,tf]))
            SELFIE_mtv.append(mtv.compute(minmax_binning(output)[:,:,tf]))

        metrics['BEW'][filename]['NUFFT'][slicename] = nufft_bew
        metrics['BEW'][filename]['grasp'][slicename] = grasp_bew
        metrics['BEW'][filename]['Movienet'][slicename] = Movienet_bew 
        metrics['BEW'][filename]['SELFIE'][slicename] = SELFIE_bew

        metrics['MLC'][filename]['NUFFT'][slicename] = nufft_mlc
        metrics['MLC'][filename]['grasp'][slicename] = grasp_mlc
        metrics['MLC'][filename]['Movienet'][slicename] = Movienet_mlc
        metrics['MLC'][filename]['SELFIE'][slicename] = SELFIE_mlc

        metrics['MLC_in'][filename]['NUFFT'][slicename] = nufft_mlc_in
        metrics['MLC_in'][filename]['grasp'][slicename] = grasp_mlc_in
        metrics['MLC_in'][filename]['Movienet'][slicename] = Movienet_mlc_in
        metrics['MLC_in'][filename]['SELFIE'][slicename] = SELFIE_mlc_in

        metrics['MTV'][filename]['NUFFT'][slicename] = nufft_mtv
        metrics['MTV'][filename]['grasp'][slicename] = grasp_mtv
        metrics['MTV'][filename]['Movienet'][slicename] = Movienet_mtv
        metrics['MTV'][filename]['SELFIE'][slicename] = SELFIE_mtv

        print('Done')

    np.savez(f'{outputfolder}/metric_{filename}_slice_and_time_wise.npz', **metrics)


def load_metrics(outputfolder, filename):
    """
    Load the dictionary containing metric values.

    :param outputfolder: folder where dictionary is stored
    :param filename: filename/studyname of metric *npz file.

    :return: the metrics dictionary
    """
    metric_dict = np.load(f'{outputfolder}/metric_{filename}_slice_and_time_wise.npz', allow_pickle=True)
    metrics = dict()
    for key in metric_dict:
        metrics[key] = metric_dict[key].item()
    return metrics


def calculate_mean_and_std(
        rnufft_roi_all_norm: list,
        grasp_roi_all_norm: list,
        output_Movienet_roi_all_norm: list,
        output_roi_all_norm: list,
        masks: list,
        roi_number: int,
        Tf: int):
    """
    Calculates the mean and standard deviation for values within a specified ROI
    over a given number of time frames. For each input dataset, it computes the mean and
    standard deviation (mean ± standard deviation).

    :param rnufft_roi_all_norm: list of normalized rNUFFT data within the ROI, with shape (Tf, height, width)
    :param grasp_roi_all_norm: list of normalized GRASP data within the ROI, with shape (Tf, height, width)
    :param output_Movienet_roi_all_norm: list of normalized Movienet data within the ROI, with shape (Tf, height, width)
    :param output_roi_all_norm: list of normalized output data within the ROI, with shape (Tf, height, width)
    :param masks: list of binary masks indicating the region of interest (ROI)
    :param roi_number: index for selecting the specific ROI mask from the `masks` list
    :param Tf: number of time frames over which the statistics are calculated

    :return mean_std_dict: dictionary containing the means and confidence bounds (mean ± std) for each input
        dataset within the ROI across the specified time frames. The keys are as follows:
        - 'rnufft_mean', 'grasp_mean', 'output_Movienet_mean', 'output_mean': Mean values.
        - 'rnufft_plus', 'grasp_plus', 'output_Movienet_plus', 'output_plus': Mean + std.
        - 'rnufft_minus', 'grasp_minus', 'output_Movienet_minus', 'output_minus': Mean - std
    """

    rnufft_roi_all_norm_non_zero_mean = [
        np.mean(rnufft_roi_all_norm[:,:,i][np.nonzero(masks[roi_number])]) for i in range(Tf)]
    grasp_roi_all_norm_non_zero_mean = [
        np.mean(grasp_roi_all_norm[:,:,i][np.nonzero(masks[roi_number])]) for i in range(Tf)]
    output_Movienet_roi_all_norm_non_zero_mean = [
        np.mean(output_Movienet_roi_all_norm[:,:,i][np.nonzero(masks[roi_number])]) for i in range(Tf)]
    output_roi_all_norm_non_zero_mean = [
        np.mean(output_roi_all_norm[:,:,i][np.nonzero(masks[roi_number])]) for i in range(Tf)]

    mean_std_dict = dict()
    mean_std_dict['rnufft_mean'] = rnufft_roi_all_norm_non_zero_mean
    mean_std_dict['grasp_mean'] = grasp_roi_all_norm_non_zero_mean
    mean_std_dict['output_Movienet_mean'] = output_Movienet_roi_all_norm_non_zero_mean
    mean_std_dict['output_mean'] = output_roi_all_norm_non_zero_mean

    rnufft_roi_all_norm_non_zero_std = [
        np.std(rnufft_roi_all_norm[:,:,i][np.nonzero(masks[roi_number])]) for i in range(Tf)]
    grasp_roi_all_norm_non_zero_std = [
        np.std(grasp_roi_all_norm[:,:,i][np.nonzero(masks[roi_number])]) for i in range(Tf)]
    output_Movienet_roi_all_norm_non_zero_std = [
        np.std(output_Movienet_roi_all_norm[:,:,i][np.nonzero(masks[roi_number])]) for i in range(Tf)]
    output_roi_all_norm_non_zero_std = [
        np.std(output_roi_all_norm[:,:,i][np.nonzero(masks[roi_number])]) for i in range(Tf)]

    mean_std_dict['rnufft_plus'] = [
        rnufft_roi_all_norm_non_zero_mean[i] + rnufft_roi_all_norm_non_zero_std[i] for i in range(Tf)]
    mean_std_dict['rnufft_minus'] = [
        rnufft_roi_all_norm_non_zero_mean[i] - rnufft_roi_all_norm_non_zero_std[i] for i in range(Tf)]

    mean_std_dict['grasp_plus'] = [
        grasp_roi_all_norm_non_zero_mean[i] + grasp_roi_all_norm_non_zero_std[i] for i in range(Tf)]
    mean_std_dict['grasp_minus'] = [
        grasp_roi_all_norm_non_zero_mean[i] - grasp_roi_all_norm_non_zero_std[i] for i in range(Tf)]

    mean_std_dict['output_Movienet_plus'] = [
        output_Movienet_roi_all_norm_non_zero_mean[i] + output_Movienet_roi_all_norm_non_zero_std[i] for i in range(Tf)]
    mean_std_dict['output_Movienet_minus'] = [
        output_Movienet_roi_all_norm_non_zero_mean[i] - output_Movienet_roi_all_norm_non_zero_std[i] for i in range(Tf)]

    mean_std_dict['output_plus'] = [
        output_roi_all_norm_non_zero_mean[i] + output_roi_all_norm_non_zero_std[i] for i in range(Tf)]
    mean_std_dict['output_minus'] = [
        output_roi_all_norm_non_zero_mean[i] - output_roi_all_norm_non_zero_std[i] for i in range(Tf)]

    return mean_std_dict


def calculate_start_slope(mean_std_dict: dict, Tf: int, coeff: int=30, plot_path: str=None):
    """
    Calculates features of contrast rise (position of max value 'peak_pos', start of peak 'start',
    end of peak 'end', and slope 'slope') for each of the four methods 'rnufft_mean', 'grasp_mean',
    'output_Movienet_mean', 'output_mean'. The features are extracted using a polynomial fit on the mean
    contrast enhancement.

    :param mean_std_dict: dictionary containing the means and confidence bounds (mean ± std) for each input
    :param Tf: number of time frames
    :param coeff: number of coefficients for fitting
    :param plot_path: path where plots are saved

    :return: dictionary with peak_pos, start, end, and slope as keys for each method.
    """
    start_slope = dict()
    if plot_path is not None:
        fig, axs = plt.subplots(1, 4, figsize=(8, 4))
        fig.subplots_adjust(left=0.1, right=0.99)


    for index, method in enumerate(['rnufft_mean', 'grasp_mean', 'output_Movienet_mean', 'output_mean']):

        factor = np.sum(np.array(mean_std_dict[method]) * np.array(mean_std_dict['rnufft_mean'])) / np.sum(
            np.array(mean_std_dict[method]) * np.array(mean_std_dict[method]))

        data = factor * np.array(mean_std_dict[method])
        bins = (Tf-1)/10000
        x = np.arange(0, Tf-1, bins)
        start_slope[method] = dict()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            a = np.polyfit(range(Tf), data, coeff)
            p = np.poly1d(a)
            der1 = np.poly1d.deriv(p, m=1)
            der2 = np.poly1d.deriv(p, m=2)
            p_der1 = np.poly1d(der1)
            p_der2 = np.poly1d(der2)

        # get values of 1st derivative
        values_der1 = p_der1(x)

        # find max of 1st derivative that defines 1st derivative peak position
        pos_max = np.where(values_der1 == np.max(values_der1[1000:-1000]))
        start_slope[method]['peak_pos'] = np.round((pos_max[0][0]) * bins, decimals=3)

        # get values of 2nd derivative
        values_der2 = p_der2(x)
        # get 2nd derivative at peak pos
        max_2nd = values_der2[pos_max[0][0]]
        # go left from that position until the 2nd derivative is maximal. This is the position of the start.
        for i in range(len(x)):
            if values_der2[pos_max[0][0] - i] > max_2nd:
                max_2nd = values_der2[pos_max[0][0] - i]
            if values_der2[pos_max[0][0] - i] < max_2nd:
                break
        start = (pos_max[0][0] - i) * bins + 0.5 * bins
        start_slope[method]['start'] = np.round(start, decimals=3)

        # get 2nd derivative at peak pos
        min_2nd = values_der2[pos_max[0][0]]
        # go right from that position until the 2nd derivative is minimal. This is the position of the peak.
        for i in range(len(x)):
            if values_der2[pos_max[0][0] + i] < min_2nd:
                min_2nd = values_der2[pos_max[0][0] + i]
            if values_der2[pos_max[0][0] + i] > min_2nd:
                break
        end = (pos_max[0][0] + i) * bins - 0.5 * bins
        start_slope[method]['end'] = np.round(end, decimals=3)

        slope = (p(end) - p(start)) / (end - start)
        start_slope[method]['slope'] = np.round(slope, decimals=3)

        if plot_path is not None:
            axs[index].plot(data, label='data')
            axs[index].plot(p(range(Tf)), label='poly-fit')
            axs[index].plot(start_slope[method]['start'], p(int(np.round(start_slope[method]['start']))), 'o', label='start')
            axs[index].plot(start_slope[method]['end'], p(int(np.round(start_slope[method]['end']))), 'o', label='end')
            axs[index].set_title(method)
            axs[index].legend()
            axs[index].set_xlabel('Time frames [a.u.]')
            axs[0].set_ylabel('Signal intensity [a.u.]')
           

    if plot_path is not None:
        plt.savefig(plot_path)
        plt.close()

    return start_slope


def print_metrics(filenames_metric, outputfolder):
    """
    Prints metrics aggregated across study/studies.
    
    :param filenames_metric: studies to use
    :param outputfolder: outputfolder folder that contains metrics
    
    :return: None but results are printed.
    """
    print("#############")
    for metric in [ 'BEW', 'MLC', 'MLC_in', 'MTV']: 
        print(metric)
        print("#############")
        for method in ['grasp', 'Movienet', 'SELFIE']:
            print(method)
            metric_list = []
            if len(filenames_metric) == 1:
                metrics = load_metrics(outputfolder, filenames_metric[0])
                print(np.round(np.mean(np.asarray(list(metrics[metric][filenames_metric[0]][method].values()))), decimals=3), "+-",  np.round(np.std(np.asarray(list(metrics[metric][filenames_metric[0]][method].values()))), decimals=3))
            else:
                for study in filenames_metric:
                    metrics = load_metrics(outputfolder, study)
                    metric_list.append(np.mean(np.asarray(list(metrics[metric][study][method].values()))))
                print(np.round(np.mean(metric_list), decimals=3), "+-",  np.round(np.std(metric_list), decimals=3))
        print("#############")


def extract_data(data, data_Movienet, PURE: bool):
    """
    Extract results of different reconstruction methods and performs flip and transpose accordingly.

    :param data: dataset dict of SELFIE
    :param data_Movienet: dataset dict of Movienet
    :param PURE: boolean that is either True or False dependent on Pure correction was applied or not

    :param return: the NUFFT, GRASP, Movienet, and SELFIE slices
    """
    if PURE is True:
        rnufft = np.flip(data['rnufft_corrected'].transpose(2, 0, 1), axis=2)
        output = np.flip(data['selfie_corrected'].transpose(2, 0, 1), axis=2)
        grasp = np.flip(data['grasp_corrected'].transpose(2, 0, 1), axis=2)
        output_Movienet = np.flip(data_Movienet['movienet_corrected'].transpose(2, 0, 1), axis=2)
    if PURE is False:
        rnufft = data['rnufft'].squeeze()
        output = data['output'].squeeze()
        grasp = data['grasp'].squeeze()
        output_Movienet = data_Movienet['output'].squeeze()            
    return rnufft, grasp, output_Movienet, output


def aggregate_metrics_and_csv(filenames, metric_names, Tf, outputfolder):
    """
    Loads metrics, aggregates mean along slices and time frames, and writes it into a *.csv.
    
    :param filenames: filenames/studynames to analyze
    :param metric_names: list of metric_names to analyze
    :param Tf: number of time frames
    :param outputfolder: foldername of where metrics are stored and where *.csv will be stored

    :return: None, but challengeR.csv is saved accordingly. 

    """
    rows = []
    for filename in filenames:
        metrics = load_metrics(outputfolder, filename)
        for metric in metric_names:
                for case, algos in metrics[metric].items():
                    for algo, slices in algos.items():
                        values = np.zeros((len(slices), Tf))
                        idx = 0
                        for slice, metric_values in slices.items():
                            values[idx] = metric_values
                            idx += 1
                        if metric == 'MLC_in':
                            rows.append({
                            "Task": metric,
                            "Case": f'{case}',
                            "Algorithm": algo,
                            "Metric": 1 - np.mean(values)
                            })
                        else:
                            rows.append({
                                "Task": metric,
                                "Case": f'{case}',
                                "Algorithm": algo,
                                "Metric": np.mean(values)
                            })
    df = pd.DataFrame(rows)
    df.to_csv(f'{outputfolder}/challengeR.csv')