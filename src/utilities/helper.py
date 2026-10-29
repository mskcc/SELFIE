import pickle
import random
import torch
import numpy as np
from pathlib import Path
import skimage
from skimage.morphology import disk
from skimage.segmentation import clear_border
from src.architecture.Movienet_CNN import MovieNet,TwoDPlusTimeUnet_larger, UNet3D
from src.data.dynamic_dataset import MovienetDataset_withPriors_shifted_complex_norm_90
from src.visualization.normalize import minmax_norm_0_1
from typing import Union


def savelist(filename: str, data: list):
    """
    Saves a list as pickle file.

    :param filename: the save path
    :param data: the list to save

    :return: None, but list is saved
    """

    open_file = open(filename, "wb")
    pickle.dump(data, open_file)
    open_file.close()


def manual_seed(seed: int):
    """
    Set seed for numpy, random, torch using seed.

    :param seed: int of seed

    :return: None, but seeds are set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker():
    """
    Reseeds every worker with the initial seed of torch. This enables that the workers all use the same seed
    and produce the same random numbers.

    :return: None
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def get_device(cfg):
    """
    Reads config to determine device.

    :param cfg: configs

    :return: device
    """
    if cfg.hparams.cuda is True:
        device = torch.device(cfg.hparams.node if torch.cuda.is_available() else "cpu")
    elif cfg.hparams.cuda == "parallel":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    return device


def get_model(cfg, device):
    """
    Reads config to determine and load model. The model is also moved to device.

    :param cfg: configs
    :param device: the device

    :return: model
    """
    if cfg.hparams.modelname == 'Movienet':
        model = MovieNet(cfg.hparams.Ninputchannels, cfg.hparams.Noutputchannels).to(device)
    elif cfg.hparams.modelname == 'TwoDPlusTimeUnet_larger':
        model = TwoDPlusTimeUnet_larger(
            cfg.hparams.Ninputchannels, cfg.hparams.Noutputchannels).to(device)
    elif cfg.hparams.modelname == 'UNet3D':
        model = UNet3D(
            cfg.hparams.Ninputchannels, cfg.hparams.Noutputchannels).to(device)
    else:
        raise NotImplementedError('This modelname does not exist in framework. We have Movienet, \
            TwoDPlusTimeUnet_larger, and UNet3D.')
    return model


def get_averaging_kernel(ana_frames, device):
    """
    Reads config to determine averaging kernel, that is directly moved to device.

    :param ana_frames: number of frames to average
    :param device: the device

    :return: kernel
    """
    diviser = int(ana_frames)
    k = torch.nn.Conv1d(1, 1, kernel_size=diviser, bias=False, stride=1).to(torch.float32).to(device)
    k.weight = torch.nn.Parameter(torch.ones_like(k.weight) * 1/diviser)

    return k


def get_dataset(cfg):
    """
    Reads config to load dataset.

    :param cfg: configs

    :return: dataset
    """
    if cfg.hparams.complex is True:
        MovieDataset = MovienetDataset_withPriors_shifted_complex_norm_90
           
    else:
        raise NotImplementedError("Currently, only MovienetDataset_withPriors_shifted_complex_norm_90 is reliable.")
    return MovieDataset


def get_criterions(
        method1: str = 'MSE', reduction1: str = 'mean', method2: str = 'MSE', reduction2: str = 'mean'):
    """
    Defines criterions.

    :param method1: name of method1, either 'MSE' or 'SmoothL1'
    :param reduction1: name of reduction method for method1, either 'mean' or 'sum'
    :param method2: name of method2, either 'MSE' or 'SmoothL1'
    :param reduction2: name of reduction method for method2, either 'mean' or 'sum'

    :return: criterions
    """
    if reduction1 not in ['mean', 'sum']:
        raise NotImplementedError("Reduction must be either 'mean' or 'sum'.")
    if reduction2 not in ['mean', 'sum']:
        raise NotImplementedError("Reduction must be either 'mean' or 'sum'.")

    if method1 == 'MSE':
        criterion1 = torch.nn.MSELoss(reduction=reduction1)
    elif method1 == 'Smoothl1':
        criterion1 = torch.nn.SmoothL1Loss(reduction=reduction1)
    else:
        raise NotImplementedError("Method must be either 'MSE' or 'SmoothL1'.")

    if method2 == 'MSE':
        criterion2 = torch.nn.MSELoss(reduction=reduction2)
    elif method2 == 'Smoothl1':
        criterion2 = torch.nn.SmoothL1Loss(reduction=reduction2)
    else:
        raise NotImplementedError("Method_edge must be either 'MSE' or 'SmoothL1'.")

    return criterion1, criterion2


def load_model(cfg, model, model_path, mode, fold_name=None):
    """
    Loads pre-trained model in either eval or train mode.

    :param cfg: configs
    :param model: model architecture
    :param model_path: path where model weights are stored
    :param mode: 'train' or 'eval' mode
    :param fold_name: optional fold_name

    :return: model with pre-trained weights
    """
    if fold_name != None:
        model.load_state_dict(torch.load(Path(model_path / f'{cfg.hparams.experiment_name_load}{fold_name}.pt')))
    else:
        model.load_state_dict(torch.load(Path(model_path / f'{cfg.hparams.experiment_name_load}.pt')))

    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()
    else:
        raise NotImplementedError("Mode to load model must be either 'train' or 'val.")
    return model


def save_model(cfg, val_loss_list, val_loss, model, model_path):
    """
    Saves best and last model dependent on parallel or single GPU training.

    :param cfg: configs
    :param val_loss_list: list with val_loss averaged per epoch
    :param val_loss: current val_loss
    :param model: model architecture
    :param model_path: path where to save model

    :return: None but models are saved
    """
    if len(val_loss_list) > 1:
        if val_loss < min(val_loss_list[:-1]):
            if cfg.hparams.cuda == "parallel":
                torch.save(model.module.state_dict(), Path(model_path / f'{cfg.hparams.experiment_name}.pt'))
            else:
                torch.save(model.state_dict(), Path(model_path / f'{cfg.hparams.experiment_name}.pt'))
    else:
        if cfg.hparams.cuda == "parallel":
            torch.save(model.module.state_dict(), Path(model_path / f'{cfg.hparams.experiment_name}.pt'))
        else:
            torch.save(model.state_dict(), Path(model_path / f'{cfg.hparams.experiment_name}.pt'))
    # save the last model
    if cfg.hparams.cuda == "parallel":
        torch.save(model.module.state_dict(), Path(model_path / f'{cfg.hparams.experiment_name}_last.pt'))
    else:
        torch.save(model.state_dict(), Path(model_path / f'{cfg.hparams.experiment_name}_last.pt'))


def TV_along_time(img: torch.Tensor):
    """
    Calculates absolute differences of a 4D image of shape [bs, time, x, y] along the time
    dimension (second dimension). The mean of the difference values is returned.

    :param img: the image of shape [bs, time, x, y]

    :return: mean total variation
    """
    diff = img[:, 1:, ...] - img[:, :-1, ...]
    score = diff.abs().mean()
    return score


def optimization_loop(
        cfg,
        mode: str,
        device,
        model,
        optimizer,
        criterion1,
        criterion2,
        epoch: int,
        batch: int,
        results_path: Union[str, Path],
        test_name: str,
        tp_to_show: int,
        loss_per_batch: list,
        coilwise_rnufft: torch.Tensor,
        grasp: torch.Tensor,
        rnufft: torch.Tensor,
        ana_prior: torch.Tensor,
        ana_frames,
        scaler,
        time_measurement: bool = False):
    """
    The optimization loop, that applies the model on coilwise nufft images. The output can be compared with
    the reference image, the anatomical self-reference, and the temporal self-reference. Additional a sparsity constraint, 
    temporal total variation, can be computed on the output. The loop can be performed during training with update steps 
    or just as a single forward step during validation or testing. 
    You can choose to run the computations on CPU or GPU and with automatic precision scaler or not.

    :param cfg: configs
    :param mode: string that defines mode, either 'training', 'validation', or 'testing'
    :param device: torch device
    :param model: the actual model
    :param optimizer: the optimizer
    :param criterion1: one criterion used for GRASP and anatomical self-reference
    :param criterion2: another criterion used for temporal self-reference
    :param epoch: the current epoch number
    :param batch: the current batch number
    :param results_path: string or Path to store results
    :param test_name: the name of the dataset
    :param tp_to_show: index of time point used for plotting
    :param loss_per_batch: list with losses per batch
    :param coilwise_rnufft: torch tensor of coilwise nuffts
    :param grasp: torch tensor of grasp
    :param rnufft: torch tensor of rnufft
    :param ana_prior: torch tensor of anatomical self-reference
    :param ana_frames: number of frames to average
    :param low_rank_model: the model to perform low rank computation
    :param scaler: the scaler to perform automatic precision
    :param time_measurement: bool that defines what data to lift from GPU/CPU

    :return: loss_per_batch, if
        'training' or 'validation' mode is on and epoch != cfg.hparams.logging_epoch_freq
    :return: loss_per_batch, grasp, output (all detached numpy arrays), if
        'training' or 'validation' mode is on
    :return: loss_per_batch, rnufft, coilwise_rnufft, grasp, and output (all detached numpy arrays), if 
        in 'testing' mode and 'time_measurement' is False
    :return: loss_per_batch, output (detached numpy array), if 
        in 'testing' mode and 'time_measurement' is True
    """

    if mode == 'training':
        rnufft, coilwise_rnufft, grasp, ana_prior = rnufft.to(device), coilwise_rnufft.to(device), grasp.to(device), \
            ana_prior.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                output = model(coilwise_rnufft)
        else:
            output = model(coilwise_rnufft)
            if cfg.hparams.modelname != 'Movienet':
                output = torch.abs(output[:, 0, ...] + 1j * output[:, 1, ...])

    else:
        if scaler is not None:
            with torch.no_grad():
                rnufft, coilwise_rnufft, grasp, ana_prior = rnufft.to(device), coilwise_rnufft.to(device), \
                    grasp.to(device), ana_prior.to(device)
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    output = model(coilwise_rnufft)

        else:
            with torch.no_grad():
                if time_measurement is True:
                    coilwise_rnufft = coilwise_rnufft.to(device)
                else:
                    rnufft, coilwise_rnufft, grasp, ana_prior = rnufft.to(device), coilwise_rnufft.to(device), \
                        grasp.to(device), ana_prior.to(device)
                output = model(coilwise_rnufft)
                
                if cfg.hparams.modelname != 'Movienet':
                    output = torch.abs(output[:, 0, ...] + 1j * output[:, 1, ...])

    batch_size = output.shape[0]
    # GRASP loss if L1_lambda > 0
    if cfg.hparams.L1_lambda > 0:
        loss1 = criterion1(output, grasp)
    else:
        loss1 = 0

    # Ana. prior loss if L2_lambda > 0
    if cfg.hparams.L2_lambda > 0:
        # calculate number of frames to average
        frames = ana_frames - 1

        output_small = torch.zeros(output.shape).to(output.device)
        for bs in range(batch_size):
            # reshape of output to [-1, 1, time] and padding of ouput along time dimension in "circular" mode
            output_small_padded = torch.nn.functional.pad(output[bs:bs+1].reshape(
                1,
                cfg.hparams.timeframes,
                -1).swapaxes(1, 2).reshape(
                    -1,
                    cfg.hparams.timeframes).unsqueeze(1), (0, frames[bs]), "circular")
            # apply averaging kernel on padded output in sliding window fashion
            k = get_averaging_kernel(frames[bs]+1, device)
            output_small_onebatch = k(output_small_padded)
            # reshape averaged output to [bs, time, x, y]
            output_small_onebatch = output_small_onebatch.reshape(
                1, -1, cfg.hparams.timeframes).swapaxes(
                    1, 2).reshape(1, cfg.hparams.timeframes, cfg.hparams.img_size, cfg.hparams.img_size)
            output_small[bs] = output_small_onebatch

        if cfg.hparams.L2_method == "Average":
            loss2 = criterion1(ana_prior, output_small)
        else:
            print(f'The method {cfg.hparams.L2_method} does not exist. Choose Average or implement sth. else.')
            loss2 = 0
    else:
        loss2 = 0

    # Temp. prior loss if L3_lambda > 0
    if cfg.hparams.L3_lambda > 0:
        loss3 = criterion2(rnufft, output)
    else:
        loss3 = 0

    # TV loss if TV_lambda > 0
    if cfg.hparams.TV_lambda > 0:
        score = TV_along_time(output)
        loss4 = score
    else:
        loss4 = 0

    # sum up loss terms with weighting factors
    loss = cfg.hparams.L1_lambda * loss1 + cfg.hparams.L2_lambda * loss2 + \
        cfg.hparams.L3_lambda * loss3 + cfg.hparams.TV_lambda * loss4

    # print loss terms wo. weighting
    print(f'{mode}### L1: ', loss1, 'L2: ', loss2, 'L3: ', loss3, 'L4: ', loss4)
    if loss == 0:
        loss_per_batch.append(loss)
    else:
        loss_per_batch.append(loss.item())

    # apply optimizing step in training mode
    if mode == 'training':
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # every logging_epoch_freq epochs, grasp, and ouput shall be also returned
    if mode != 'testing' and batch == cfg.hparams.logging_epoch_freq:
        loss_per_batch, \
            grasp.detach().cpu().numpy(), output.detach().cpu().numpy()

        return loss_per_batch, grasp.detach().cpu().numpy(), output.detach().cpu().numpy()
    # in test mode, always return input (rnufft, coilwise_rnufft), reference (grasp), and output
    # exception if time_measurement is True
    elif mode == 'testing':
        if time_measurement is True:
            return loss_per_batch, None, None, \
            None, output.detach()
        else:
            return loss_per_batch, rnufft.detach(), coilwise_rnufft.detach(), \
                grasp.detach(), output.detach()
    # for training and validation epochs !=logging_epoch_freq, just return loss to be fast
    else:
        return loss_per_batch, None, None


def create_mask(grasp_mean: np.array):
    """
    Rule-based method to extract body area (binary mask) from time-averaged grasp image. 

    :param grasp_mean: time-averaged grasp slice

    :return: binary mask
    """
    mask = minmax_norm_0_1(grasp_mean)>0.05
    background_large = np.zeros((256, 260))
    background = ~mask
    # extent images left-rigth to better detect the bone holes
    background_large[:, 2:-2] = background
    # Remove background regions not connected to the border
    bg_no_border = clear_border(background_large)
    # bring to original size
    bg_no_border = bg_no_border[:, 2:-2]
    # Create a new mask with only inner holes filled
    inner_holes = background ^ bg_no_border.astype(bool) # holes inside object
    # Invert
    mask = ~inner_holes
    # Morphology operations
    selem5 = disk(radius=5) 
    mask = skimage.morphology.closing(mask, np.ones((5,5)))
    mask = skimage.morphology.opening(mask, np.ones((5,5)))
    mask = skimage.morphology.remove_small_holes(mask, 500)
    mask = skimage.morphology.remove_small_objects(mask, 500)
    mask = skimage.morphology.closing(mask, selem5)

    return mask