import pickle
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MyContrastTransform:
    """Rotate by one of the given angles."""

    def __init__(self, adjust_contrast):
        self.adjust_contrast = adjust_contrast

    def __call__(self, x):
        max_factor = torch.max(x)
        min_factor = torch.min(x)
        x = (x - min_factor) / (max_factor - min_factor)
        x = TF.adjust_sharpness(x, self.adjust_contrast)
        x =  x * (max_factor - min_factor) + min_factor

        return x
    

class MySharpnessTransform:
    """Rotate by one of the given angles."""

    def __init__(self, adjust_sharpness):
        self.adjust_sharpness = adjust_sharpness

    def __call__(self, x):
        max_factor = torch.max(x)
        min_factor = torch.min(x)

        x = (x - min_factor) / (max_factor - min_factor)
        x = TF.adjust_sharpness(x, self.adjust_sharpness)
        x =  x * (max_factor - min_factor) + min_factor

        return x


def self_apply_augmentation(image):
    """
    TODO
    """
    params = {}
    # Random Horizontal Flip
    if torch.rand(1) > 0.5:
        params['hflip'] = True
        flipped_image = TF.hflip(image)
    else:
        params['hflip'] = False
        flipped_image = image

    # Random Vertical Flip
    if torch.rand(1) > 0.5:
        params['vflip'] = True
        image = TF.vflip(flipped_image)
    else:
        params['vflip'] = False
        image = flipped_image

    return image, params


def apply_params(image, params):
    # Random Horizontal Flip
    if 'hflip' in params.keys():
        if params['hflip'] is True:
            flipped_image = TF.hflip(image)
        else:
            flipped_image = image
    else:
        flipped_image = image


    # Random Vertical Flip
    if 'vflip' in params.keys():
        if params['vflip'] is True:
            image = TF.vflip(flipped_image)
        else:
            image = flipped_image
    else:
        image = flipped_image

    if 'contrast' in params.keys():
        contrast = MyContrastTransform(params['contrast'])
        image = contrast(image)
    else:
        image = image

    if 'crop_params' in params.keys():
        i, j, h, w = params['crop_params']
        cropped_image = TF.resized_crop(image, i, j, h, w, (256, 256))
    else:
        cropped_image = image

    if 'sharpness' in params.keys():
        sharpness = MySharpnessTransform(params['sharpness'])
        image = sharpness(cropped_image)
    else:
        image = cropped_image
    return image


class MovienetDataset_withPriors_shifted(Dataset):
    def __init__(self, data_dir, mode, cfg):
        self.data_dir = Path(data_dir)

        self.data_slices = [x for x in Path(self.data_dir / mode).iterdir() if x.is_file()]
        self.mode = mode

        self.modelname = cfg.hparams.modelname
        self.GRASP_complex = cfg.hparams.GRASP_complex

        self.splits_path = Path(self.data_dir / f'{mode}.pkl')
        file = open(self.splits_path, 'rb')
        self.split_list = pickle.load(file)

        self.julia_reco = cfg.hparams.julia_reco
        self.do_transform = cfg.hparams.do_transform
        self.do_aug = cfg.hparams.do_aug
        self.tf = cfg.hparams.timeframes
        self.coils = cfg.hparams.coils
        self.padder = cfg.hparams.padder
        self.start_padding = cfg.hparams.start_padding
        self.img_size = cfg.hparams.img_size
        self.complex_channels = 1
        self.Ninputchannels = cfg.hparams.Ninputchannels
        self.modelname = cfg.hparams.modelname

        self.meta_path =  Path(f'{cfg.hparams.meta_dir}')
        self.df = pd.read_excel(self.meta_path, sheet_name='simple', index_col=0)


    def create_pad_indices(self):
        self.pad_indices = torch.tile(torch.tensor(range(self.tf))[None, None, :], (self.img_size, self.img_size, 1))
        self.pad_indices_coilwise = torch.tile(
            self.pad_indices[None, :, :, :],
            (self.coils * self.complex_channels, 1, 1, 1))

    def __len__(self):
        # Return the number of input samples
        return len(self.split_list)

    def __getitem__(self, idx):
        file_path = Path(self.data_dir / self.mode / f'{self.split_list[idx]}.npz')
        studyname = eval(file_path.name.split("_")[0][-4:])
        self.ana_frames = int(np.ceil(np.ceil(self.img_size * np.pi/2) / self.df.loc[self.df.index==studyname, 'roundedspokespertf'].values[0]))
        print('ana_frames is: ', self.ana_frames)

        rnufft, coilwise_rnufft, grasp, ana_prior, name = self.load_image(file_path)

        return rnufft, coilwise_rnufft, grasp, ana_prior, name, name, self.ana_frames


    def shift(self, rnufft, coilwise_rnufft, grasp, ana_data):
        padding = (self.padder, self.padder)

        rnufft_padded = torch.nn.functional.pad(rnufft, padding, "reflect")

        coilwise_rnufft_reshape = coilwise_rnufft.reshape(-1, self.tf)
        coilwise_rnufft_reshape_padded = torch.nn.functional.pad(coilwise_rnufft_reshape, padding, "reflect")
        coilwise_rnufft_padded = coilwise_rnufft_reshape_padded.reshape(
            self.coils * self.complex_channels, self.img_size, self.img_size, -1)
        grasp_padded = torch.nn.functional.pad(grasp, padding, "reflect")
        ana_data_padded = torch.nn.functional.pad(ana_data, padding, "reflect")

        pad_randn = torch.clip(
            torch.rand(1) * int(self.padder + self.start_padding),
            self.start_padding,
            int(self.padder * 2 - self.start_padding)).round(decimals=0).int()

        rnufft_shifted = torch.gather(rnufft_padded, 2, self.pad_indices + pad_randn)
        grasp_shifted = torch.gather(grasp_padded, 2, self.pad_indices + pad_randn)
        ana_data_shifted = torch.gather(ana_data_padded, 2, self.pad_indices + pad_randn)
        coilwise_rnufft_shifted = torch.gather(coilwise_rnufft_padded, 3, self.pad_indices_coilwise + pad_randn)

        return rnufft_shifted, coilwise_rnufft_shifted, grasp_shifted, ana_data_shifted

    def transform_and_reshape_data(self, file_path, rnufft, coilwise_rnufft, grasp, ana_data):
        name = Path(file_path.name).stem

        if self.do_transform is True:
            rnufft, coilwise_rnufft, grasp, ana_data = self.shift(rnufft, coilwise_rnufft, grasp, ana_data)

        rnufft = rnufft.permute(2, 0, 1).type(torch.float32)
        coilwise_rnufft = coilwise_rnufft.permute(0, 3, 1, 2)
        grasp = grasp.permute(2, 0, 1).type(torch.float32)

        ana_prior = torch.zeros((self.img_size, self.img_size, self.tf))
        for i in range(self.tf):
            ana_data_padded = torch.nn.functional.pad(ana_data, (0, self.ana_frames - 1), "circular")
            ana_prior[:, :, i] = torch.abs(torch.mean(ana_data_padded[:, :, i:i+self.ana_frames], dim=2))
        ana_prior /= ana_prior.max()
        ana_prior = ana_prior.permute(2, 0, 1).type(torch.float32)

        return rnufft, coilwise_rnufft, grasp, ana_prior, name
    
    def load_image(self, file_path):
        # Load binary data and convert it to a NumPy array
        image_data = np.load(file_path)

        if self.julia_reco is True:
            rnufft = torch.from_numpy(image_data['rnufft']['real'] + 1j * image_data['rnufft']['imag']).permute(2, 1, 0)
            ana_data = rnufft
            rnufft = torch.abs(rnufft)
        else:
            image_data_abs_path = Path(str(file_path).replace(
                "ProcessedData_lambda02_complex", "ProcessedData_lambda02"))
            image_data_abs = np.load(image_data_abs_path)
            rnufft = torch.from_numpy(image_data['rnufft'])

            ana_data = rnufft[:, :, :, 0] + 1j * rnufft[:, :, :, 1]
        rnufft /= rnufft.max()

        if self.julia_reco is True:
            coilwise_rnufft = torch.abs(
                torch.from_numpy(
                    image_data[
                        'coilwise_rnufft']['real']) + 1j * torch.from_numpy(image_data['coilwise_rnufft']['imag']))
            coilwise_rnufft = torch.permute(coilwise_rnufft, (1, 3, 2, 0))
        else:
            coilwise_rnufft = torch.from_numpy(image_data_abs['coilwise_rnufft'])
            coilwise_rnufft = torch.permute(coilwise_rnufft, (2, 0, 1, 3))
        coilwise_rnufft /= coilwise_rnufft.max()

        if self.julia_reco is True:
            if self.GRASP_complex is False:
                grasp = torch.from_numpy(image_data['grasp']).permute(2, 1, 0)
            else:
                grasp = torch.from_numpy(
                    image_data[
                        'grasp']['real'] + 1j * image_data['grasp']['imag']).permute(2, 1, 0)
                grasp = torch.abs(grasp)
        else:
            grasp = torch.from_numpy(image_data_abs['grasp'])
        grasp /= grasp.max()

        rnufft, coilwise_rnufft, grasp, ana_data, name = self.transform_and_reshape_data(
            file_path, rnufft, coilwise_rnufft, grasp, ana_data)
        if self.modelname == "Movienet":
            coilwise_rnufft = coilwise_rnufft.reshape(-1, self.img_size, self.img_size).type(torch.float32)

        return rnufft, coilwise_rnufft, grasp, ana_data, name


class MovienetDataset_withPriors_shifted_complex_norm_90(MovienetDataset_withPriors_shifted):
    def __init__(self, data_dir, mode, cfg):
        super().__init__(data_dir, mode, cfg)

        self.complex_channels = 2
        self.create_pad_indices()

        self.norm_dict_path = Path(f'{self.data_dir}/value90_value_dict_new.npz')

        self.dict_file = np.load(self.norm_dict_path, allow_pickle=True)
        if self.mode == 'training':
            self.norm_dict = self.dict_file['train_max_value_dict'].item()
        elif self.mode == 'validation':
            self.norm_dict = self.dict_file['val_max_value_dict'].item()
        elif self.mode == 'test':
            self.norm_dict = self.dict_file['test_max_value_dict'].item()
        else:
            raise NotImplementedError('self.mode must be training, validation or test')

    def transform_and_reshape_data(self, file_path, rnufft, coilwise_rnufft, grasp, ana_data, norm_value):
        if self.do_transform is True:
            rnufft, coilwise_rnufft, grasp, ana_data = self.shift(rnufft, coilwise_rnufft, grasp, ana_data)
    
        rnufft_complex = torch.zeros(2, *ana_data.shape)
        rnufft_complex[0] = ana_data.real
        rnufft_complex[1] = ana_data.imag
        rnufft_complex = rnufft_complex.permute(0, 3, 1, 2).type(torch.float32)

        rnufft = rnufft.permute(2, 0, 1).type(torch.float32)
        coilwise_rnufft = coilwise_rnufft.permute(0, 3, 1, 2)
        grasp = grasp.permute(2, 0, 1).type(torch.float32)

        ana_prior = torch.zeros((self.img_size, self.img_size, self.tf))
        for i in range(self.tf):
            ana_data_padded = torch.nn.functional.pad(ana_data, (0, self.ana_frames - 1), "circular")
            ana_prior[:, :, i] = torch.abs(torch.mean(ana_data_padded[:, :, i:i+self.ana_frames], dim=2))
        ana_prior /= norm_value #* self.spokes_per_tf
        ana_prior = ana_prior.permute(2, 0, 1).type(torch.float32)

            
        if self.do_aug is True:
            grasp_permuted = grasp[:, None, ...]
            grasp_permuted, params = self_apply_augmentation(grasp_permuted)

            ana_data_permuted = ana_prior[:, None, ...]
            ana_data_permuted= apply_params(ana_data_permuted, params)

            rnufft_permuted = rnufft[:, None, ...]
            rnufft_permuted= apply_params(rnufft_permuted, params)

            coilwise_rnufft_permuted = coilwise_rnufft.reshape(-1, 1, self.img_size, self.img_size)
            
            if self.modelname == "Movienet":
                coilwise_rnufft_permuted = apply_params(coilwise_rnufft_permuted, params).reshape(self.coils*2, self.tf, 1, self.img_size, self.img_size)
            else:
                coilwise_rnufft_permuted = apply_params(coilwise_rnufft_permuted, params).reshape(self.Ninputchannels, self.tf, 1, self.img_size, self.img_size)
            
            grasp = grasp_permuted[:, 0, ...]
            ana_prior = ana_data_permuted[:, 0, ...]
            rnufft = rnufft_permuted[:, 0, ...]
            coilwise_rnufft = coilwise_rnufft_permuted[:, :, 0, ...]

        return rnufft, coilwise_rnufft, grasp, ana_prior, rnufft_complex
    
    def __getitem__(self, idx):
        file_path = Path(self.data_dir / self.mode / f'{self.split_list[idx]}.npz')
        match = re.search(r'(?:Exam|Study)(\d+)', file_path.stem)

        if match:
            studyname = int(match.group(1))
        else: 
            raise NotImplementedError(f'The filename can not be extracted from {file_path}')
        self.ana_frames = int(np.ceil(np.ceil(self.img_size * np.pi/2) / self.df.loc[self.df.index==studyname, 'roundedspokespertf'].values[0]))
        # print('ana_frames is: ', self.ana_frames)
        rnufft, coilwise_rnufft, grasp, ana_prior, rnufft_complex, name = self.load_image(file_path)

        return rnufft, coilwise_rnufft, grasp, ana_prior, rnufft_complex, name, self.ana_frames

    def load_image(self, file_path):
        # Load binary data and convert it to a NumPy array
        image_data = np.load(file_path)
        name = Path(file_path.name).stem
        name1 = name.split('_')[0]
        name2 = name.split('_')[-1]
        norm_value = [v for k, v in self.norm_dict.items() if k.startswith(name1) and k.endswith(name2)][0]


        if self.julia_reco is True:
            rnufft = torch.from_numpy(image_data['rnufft']['real'] + 1j * image_data['rnufft']['imag']).permute(2, 1, 0)
            ana_data = rnufft
            rnufft = torch.abs(rnufft)
        else:
            rnufft = torch.from_numpy(image_data['rnufft'])
            ana_data = rnufft[:, :, :, 0] + 1j * rnufft[:, :, :, 1]
            rnufft = torch.abs(rnufft[:, :, :, 0] + 1j * rnufft[:, :, :, 1])
        rnufft /= norm_value

        if self.julia_reco is True:
            coilwise_rnufft = torch.cat(
                [torch.from_numpy(image_data['coilwise_rnufft']['real'][None, ...]),
                 torch.from_numpy(image_data['coilwise_rnufft']['imag'][None, ...])],
                dim=0)
            coilwise_rnufft = torch.permute(coilwise_rnufft, (0, 2, 4, 3, 1)).reshape(-1, self.img_size, self.img_size, self.tf)
        else:
            coilwise_rnufft = torch.from_numpy(image_data['coilwise_rnufft'])
            coilwise_rnufft = torch.permute(coilwise_rnufft, (4, 2, 0, 1, 3)).reshape(-1, self.img_size, self.img_size, self.tf)
        coilwise_rnufft /= norm_value

        if self.julia_reco is True:
            if self.GRASP_complex is False:
                grasp = torch.from_numpy(image_data['grasp']).permute(2, 1, 0)
            else:
                grasp = torch.from_numpy(
                    image_data['grasp']['real'] + 1j * image_data['grasp']['imag']).permute(2, 1, 0)
                grasp = torch.abs(grasp)
        else:
            grasp = torch.from_numpy(image_data['grasp'])
            grasp = torch.abs(grasp[:, :, :, 0] + 1j * grasp[:, :, :, 1])
        grasp /= norm_value 
        grasp = grasp * 10000

        rnufft, coilwise_rnufft, grasp, ana_data, rnufft_complex = self.transform_and_reshape_data(
            file_path, rnufft, coilwise_rnufft, grasp, ana_data, norm_value)
        
        rnufft_complex /= norm_value

        if self.modelname == "Movienet":
            coilwise_rnufft = torch.abs(coilwise_rnufft[:8] + 1j * coilwise_rnufft[8:])
            coilwise_rnufft = coilwise_rnufft.reshape(-1, self.img_size, self.img_size).type(torch.float32)
        return rnufft, coilwise_rnufft, grasp, ana_data, rnufft_complex, name