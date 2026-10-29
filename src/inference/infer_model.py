import hydra
import os
import time
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from scipy.io import savemat
from src.utilities.helper import seed_worker, manual_seed, get_device, get_model, get_dataset, get_criterions, load_model
from src.utilities.helper import optimization_loop
from torch.utils.data import DataLoader
from thop import profile

#######################################################################
os.chdir(os.getcwd())


@hydra.main(config_path=str(Path(Path.cwd().parents[1] / 'conf')), config_name="config_test", version_base="1.3")
def test(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # overwrite config to ensure they are set correctly
    tp_to_show = 12
    time_measurement = False
    get_FLOPS = False

    model_path = Path(Path(cfg.hparams.data_dir).parent / 'Networks')
    results_path = Path(Path(cfg.hparams.results_dir) / 'Networks_results' / cfg.hparams.experiment_name)
    Path.mkdir(Path(cfg.hparams.results_dir), exist_ok=True)
    Path.mkdir(results_path, exist_ok=True)

    #######################################################################
    # reproducibility
    manual_seed(1234)
    # not possible torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    #######################################################################

    # get device, model, averaging kernel, low-rank model, and criterions
    device = get_device(cfg)
    model = get_model(cfg, device)
    criterion1, criterion2 = get_criterions(method1='MSE', reduction1='mean', method2='MSE', reduction2='mean')

    # get dataset
    MovieDataset = get_dataset(cfg)
    test_dataset = MovieDataset(Path(cfg.hparams.data_dir), 'test', cfg)

    # define dataloader
    g = torch.Generator()
    g.manual_seed(1234)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.hparams.batch_size,
        shuffle=False,
        num_workers=0,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=False)

    # load model
    model = load_model(cfg, model, model_path, 'eval')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    
    # loop through dataset
    test_loss_per_image = []
    epoch = 0
    batch = -1

    inference_time = []
    saving_time = []
    for rnufft, coilwise_rnufft, grasp, ana_prior, _, test_name, ana_frames in test_loader:
        
        batch += 1
        print(test_name[0])
        if batch == 0 and get_FLOPS is True:
            macs, params = profile(model, inputs=(coilwise_rnufft.to(device),))
            print(f"MACs: {macs}, Parameters: {params}")
            break

        t0 = time.time()
        test_loss_per_image, rnufft, coilwise_rnufft, \
            grasp, output = optimization_loop(
                cfg,
                'testing',
                device,
                model,
                None,
                criterion1,
                criterion2,
                epoch,
                batch,
                results_path,
                test_name,
                tp_to_show,
                test_loss_per_image,
                coilwise_rnufft,
                grasp,
                rnufft,
                ana_prior,
                ana_frames,
                None,
                time_measurement=time_measurement)

        if time_measurement is False:
            rnufft = rnufft.cpu().numpy()
            coilwise_rnufft = coilwise_rnufft.cpu().numpy()
            grasp = grasp.cpu().numpy()
            output = output.cpu().numpy()
        t1 = time.time()

        if time_measurement is False:
            for i in range(rnufft.shape[0]):
                savemat(
                    Path(results_path / f'{test_name[i]}_test_output.mat'),
                    {
                        'rnufft': rnufft[i],
                        'coilwise_rnufft': coilwise_rnufft[i],
                        'grasp': grasp[i],
                        'output': output[i],
                        }
                    )
        test_loss = np.mean(test_loss_per_image)
        print('\t Test Loss: {:.6f} \t Time for one epoch (sec): {:.6f}'.format(
                    test_loss, t1-t0))
        inference_time.append(t1-t0)
        t2 = time.time()
        print('\t Saving test took (sec): {:.6f}'.format(t2-t0))
        saving_time.append(t2-t0)

    print(f'aveage inference time per slice is {np.mean(inference_time)}')
    print(f'median inference time per slice is {np.median(inference_time)}')
    print(f'aveage saving time per slice is {np.mean(saving_time)}')

    print(f'average inference time for {len(test_dataset)} test slices is {np.mean(inference_time) * len(test_dataset)}')
    print(f'median inference time for {len(test_dataset)} test slices is {np.median(inference_time) * len(test_dataset)}')
if __name__ == "__main__":
    test()
