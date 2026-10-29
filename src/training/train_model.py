import hydra
import os
import time
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.utilities.helper import manual_seed, seed_worker, optimization_loop, get_device, get_model, \
    get_criterions, load_model, get_dataset, save_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm

# restrict number of threads
nthreads = 5
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)

# change to working directory
os.chdir(os.getcwd())

torch.set_num_threads(nthreads)
print('number of threads: ', torch.get_num_threads())


@hydra.main(config_path=str(Path(Path.cwd().parents[1] / 'conf')), config_name="config_example", version_base="1.3")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    #######################################################################

    # set path where to load pretrained model / save model
    model_path = Path(Path(cfg.hparams.data_dir).parent / 'Networks')
    # set path where to log cfg and outputs
    log_dir = Path(Path(cfg.hparams.data_dir).parent / 'logs' / cfg.hparams.experiment_name)
    # create paths if not existent
    Path.mkdir(model_path, exist_ok=True)
    Path.mkdir(log_dir, exist_ok=True)

    # set slice to show in logged outputs
    tp_to_show = 5

    # get device, model, averaging kernel, low-rank model, and criterions
    device = get_device(cfg)
    model = get_model(cfg, device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    criterion1, criterion2 = get_criterions(method1='MSE', reduction1='mean', method2='MSE', reduction2='mean')

    # in case, load pretrained model
    if cfg.hparams.load_model is True:
        model = load_model(cfg, model, model_path, 'train')
    if cfg.hparams.cuda == "parallel":
        model = torch.nn.DataParallel(model)

    #######################################################################
    # set seeds to make experiments reproducible
    manual_seed(1234)
    # not possible
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    #######################################################################
    # get dataset
    MovieDataset = get_dataset(cfg)
    # define datasets
    training_dataset = MovieDataset(Path(cfg.hparams.data_dir), 'training', cfg)
    validation_dataset = MovieDataset(Path(cfg.hparams.data_dir), 'validation', cfg)

    # define dataloaders with seed
    g = torch.Generator()
    g.manual_seed(1234)
    train_loader = DataLoader(
        training_dataset,
        batch_size=cfg.hparams.batch_size,
        shuffle=True,
        num_workers=0, #3
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=True)
    val_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.hparams.batch_size,
        shuffle=True,
        num_workers=0, #3
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=True)

    # set optimizer, scheduler, and loss functions
    optimizer = optim.Adam(model.parameters(), lr=cfg.hparams.learning_rate, weight_decay=cfg.hparams.WD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=5e-6, patience=cfg.hparams.patience)

    # prepare summary writer and loss lists
    writer = SummaryWriter(log_dir=log_dir)
    train_loss_list = []
    val_loss_list = []

    #################################################################################################################
    # loop over data for n_epochs
    for epoch in range(cfg.hparams.n_epochs):
        t0 = time.time()
        #######################################################################
        # Training
        #######################################################################
        model.train()
        batch = -1

        # loop over training data
        for rnufft, coilwise_rnufft, grasp, ana_prior, rnufft_complex, train_name, ana_frames in tqdm(train_loader):
            batch += 1
            train_loss_per_batch = []

            # this step applies the model, calculates the loss, backpropagates the loss, and performs an optimizer step
            train_loss_per_batch, grasp, output = optimization_loop(
                    cfg,
                    'training',
                    device,
                    model,
                    optimizer,
                    criterion1,
                    criterion2,
                    epoch,
                    batch,
                    None,
                    None,
                    tp_to_show,
                    train_loss_per_batch,
                    coilwise_rnufft,
                    grasp,
                    rnufft,
                    ana_prior,
                    ana_frames,
                    None)
            # batch number 10 is used for logging
            if batch == 10:
                output = output / np.percentile(output, 99.99)
                grasp = grasp / np.percentile(grasp, 99.99)
                writer.add_image('train_image_tp12', output[0, tp_to_show, :, :], epoch, dataformats='HW')
                writer.add_image('ref_train_image_tp12', grasp[0, tp_to_show, :, :], epoch, dataformats='HW')

        # for each batch, the mean loss is added to the loss list
        train_loss = np.mean(train_loss_per_batch)
        train_loss_list.append(train_loss)

        #######################################################################
        # Validation
        #######################################################################
        model.eval()
        batch = -1
        # loop over validation data
        for rnufft, coilwise_rnufft, grasp, ana_prior, rnufft_complex, val_name, ana_frames in tqdm(val_loader):
            batch += 1
            val_loss_per_batch = []
            # this step applies the model, calculates the loss
            val_loss_per_batch, grasp, output = optimization_loop(
                    cfg,
                    'validation',
                    device,
                    model,
                    optimizer,
                    criterion1,
                    criterion2,
                    epoch,
                    batch,
                    None,
                    None,
                    tp_to_show,
                    val_loss_per_batch,
                    coilwise_rnufft,
                    grasp,
                    rnufft,
                    ana_prior,
                    ana_frames,
                    None)
            # batch number 10 is used for logging
            if batch == 10:
                output = output / np.percentile(output, 99.99)
                grasp = grasp / np.percentile(grasp, 99.99)
                writer.add_image('val_image_tp12', output[0, tp_to_show, :, :], epoch, dataformats='HW')
                writer.add_image('ref_val_image_tp12', grasp[0, tp_to_show, :, :], epoch, dataformats='HW')

        # for each batch, the mean loss is added to the loss list
        val_loss = np.mean(val_loss_per_batch)
        val_loss_list.append(val_loss)

        # logging of train and validation losses
        writer.add_scalars(
            'Loss', {
                'train': train_loss,
                'val': val_loss},
            epoch)
        # logging of learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        writer.add_scalar('Learning_rate', lr, epoch)

        # if epoch is larger than scheduler_start_epoch, the scheduelr is applied
        if epoch > cfg.hparams.scheduler_start_epoch:
            scheduler.step(val_loss) # TODO

        # print losses and time needed for epoch
        t1 = time.time()
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTime for one epoch (sec): {:.6f}'.format(
            epoch, train_loss, val_loss, t1-t0))

        # save the best model
        save_model(cfg, val_loss_list, val_loss, model, model_path)

    # at the end flush and close the writer
    writer.flush()
    writer.close()


if __name__ == "__main__":
    train()
