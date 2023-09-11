import yaml
import argparse
import torch
import cv2
import random
import os
import shutil
import numpy as np
from tqdm import tqdm
from model.vae import get_model
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_loader import MnistDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools.inference import visualize_latent_space

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer, crtierion, config):
    r"""
    Method to run the training for one epoch.
    :param epoch_idx: iteration number of current epoch
    :param model: VAE model
    :param mnist_loader: Data loder for mnist
    :param optimizer: optimzier to be used taken from config
    :param crtierion: For computing the loss
    :param config: configuration for the current run
    :return:
    """
    recon_losses = []
    kl_losses = []
    losses = []
    # We ignore the label for VAE
    for im, label in tqdm(mnist_loader):
        im = im.float().to(device)
        label = label.long().to(device)
        optimizer.zero_grad()
        output = model(im, label)
        mean = output['mean']
        std, log_variance = None, None
        if config['model_params']['log_variance']:
            log_variance = output['log_variance']
        else:
            std = output['std']
        generated_im = output['image']
        if config['train_params']['save_training_image']:
            cv2.imwrite('input.jpeg', (255 * (im.detach() + 1) / 2).cpu().numpy()[0, 0])
            cv2.imwrite('output.jpeg', (255 * (generated_im.detach() + 1) / 2).cpu().numpy()[0, 0])
        
        if config['model_params']['log_variance']:
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_variance) + mean ** 2 - 1 - log_variance, dim=-1))
        else:
            kl_loss = torch.mean(0.5 * torch.sum(std ** 2 + mean ** 2 - 1 - torch.log(std ** 2), dim=-1))
        recon_loss = crtierion(generated_im, im)
        loss = recon_loss + config['train_params']['kl_weight'] * kl_loss
        recon_losses.append(recon_loss.item())
        losses.append(loss.item())
        kl_losses.append(kl_loss.item())
        loss.backward()
        optimizer.step()
    print('Finished epoch: {} | Recon Loss : {:.4f} | KL Loss : {:.4f}'.format(epoch_idx + 1,
                                                                               np.mean(recon_losses),
                                                                               np.mean(kl_losses)))
    return np.mean(losses)


def train(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #######################################
    
    ######## Set the desired seed value #######
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    #######################################
    
    # Create the model and dataset
    model = get_model(config).to(device)
    mnist = MnistDataset('train', config['train_params']['train_path'])
    mnist_test = MnistDataset('test', config['train_params']['test_path'])
    mnist_loader = DataLoader(mnist, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    mnist_test_loader = DataLoader(mnist_test, batch_size=config['train_params']['batch_size'], shuffle=False,
                                   num_workers=0)
    num_epochs = config['train_params']['epochs']
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    criterion = {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }.get(config['train_params']['crit'])
    
    # Deleting old outputs for this task
    # Create output directories
    if os.path.exists(config['train_params']['task_name']):
        shutil.rmtree(config['train_params']['task_name'])
    os.mkdir(config['train_params']['task_name'])
    os.mkdir(os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir']))
    
    best_loss = np.inf
    latent_im_path = os.path.join(config['train_params']['task_name'],
                                  config['train_params']['output_train_dir'],
                                  'latent_epoch_{}.jpeg')
    with torch.no_grad():
        model.eval()
        visualize_latent_space(config, model, mnist_test_loader, save_fig_path=latent_im_path.format(0))
        model.train()
    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer, criterion, config)
        if config['train_params']['save_latent_plot']:
            model.eval()
            with torch.no_grad():
                print('Generating latent plot on test set')
                visualize_latent_space(config, model, mnist_test_loader,
                                       save_fig_path=latent_im_path.format(epoch_idx + 1))
            model.train()
        scheduler.step(mean_loss)
        # Simply update checkpoint if found better version
        if mean_loss < best_loss:
            print('Improved Loss to {:.4f} .... Saving Model'.format(mean_loss))
            torch.save(model.state_dict(), os.path.join(config['train_params']['task_name'],
                                                        config['train_params']['ckpt_name']))
            best_loss = mean_loss
        else:
            print('No Loss Improvement')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for conditional vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/vae_kl.yaml', type=str)
    args = parser.parse_args()
    train(args)