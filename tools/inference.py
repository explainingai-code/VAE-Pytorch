import shutil
import yaml
import argparse
import torch
import cv2
import os
from tqdm import tqdm
import torchvision
from model.vae import get_model
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_loader import MnistDataset
from torchvision.utils import make_grid
from einops import rearrange
import pickle
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reconstruct(config, model, dataset, num_images=100):
    r"""
    Randomly sample points from the dataset and visualize image and its reconstruction
    :param config: Config file used to create the model
    :param model: Trained model
    :param dataset: Mnist dataset(not the data loader)
    :param num_images: NUmber of images to visualize
    :return:
    """
    print('Generating reconstructions')
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    if not os.path.exists(
            os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir'])):
        os.mkdir(os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir']))
    
    idxs = torch.randint(0, len(dataset) - 1, (num_images,))
    ims = torch.cat([dataset[idx][0][None, :] for idx in idxs]).float()
    labels = torch.cat([dataset[idx][1][None] for idx in idxs]).long()
    
    output = model(ims, labels)
    generated_im = output['image']
    
    # Dataset generates -1 to 1 we convert it to 0-1
    ims = (ims + 1) / 2
    # For reconstruction, we specifically flip it(white digit on black background -> black digit on white background)
    # for easier visualization
    generated_im = 1 - (generated_im + 1) / 2
    out = torch.hstack([ims, generated_im])
    output = rearrange(out, 'b c h w -> b () h (c w)')
    grid = make_grid(output, nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(os.path.join(config['train_params']['task_name'],
                          config['train_params']['output_train_dir'],
                          'reconstruction.png'))


def visualize_latent_space(config, model, data_loader, save_fig_path):
    r"""
    Method to visualize the latent dimension by simply plotting the means for each of the images
    :param config: Config file used to create the model
    :param model:
    :param data_loader:
    :param save_fig_path: Path where the latent space image will be saved
    :return:
    """
    labels = []
    means = []
    
    for im, label in tqdm(data_loader):
        im = im.float().to(device)
        label = label.long().to(device)
        output = model(im, label)
        labels.append(label)
        mean = output['mean']
        means.append(mean)
    
    labels = torch.cat(labels, dim=0).reshape(-1)
    means = torch.cat(means, dim=0)
    if model.latent_dim != 2:
        print('Latent dimension > 2 and hence projecting')
        U, _, V = torch.pca_lowrank(means, center=True, niter=2)
        proj_means = torch.matmul(means, V[:, :2])
        if not os.path.exists(config['train_params']['task_name']):
            os.mkdir(config['train_params']['task_name'])
        pickle.dump(V, open('{}/pca_matrix.pkl'.format(config['train_params']['task_name']), 'wb'))
        means = proj_means
    
    fig, ax = plt.subplots()
    for num in range(10):
        idxs = torch.where(labels == num)[0]
        ax.scatter(means[idxs, 0].cpu().numpy(), means[idxs, 1].cpu().numpy(), s=10, label=str(num),
                   alpha=1.0, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.savefig(save_fig_path)


def visualize_interpolation(config, model, dataset, interpolation_steps=500, save_dir='interp'):
    r"""
        We randomly fetch two points and linearly interpolate between them.
        We only use the mean values for interpolation
    :param config:
    :param model:
    :param dataset:
    :param interpolation_steps: We will interpolate these many points between start and end
    :param save_dir:
    :return:
    """
    # if model.config['conditional']:
    #     print('Interpolation is only for non conditional model. Check README for details. Skipping...')
    #     return
    print('Interpolating between images')
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    if not os.path.exists(
            os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir'])):
        os.mkdir(os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir']))
    
    if os.path.exists(os.path.join(config['train_params']['task_name'],
                                   config['train_params']['output_train_dir'],
                                   save_dir)):
        shutil.rmtree(os.path.join(config['train_params']['task_name'],
                                   config['train_params']['output_train_dir'],
                                   save_dir))
    os.mkdir(os.path.join(config['train_params']['task_name'],
                          config['train_params']['output_train_dir'],
                          save_dir))
    
    idxs = torch.randint(0, len(dataset)-1, (2,))
    if model.config['conditional']:
        label_val = torch.randint(0, 9, (1,))
        labels = (torch.ones((1,)).long().to(device) * label_val).repeat((2))
    else:
        labels = None
    ims = torch.cat([dataset[idx][0][None, :] for idx in idxs]).float()
    means = model(ims, labels)['mean']
    factors = torch.linspace(0, 1.0, steps=interpolation_steps)
    means_start = means[0]
    means_end = means[1]
    if model.config['conditional']:
        label_val = torch.randint(0, 9, (1,))
        labels = (torch.ones((1,)).long().to(device) * label_val).repeat((interpolation_steps))
    else:
        labels = None
    means = factors[:, None] * means_end[None, :] + (1 - factors[:, None]) * means_start[None, :]
    out = model.generate(means, labels)
    for idx in tqdm(range(out.shape[0])):
        # Convert generated output from -1 to 1 range to 0-255
        im = 255 * (out[idx, 0] + 1) / 2
        cv2.imwrite('{}/{}.png'.format(os.path.join(config['train_params']['task_name'],
                                                    config['train_params']['output_train_dir'],
                                                    save_dir), idx), im.cpu().numpy())


def visualize_manifold(config, model):
    print('Generating the manifold')
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    if not os.path.exists(
            os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir'])):
        os.mkdir(os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir']))
    
    # For conditional model we can generate all numbers for all points in the space.
    # This because the condition introduces the variance even if the point (z) is the same
    # But for non-conditional only one output is possible for one z hence progress bar range is 1
    if model.config['conditional']:
        pbar_range = model.num_classes
    else:
        pbar_range = 1
    for label_val in tqdm(range(pbar_range)):
        num_images = 900
        # For values below use the latent images to get a sense of what ranges we need to plot
        xs = torch.linspace(-10, 10, 30)
        ys = torch.linspace(-10, 10, 30)
        
        xs, ys = torch.meshgrid([xs, ys])
        xs = xs.reshape(-1, 1)
        ys = ys.reshape(-1, 1)
        zs = torch.cat([xs, ys], dim=-1)
        if model.latent_dim != 2:
            if not os.path.exists(os.path.join(config['train_params']['task_name'], 'pca_matrix.pkl')):
                print('Latent dimension > 2 but no pca info available. '
                      'Call visualize_latent_space first. Skipping visualize_manifold')
            else:
                V = pickle.load(open(os.path.join(config['train_params']['task_name'], 'pca_matrix.pkl'), 'rb'))
                reconstruct_means = torch.matmul(zs, V[:, :2].T)
                zs = reconstruct_means
        label = (torch.ones((1,)).long().to(device) * label_val).repeat((num_images))
        generated_ims = model.sample(label, num_images, z=zs)
        generated_ims = ((generated_ims + 1) / 2)
        grid = make_grid(generated_ims, nrow=30)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save(os.path.join(config['train_params']['task_name'],
                              config['train_params']['output_train_dir'],
                              'manifold_{}.png'.format(label_val) if model.config['conditional'] else 'manifold.png'))


def inference(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                  config['train_params']['ckpt_name']), map_location='cpu'))
    model.eval()
    mnist = MnistDataset('test', 'data/test/images')
    mnist_loader = DataLoader(mnist, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    
    with torch.no_grad():
        latent_im_path = os.path.join(config['train_params']['task_name'],
                                      config['train_params']['output_train_dir'],
                                      'latent_inference.jpeg')
        visualize_latent_space(config, model, mnist_loader, latent_im_path)
        visualize_interpolation(config, model, mnist)
        reconstruct(config, model, mnist)
        visualize_manifold(config, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/vae_kl.yaml', type=str)
    args = parser.parse_args()
    inference(args)