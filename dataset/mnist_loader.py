import os
import cv2
import glob
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


r"""
Simple Dataloader for mnist.
This assumes the images are already extracted from csv.
For extracting images from csv one can use utils/extract_mnist_images
"""
class MnistDataset(Dataset):
    def __init__(self, split, im_path, im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)
        
    def load_images(self, im_path):
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = cv2.imread(self.images[index], 0)
        label = self.labels[index]
        # Convert to 0 to 255 into -1 to 1
        im = 2*(im / 255) - 1
        # Convert H,W,C into 1,C,H,W
        im_tensor = torch.from_numpy(im)[None,:]
        return im_tensor, torch.as_tensor(label)
        


if __name__ == '__main__':
    mnist = MnistDataset('test', 'data/test/images')
    mnist_loader = DataLoader(mnist, batch_size=16, shuffle=True, num_workers=0)
    for im, label in mnist_loader:
        print('Image dimension', im.shape)
        print('Label dimension: {}'.format(label.shape))
        break
    
    
    