VAE Implementation in pytorch with visualizations
========

This repository implements a simple VAE for training on CPU on the MNIST dataset and provides ability
to visualize the latent space, entire manifold as well as visualize how numbers interpolate between each other.

The purpose of this project is to get a better understanding of VAE by playing with the different parameters
and visualizations.

## VAE Tutorial Videos
<a href="https://www.youtube.com/watch?v=1RPdu_5FCfk">
   <img alt="VAE Understanding" src="https://github.com/explainingai-code/VAE-Pytorch/assets/144267687/da55f1cb-0cdc-46ec-82db-6ed6da2cca58"
   width="300">
</a>
<a href="https://www.youtube.com/watch?v=pEsC0Vcjc7c">
   <img alt="Implementing VAE" src="https://github.com/explainingai-code/VAE-Pytorch/assets/144267687/d5590353-a9e9-44df-b99c-6b7f389b4f54"
   width="300">
</a>

## Architecture
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/a67b9a50-78fa-4876-8094-0db6db08e91c" width="700">

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/Pytorch-VAE.git```
* ```cd Pytorch-VAE```
* ```pip install -r requirements.txt```
* For running a simple fc layer backed VAE with latent dimension as 2 run ```python run_simple_vae.py```
* For playing around with VAE and running visualizations, replace tools/train_vae.py and tools/inference.py config argument with the desired one or pass that in the next set of commands
* ```python -m tools.train_vae```
* ```python -m tools.inference```

## Configurations
* ```config/vae_nokl.yaml``` - VAE with only reconstruction loss
* ```config/vae_kl.yaml``` - VAE with reconstruction and KL loss
* ```config/vae_kl_latent4.yaml``` - VAE with reconstruction and KL loss with latent dimension as 4(instead of 2)
* ```config/vae_kl_latent4_enc_channel_dec_fc_condition.yaml``` - Conditional VAE with reconstruction and KL loss with latent dimension as 4

## Data preparation
We don't use the torchvision mnist dataset to allow replacement with any other image dataset. 

For setting up the dataset:
* Download the csv files for mnist(https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
and save them under ```data```directory.
* Run ```python utils/extract_mnist_images.py``` 

Verify the data directory has the following structure:
```
Pytorch-VAE/data/train/images/{0/1/.../9}
	*.png
Pytorch-VAE/data/test/images/{0/1/.../9}
	*.png
```

## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created and ```output_train_dir``` will be created inside it.

During training the following output will be saved 
* Best Model checkpoints in ```task_name``` directory
* PCA information in pickle file in ```task_name``` directory
* 2D Latent space plotting the images of test set for each epoch in ```task_name/output_train_dir``` directory

During inference the following output will be saved
* Reconstructions for sample of test set in ```task_name/output_train_dir/reconstruction.png``` 
* Decoder output for sample of points evenly spaced across the projection of latent space on 2D in ```task_name/output_train_dir/manifold.png```
* Interpolation between two randomly sampled points in ```task_name/output_train_dir/interp``` directory


## Sample Output for VAE
Latent Visualization

<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/ca748b18-c93c-4d60-a88e-7b85ecbca48f" width="500">

Manifold

<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/b3be5973-a96d-4c44-9e27-2646bb9d35b7" width="500">


Reconstruction Images(reconstruction in black font and original in white font)

<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/5ded5421-ae28-4a97-afc0-4aa58c706648" width="500">

## Sample Output for Conditional VAE

Because we end up passing the label to the decoder, the model ends up learning the capability to generate ALL numbers from all points in the latent space. 

The model will learn to distinguish points in latent space based on if it should generate a left or right tilted digit or how thick the stroke for digit should be. 
Below one can visulize those patterns when we attempt to generate all numbers from all points.



<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/d5ef578c-2fce-46e9-bd3a-6c28ef4daffa" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/5dca94c5-0402-4048-a073-0cc84452f276" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/85f5452b-a64a-47a3-87cc-18510728b12f" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/a0361af0-000d-45dc-ab45-e28aa535cf0f" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/d4d5cf96-30ab-4b75-bf37-1773a0ca236d" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/088bc4df-9a52-42e5-a19d-6ee0cb537785" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/9d2c19d1-d1e9-46cc-8b0e-60c3f4a6d033" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/59bb32c1-900b-4ca1-a22b-5a1038a35389" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/2d12a561-33e0-4960-85ce-d28069d35fb5" width="200">
<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/cefd0d39-a5f3-4b01-902b-510a47533629" width="200">

Reconstruction Images(reconstruction in black font and original in white font)

<img src="https://github.com/explainingai-code/Pytorch-VAE/assets/144267687/20241c23-9256-46f2-b525-4d2215e8c18e" width="500">
