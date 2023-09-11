VAE Implementation in pytorch with visualizations
========

This repository implements a simpleVAE for training on CPU on the MNIST dataset and provides ability
to visualize the latent space, entire manifold as well as visualize how numbers interpolate between each other.

The purpose of this project is to get a better understanding of VAE by playing with the different parameters
and visualizations.

<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/61c9c395-9daf-414b-9f0b-93a9dd330723" width="700">

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/Pytorch-VAE.git```
* ```cd Pytorch-VAE```
* ```pip install -r requirements.txt```
* For running a simple fc layer backed VAE with latent dimension as 2 run ```python run_simple_vae.py```
* For playing around with VAE and running visualizations, replace tools/train_vae.py and tools/inference.py config argument with the desired one or pass that in the next set of commands
* ```python -m tools.train_vae.py```
* ```python -m tools.inference.py ```

## Configurations
* ```config/vae_nokl.py``` - VAE with only reconstruction loss
* ```config/vae_kl.py``` - VAE with reconstruction and KL loss
* ```config/vae_kl_latent4.py``` - VAE with reconstruction and KL loss with latent dimension as 4(instead of 2)
* ```config/vae_kl_latent4_enc_channel_dec_fc_condition.py``` - Conditional VAE with reconstruction and KL loss with latent dimension as 4

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

<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/e0b3f001-26d8-42bb-8b4b-15606c90fc42" width="500">

Manifold

<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/99322dd6-3775-4d7a-9d98-c23ec922921b" width="500">


Reconstruction Images(reconstruction in black font and original in white font)

<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/472433c0-aeab-4ace-aca0-753d9a5d8b70" width="500">

## Sample Output for Conditional VAE

Because we end up passing the label to the decoder, the model ends up learning the capability to generate ALL numbers from all points in the latent space. 

The model will learn to distinguish points in latent space based on if it should generate a left or right tilted digit or how thick the stroke for digit should be. 
Below one can visulize those patterns when we attempt to generate all numbers from all points.


<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/1121095f-7cd3-4192-9ee2-590f042f783c" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/11bfe192-723e-4dff-87e6-1ff73d9ab23e" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/a914dba9-f17f-420c-b255-ec1e2a91eee8" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/4671ea59-38cf-4473-ab18-b1bedcf3d77a" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/3d2e8095-9922-44ed-86b5-70a568bd7f6a" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/de938dae-6f22-4282-b465-bdddf6c8d3ea" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/3a811160-8825-4d7d-9cd1-7e967caa2877" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/5230e3a2-131e-4858-bac1-b8a5585c61d0" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/4c9c18d2-d7f6-4d19-9fea-5ae2f0ffcd19" width="200">
<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/782bce82-fc73-4aa6-98ff-05202c628773" width="200">

Reconstruction Images(reconstruction in black font and original in white font)

<img src="https://github.com/tusharkumar91/Pytorch-VAE/assets/462771/c4ccae12-baae-4b52-9e52-16f33e62210f" width="500">
