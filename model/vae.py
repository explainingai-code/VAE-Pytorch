import torch
import torch.nn as nn


r"""
A very simple VAE which has the following architecture
Encoder
    For Conditional model we stack num_classes empty channels onto the image
        We make the gt_label index channel as `1` (See figure in README)
    N * Conv BN Activation Blocks
    FC layers for mean
    FC layers for variance

Decoder
    For Conditional model we also concat the one hot label feature onto the z input
        (See figure in README)
    FC Layers taking z to higher dimensional feature
    N * ConvTranspose BN Activation Blocks
"""
class VAE(nn.Module):
    def __init__(self,
                 config
                 ):
        super(VAE, self).__init__()
        activation_map = {
            'relu': nn.ReLU(),
            'leaky': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        
        self.config = config
        ##### Validate the configuration for the model is correctly setup #######
        assert config['transpose_activation_fn'] is None or config['transpose_activation_fn'] in activation_map
        assert config['dec_fc_activation_fn'] is None or config['dec_fc_activation_fn'] in activation_map
        assert config['conv_activation_fn'] is None or config['conv_activation_fn'] in activation_map
        assert config['enc_fc_activation_fn'] is None or config['enc_fc_activation_fn'] in activation_map
        assert config['enc_fc_layers'][-1] == config['dec_fc_layers'][0] == config['latent_dim'], \
            "Latent dimension must be same as fc layers number"
        
        self.num_classes = config['num_classes']
        self.transposebn_channels = config['transposebn_channels']
        self.latent_dim = config['latent_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Number of input channels will change if its a conditional model
        if config['concat_channel'] and config['conditional']:
            config['convbn_channels'][0] += self.num_classes
        
        # Encoder is just Conv bn blocks followed by fc for mean and variance
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config['convbn_channels'][i], config['convbn_channels'][i + 1],
                          kernel_size=config['conv_kernel_size'][i], stride=config['conv_kernel_strides'][i]),
                nn.BatchNorm2d(config['convbn_channels'][i + 1]),
                activation_map[config['conv_activation_fn']]
            )
            for i in range(config['convbn_blocks'])
        ])
        
        encoder_mu_activation = nn.Identity() if config['enc_fc_mu_activation'] is None else activation_map[
            config['enc_fc_mu_activation']]
        self.encoder_mu_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['enc_fc_layers'][i], config['enc_fc_layers'][i + 1]),
                encoder_mu_activation
            )
            for i in range(len(config['enc_fc_layers']) - 1)
        ])
        encoder_var_activation = nn.Identity() if config['enc_fc_var_activation'] is None else activation_map[
            config['enc_fc_var_activation']]
        self.encoder_var_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['enc_fc_layers'][i], config['enc_fc_layers'][i + 1]),
                encoder_var_activation
            )
            for i in range(len(config['enc_fc_layers']) - 1)
        ])
        
        # Number of features will change if it's a conditional model
        if config['decoder_fc_condition'] and config['conditional']:
            config['dec_fc_layers'][0] += self.num_classes

        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(config['transposebn_channels'][i], config['transposebn_channels'][i + 1],
                                   kernel_size=config['transpose_kernel_size'][i],
                                   stride=config['transpose_kernel_strides'][i]),
                nn.BatchNorm2d(config['transposebn_channels'][i + 1]),
                activation_map[config['transpose_activation_fn']]
            )
            for i in range(config['transpose_bn_blocks'])
        ])
        
        self.decoder_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['dec_fc_layers'][i], config['dec_fc_layers'][i + 1]),
                activation_map[config['dec_fc_activation_fn']]
            )
            for i in range(len(config['dec_fc_layers']) - 1)
        
        ])
    
    def forward(self, x, label=None):
        out = x
        if self.config['concat_channel'] and self.config['conditional']:
            # Stack the label feature maps onto the input if its a conditional model
            # And config asks to do so
            label_ch_map = torch.zeros((x.size(0), self.num_classes, *x.shape[2:])).to(self.device)
            batch_idx, label_idx = (torch.arange(0, x.size(0), device=self.device),
                                    label[torch.arange(0, x.size(0), device=self.device)])
            label_ch_map[batch_idx, label_idx, :, :] = 1
            out = torch.cat([x, label_ch_map], dim=1)
            
        for layer in self.encoder_layers:
            out = layer(out)
        out = out.reshape((x.size(0), -1))
        mu = out
        for layer in self.encoder_mu_fc:
            mu = layer(mu)
        std = out
        for layer in self.encoder_var_fc:
            std = layer(std)

        z = self.reparameterize(mu, std)
        generated_out = self.generate(z, label)
        if self.config['log_variance']:
            return {
                'mean': mu,
                'log_variance': std,
                'image': generated_out,
            }
        else:
            return {
                'mean': mu,
                'std': std,
                'image': generated_out,
            }
    
    def generate(self, z, label=None):
        out = z
        if self.config['decoder_fc_condition'] and self.config['conditional']:
            assert label is not None, "Label cannot be none for conditional generation"
            # Concat the num_classes dimensional one hot feature vector onto z
            # For label 9 this will be [0,0,0,0,0,0,0,0,0,1]
            label_fc_input = torch.zeros((z.size(0), self.num_classes)).to(self.device)
            batch_idx, label_idx = (torch.arange(0, z.size(0), device=self.device),
                                    label[torch.arange(0, z.size(0), device=self.device)])
            label_fc_input[batch_idx, label_idx] = 1
            out = torch.cat([out, label_fc_input], dim=-1)
        for layer in self.decoder_fc:
            out = layer(out)
        # Figure out how to reshape based on desired number of channels in transpose convolution
        hw = torch.as_tensor(out.size(-1) / self.transposebn_channels[0]).to(self.device)
        spatial = int(torch.sqrt(hw))
        assert spatial * spatial == hw
        out = out.reshape((z.size(0), -1, spatial, spatial))
        for layer in self.decoder_layers:
            out = layer(out)
        return out
    
    def sample(self, label=None, num_images=1, z=None):
        if z is None:
            z = torch.randn((num_images, self.latent_dim))
        if self.config['conditional']:
            assert label is not None, "Label cannot be none for conditional sampling"
            assert label.size(0) == num_images
        assert z.size(0) == num_images
        out = self.generate(z, label)
        return out
    
    def reparameterize(self, mu, std_or_logvariance):
        if self.config['log_variance']:
            std = torch.exp(0.5 * std_or_logvariance)
        else:
            std = std_or_logvariance
        z = torch.randn_like(std)
        return z * std + mu


def get_model(config):
    model = VAE(
        config=config['model_params']
    )
    return model


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import yaml
    
    config_path = '../config/vae_kl_latent4.yaml'
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    model = get_model(config)
    labels = torch.zeros((3)).long()
    labels[0] = 0
    labels[1] = 2
    out = model(torch.rand((3,1,28,28)), labels)
    print(out['mean'].shape)
    print(out['image'].shape)


