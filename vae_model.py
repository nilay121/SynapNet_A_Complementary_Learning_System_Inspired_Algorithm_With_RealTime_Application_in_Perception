import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn import BCELoss
from tqdm import tqdm

##########################
### MODEL
##########################

class Reshape(nn.Module):
    def __init__(self,batch_sizeGR,features,seq_length, *args):
        super().__init__()
        self.shape = args
        self.batch_sizeGR = batch_sizeGR
        self.features = features
        self.seq_length = seq_length

    def forward(self, x):
        return x.reshape(self.batch_sizeGR,self.features,self.seq_length)


class Trim(nn.Module):
    def __init__(self,data_width, *args):
        super().__init__()
        self.data_width=data_width

    def forward(self, x):
        return x[:, :, :self.data_width, :self.data_width]

class VAE(nn.Module):
    def __init__(self,input_dim, latent_embedding, device):
        super().__init__()
        self.latent_embedding = latent_embedding
        self.input_dim = input_dim
        self.device = device
        
        self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=self.input_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(num_features=512),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(num_features=256),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(num_features=64),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Flatten(),
        )    
        
        self.z_mean = torch.nn.Linear(64, self.latent_embedding)
        self.z_log_var = torch.nn.Linear(64, self.latent_embedding)
        
        self.decoder = nn.Sequential(
                torch.nn.Linear(self.latent_embedding, 64),
                nn.LeakyReLU(0.01,inplace=False),

                Reshape(-1, 64, 1),

                nn.ConvTranspose1d(64, 128, stride=2, kernel_size=3, padding=1),    
                nn.BatchNorm1d(num_features=128),            
                nn.LeakyReLU(0.01,inplace=False),

                nn.ConvTranspose1d(128, 256, stride=2, kernel_size=3, padding=1),  
                nn.BatchNorm1d(num_features=256),              
                nn.LeakyReLU(0.01,inplace=False),

                nn.ConvTranspose1d(256, 512, stride=2, kernel_size=3, padding=1),  
                nn.BatchNorm1d(num_features=512),              
                nn.LeakyReLU(0.01,inplace=False),

                nn.ConvTranspose1d(512, 600, stride=1, kernel_size=3, padding=1), 
                nn.Tanh()
                )

    def encoding_fn(self, x):
        x = x.reshape(1, 600, 1)
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device=self.device)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        # # decoded_reshaped = decoded.reshape(self.batch_sizeGR,-1)
        # # print("decoder output",decoded_reshaped.shape)
        return encoded, z_mean, z_log_var, decoded