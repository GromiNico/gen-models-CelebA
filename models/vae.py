import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    
    def __init__(self, z = 128, c=32):
        super().__init__()
    
        def down(i, o):
            return nn.Sequential(nn.Conv2d(i,o,4,2,1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))

        def up(i,o):
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1), nn.BatchNorm2d(o), nn.ReLU())
        
        self.enc = nn.Sequential(down(3, c), down(c, 2*c), down(2*c, 4*c), down(4*c, 8*c))
        
        self.mu = nn.Linear(8*c*16,z)
        self.logvar = nn.Linear(8*c*16,z)
        
        self.fc = nn.Linear(z, 8*c*16)
        
        self.dec = nn.Sequential(up(8*c, 4*c), up(4*c, 2*c), up(2*c,c), nn.ConvTranspose2d(c, 3,4,2,1), nn.Tanh())
        
        self.c = c
        
    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(self.fc(z).view(-1, 8*self.c, 4,4))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        
        z = mu + torch.rand_like(mu) * torch.exp(0.5*logvar)
        
        return self.decode(z), mu, logvar