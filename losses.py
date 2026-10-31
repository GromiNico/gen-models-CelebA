import torch
import torch.nn.functional as F


def VAE_loss(x_hat, x, mu, logvar, beta = 0.1):
    B = x.size(0)
    
    rec = F.mse_loss(x_hat, x, reduction = 'sum')/B
    
    kl = -0.5*torch.sum(1+logvar - mu.pow(2) - logvar.exp())/B
    
    return rec + beta*kl, rec, kl


def d_loss_dcgan(d_real, d_fake):
    loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    
    return loss_real + loss_fake

def g_loss_dcgan(d_fake):
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))