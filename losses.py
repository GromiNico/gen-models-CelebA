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


def gradient_penalty(critic, real, fake):
    eps = torch.rand(real.size(0), 1,1,1, device = real.device)
    x_hat = eps * real + (1-eps)*fake.requires_grad_(True)
    
    f_hat = critic(x_hat)
    grad, = torch.autograd.grad(outputs=f_hat.sum(), inputs = x_hat, create_graph = True)
    
    grad_norm = grad.flatten(1).norm(2, dim=1)
    
    return ((grad_norm - 1).pow(2)).mean()

def c_loss_wgangp(critic, real, fake, lam = 10.0):
    f_real = critic(real).mean()
    f_fake = critic(fake).mean()
    
    g_p = gradient_penalty(critic, real, fake)
    
    w1_hat = f_real - f_fake
    loss = -w1_hat + lam * g_p
    
    return loss, w1_hat.detach(), g_p.detach()

def g_loss_wgangp(f_fake):
    return -f_fake.mean()
    