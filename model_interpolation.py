import torch
from models.vae import VAE
from eval_fid import VAEDecoder
import argparse
import torchvision.utils as vu
from pathlib import Path
from models.dcgan import G
import torch.nn.functional as F


def slerp(z1, z2, t, eps=1e-6):
    cos = (z1 * z2).sum(-1, keepdim=True) / (z1.norm(dim=-1, keepdim=True) * z2.norm(dim=-1, keepdim=True))
    omega = torch.acos(cos.clamp(-1 + eps, 1 - eps))
    so = torch.sin(omega)

    return torch.sin((1 - t) * omega) / so * z1 + torch.sin(t * omega) / so * z2

    


def  main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices = ['dcgan', 'wgangp', 'vae'], default = 'vae')
    p.add_argument('--interp', choices = ['l', 's'], default = 'l')
    p.add_argument('--ck_path', required=True)
    p.add_argument('--len', type = int, default = 8)
    p.add_argument('--out', default = "data/model_interp")
    p.add_argument('--seed1', type=int, default = 0)
    p.add_argument('--seed2', type=int, default = 1)
    
    args = p.parse_args()
    
    gan = (args.model != 'vae')
    z_dim = 100 if gan else 128
    
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    
    #generating 2 sets of random samples
    z1 = torch.randn(8, z_dim, generator = torch.Generator(device = dev).manual_seed(args.seed1))
    z2 = torch.randn(8, z_dim, generator = torch.Generator(device = dev).manual_seed(args.seed2))
    
    if not gan:
        vae = VAE(z=z_dim).to(dev)
        ck = torch.load(args.ck_path, map_location = dev)
        vae.load_state_dict(ck['model'])
        gen = VAEDecoder(vae)
        print(f"Loaded VAE Generator, {ck['epoch']} epochs done")
    
    else:
        gen = G(z=z_dim)
        ck = torch.load(args.ck_path, map_location = dev)
        gen.load_state_dict(ck['netG'])
        print(f"Loaded {args.model} Generator, {ck['epoch']} epochs done")
    
    gen.eval()
    
    t = torch.linspace(0,1, args.len, device = dev).view(1,-1,1)
    
    if args.interp == 'l':
        z = torch.lerp(z1[:,None], z2[:,None], t)
    
    else:
        z = slerp(z1[:,None], z2[:,None], t)

    z = z.reshape(-1, z_dim)
    
    with torch.no_grad():
        imgs = gen(z)*0.5+0.5
    vu.save_image(imgs, out / f'interp_{args.model}_len{args.len}_interp_{args.interp}_seed{args.seed1}_{args.seed2}.png', nrow = args.len)
    print("Image saved")
    
if __name__ == '__main__':
    main()