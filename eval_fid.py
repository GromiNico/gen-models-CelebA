import torch
import torchvision.utils as vu
from pathlib import Path
from cleanfid import fid
import torch.nn as nn
import argparse, shutil
from models.vae import VAE
from data import CelebA64

fid.EXTENSIONS = {"png"}

@torch.no_grad()
def sample_to_dir(G, out_dir, n = 10000, z_dim = 100, bs = 500, device = 'cpu'):
    G.eval()
    
    #creating the out directory
    #doesn't fail if parents miss or if it exists already
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok = True)
    
    g = torch.Generator(device=device).manual_seed(0)
    for k in range(0, n, bs):
        # generates bs vectors following normal distribution in z_dim dimensions
        z = torch.randn(bs, z_dim, device = device, generator = g)
        
        x = (G(z).clamp(-1,1)*0.5 + 0.5)
        
        # j is the index
        # img is the 
        for j, img in enumerate(x):
            vu.save_image(img, out / f"{k+j:05d}.png")
    
    G.train()


def export(idx, out, ds):
    out = Path(out)
    if out.exists():
        print(f"Folder {out} of real images already exists, deleting it...")
        shutil.rmtree(out)
        print("Folder deleted")
    out.mkdir(parents = True, exist_ok = True)
    print(f"Saving real images at {out}")
    for j, i in enumerate(idx):
        vu.save_image(ds[i], out / f"{j:5d}.png")        

class VAEDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    
    def forward(self, z):
        return self.vae.decode(z)
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required = True)
    p.add_argument('--real', required = True)
    p.add_argument('--gen', default = 'data/fid_gen')
    p.add_argument('--n', type=int, default = 10_000)
    
    args = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #creating the folder of real images
    ds = CelebA64("data/img_align_celeba", n = args.n)
    perm = torch.randperm(len(ds), generator = torch.Generator(device = dev).manual_seed(0))
    export(perm, args.real, ds)
    print(f"{args.n} random images saved at {args.real}")
    
    # Loading the model from last ckpt
    vae = VAE(z=128).to(dev)
    ck = torch.load(args.ckpt, map_location = dev)
    vae.load_state_dict(ck['model'])
    print(f"model loaded, {ck['epochs']} done")
    
    #deleting last generated attemps
    shutil.rmtree(args.gen, ignore_errors = True)
    
    #Sampling from the model distribution
    print("Sampling from the model distribution")
    sample_to_dir(VAEDecoder(vae), args.gen, n = args.n, z_dim = 128, device = dev)
    
    score = fid.compute_fid(args.real, args.gen, mode = 'clean', num_workers = 0, device = dev)
    print(f"FID {score:.2f} computed on {args.n} images after epoch {ck['epoch']}")

if __name__ == '__main__':
    main()