import torch
import torchvision.utils as vu
from pathlib import Path
from cleanfid import fid
import torch.nn as nn
import argparse, shutil
from models.vae import VAE
from data import CelebA64
from models.dcgan import G, D

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
        # generates b vectors following normal distribution in z_dim dimensions
        # the last batch is smaller so that exactly n images are saved
        b = min(bs, n - k)
        z = torch.randn(b, z_dim, device = device, generator = g)
        
        x = (G(z).clamp(-1,1)*0.5 + 0.5)
        
        # j is the index
        # img is the 
        for j, img in enumerate(x):
            vu.save_image(img, out / f"{k+j:05d}.png")
    
    G.train()

def make_ref_stats(fid_real, name, device = 'cpu'):
    # Inception statistics of the real images, computed once and reused at every fid_eval call
    if fid.test_stats_exists(name, mode = 'clean'):
        fid.remove_custom_stats(name, mode = 'clean')
    fid.make_custom_stats(name, fid_real, mode = 'clean', num_workers = 0, device = device)

def fid_eval(ref_name, G, out_dir, n = 10_000, z_dim = 100, bs = 128, device = 'cpu'):

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)

    sample_to_dir(G,out_dir = out_dir, n = n, z_dim = z_dim, device = device, bs = bs)
    score = fid.compute_fid(out_dir, dataset_name = ref_name, dataset_split = 'custom', mode = 'clean', num_workers = 4, device = device)

    return score

def export(idx, out, ds):
    out = Path(out)
    if out.exists():
        print(f"Folder {out} of real images already exists, deleting it...")
        shutil.rmtree(out)
        print("Folder deleted")
    out.mkdir(parents = True, exist_ok = True)
    print(f"Saving real images at {out}")
    for j, i in enumerate(idx):
        # ds returns images in [-1,1], save_image expects [0,1]
        vu.save_image(ds[i]*0.5 + 0.5, out / f"{j:05d}.png")

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
    p.add_argument('--model', choices = ['vae', 'dcgan'], default = 'vae')
    
    args = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #creating the folder of real images
    ds = CelebA64("data/img_align_celeba", n = args.n)
    perm = torch.randperm(len(ds), generator = torch.Generator().manual_seed(0))
    export(perm, args.real, ds)
    print(f"{args.n} random images saved at {args.real}")
    
    # Loading the model from last ckpt
    ck = torch.load(args.ckpt, map_location = dev)
    
    if args.model == "vae":
        vae, z_dim = VAE(z=128).to(dev), 128
        vae.load_state_dict(ck['model'])
        gen = VAEDecoder(vae)
        print(f"VAE generator loaded, {ck['epoch']} done")
    
    elif args.model == "dcgan":
        gen, z_dim = G(z = 100).to(dev), 100
        gen.load_state_dict(ck['netG'])
        print(f"DCGAN generator loaded, {ck['epoch']} done")
    
    #deleting last generated attemps
    shutil.rmtree(args.gen, ignore_errors = True)
    
    #Sampling from the model distribution
    print("Sampling from the model distribution")
    sample_to_dir(gen, args.gen, n = args.n, z_dim = z_dim, device = dev)
    
    score = fid.compute_fid(args.real, args.gen, mode = 'clean', num_workers = 0, device = dev)
    print(f"FID {score:.2f} computed on {args.n} images after epoch {ck['epoch']}")

if __name__ == '__main__':
    main()