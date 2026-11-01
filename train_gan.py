import argparse
import torch
from models.utils import seed_all, weights_init
from pathlib import Path
from torch.utils.data import DataLoader
from data import CelebA64
from models.dcgan import G, D
import csv
import time
from losses import d_loss_dcgan, g_loss_dcgan, c_loss_wgangp, gradient_penalty, g_loss_wgangp
import torchvision.utils as vu

Z_DIM = 100

def main():
    #arguments
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = "dcgan", choices = ['dcgan', 'wgangp'])
    p.add_argument('--n_critic', type = int, default = 5)
    p.add_argument('--lam', type = float, default = 10.0)
    p.add_argument("--data_root", default = 'data/img_align_celeba')
    p.add_argument("--n", type = int, default = None)
    p.add_argument("--epochs", type=int, default = 10)
    p.add_argument("--bs", type = int, default = 128)
    p.add_argument("--lr", type = float)
    p.add_argument("--out", default = "/runs/dcgan")
    p.add_argument('--fresh', action = 'store_true')
    
    args = p.parse_args()
    
    wgan = args.model == 'wgangp'
    if args.lr == None:
        args.lr = 1e-4 if wgan else 2e-4
    if args.out == None:
        args.out = f"runs/{args.model}"
    
    #setup, seed, GPU, out dir
    seed_all(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    
    #3 Data: batches of bs,3,64,64 of [-1,1] images
    loader = DataLoader(CelebA64(args.data_root, n = args.n), batch_size = args.bs, shuffle = True, num_workers = 2, pin_memory = (dev == 'cuda'), drop_last = True)
    
    #initializing models and optimizers
    netG = G(Z_DIM).to(dev)
    netD = D(norm = 'ln' if wgan else 'bn').to(dev)
    
    netG.apply(weights_init)    
    netD.apply(weights_init)
    betas = (0.0, 0.9) if wgan else (0.5, 0.999)
    
    optG = torch.optim.Adam(netG.parameters(), lr = args.lr, betas=betas)
    optD = torch.optim.Adam(netD.parameters(), lr = args.lr, betas=betas)
    
    #5 Fixed latent vectors so sample grids from different epochs are comparable
    z_path = out/ "z_eval.pt"
    if z_path.exists():
        z_eval = torch.load(z_path)
    else:
        z_eval = torch.randn(64, Z_DIM, generator = torch.Generator().manual_seed(1234))
        torch.save(z_eval, z_path)
    z_eval = z_eval.to(dev)
    
    
    #6 resume from last checkpoint if it exists
        
    start_epoch, step = 0,0
    ckpt_path = out / "last.pt"
        
    if ckpt_path.exists() and not args.fresh:
        ck = torch.load(ckpt_path, map_location = dev)
        netD.load_state_dict(ck['netD']); netG.load_state_dict(ck['netG'])
        optG.load_state_dict(ck['optG']); optD.load_state_dict(ck['optD'])
        start_epoch, step = ck['epoch']+1, ck['step']
        print(f"Resuming from {ckpt_path}, {start_epoch} epochs already done.")
      
    if start_epoch >= args.epochs:
        print(f"Nothing to do, already trained for {start_epoch} epochs, raise epochs or add --fresh to strat again")
        return
    
    #7 training loop
    
    log = open(out / 'log.csv', 'a' if step >0 else 'w', newline = '')
    writer = csv.writer(log)
    
    if step == 0:
        writer.writerow(['step', 'lossC', 'lossG', 'w1_hat', 'gp'] if wgan else ['step', 'lossD', 'lossG', 'D_real', 'D_fake'])
    
    for epoch in range(start_epoch, args.epochs):
        netD.train()
        netG.train()
        t0 = time.time()
        
        for i, x in enumerate(loader):
            x = x.to(dev, non_blocking = True)
            
            if not wgan:
                d_real = netD(x)
            
                z = torch.randn(x.size(0), Z_DIM, device = dev)
            
                fake = netG(z)
                d_fake = netD(fake.detach())
            
                lossD = d_loss_dcgan(d_real, d_fake)
                optD.zero_grad(set_to_none= True)
                lossD.backward()
                optD.step()
            
                lossG = g_loss_dcgan(netD(fake))
                optG.zero_grad(set_to_none=True)
                lossG.backward()
                optG.step()
            
            else: #wgangp
                z = torch.randn(x.size(0), Z_DIM, device = dev)
                
                with torch.no_grad():
                    fake = netG(z)
                lossD, w1_hat, g_p = c_loss_wgangp(netD, x, fake, lam = args.lam)
                optD.zero_grad(set_to_none = True)
                
                lossD.backward()
                optD.step()
                
                if (i+1)%args.n_critic != 0:
                    continue
                
                z = torch.randn(x.size(0), Z_DIM, device = dev)
                lossG = g_loss_wgangp(netD(netG(z)))
                optG.zero_grad(set_to_none = True)
                
                lossG.backward()
                optG.step()
            
            if step%100==0:
                if not wgan:
                    p_real = torch.sigmoid(d_real).mean().item()
                    p_fake = torch.sigmoid(d_fake).mean().item()
                    
                    writer.writerow([step, lossD.item(), lossG.item(), p_real, p_fake])
                    log.flush()
                    
                    print(f"epoch {epoch} step {step} lossD {lossD:.2f} lossG {lossG:.2f} D_real {p_real:.2f} D_fake {p_fake:.2f}")
                
                else:
                    writer.writerow([step, lossD.item(), lossG.item(), w1_hat.item(), g_p.item()])
                    log.flush()
                    
                    print(f"epoch {epoch} step {step} loss critic {lossD.item():.2f} lossG {lossG.item():.2f} w1_hat {w1_hat.item():.2f} gp {g_p.item():.2f}")
            step+=1
        
        netG.eval()
        
        with torch.no_grad():
            samples = netG(z_eval)*0.5 + 0.5
        
        vu.save_image(samples, out / f"samples_ep{epoch:02d}.png", nrow = 8)
        torch.save({'netD': netD.state_dict(), 'netG': netG.state_dict(), 'optD': optD.state_dict(), 'optG': optG.state_dict(), 'step' : step, 'epoch': epoch}, ckpt_path)
        
        print(f"epoch {epoch} done in {time.time()-t0:.0f} s")
    log.close()

if __name__ == '__main__':
    main()