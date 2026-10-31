import argparse
import torch
from models.utils import seed_all, weights_init
from pathlib import Path
from torch.utils.data import DataLoader
from data import CelebA64
from models.dcgan import G, D
import csv
import time
from losses import d_loss_dcgan, g_loss_dcgan
import torchvision.utils as vu

Z_DIM = 100

def main():
    #arguments
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = "vae")
    p.add_argument("--data_root", required=True)
    p.add_argument("--n", type = int, default =None)
    p.add_argument("--epochs", type=int, default = 10)
    p.add_argument("--bs", type = int, default = 128)
    p.add_argument("--lr", type = float, default = 1e-3)
    p.add_argument("--out", default = "/runs/vae")
    p.add_argument('--fresh', action = 'store_true')
    
    args = p.parse_args()
    
    #setup, seed, GPU, out dir
    seed_all(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    
    #3 Data: batches of bs,3,64,64 of [-1,1] images
    loader = DataLoader(CelebA64(args.data_root, n = args.n), batch_size = args.bs, shuffle = True, num_workers = 2, pin_memory = (dev == 'cuda'), drop_last = True)
    
    #initializing models and optimizers
    netG = G(Z_DIM).to(dev)
    netD = D(norm = 'bn').to(dev)
    
    netG.apply(weights_init)    
    netD.apply(weights_init)
    
    optG = torch.optim.Adam(netG.parameters(), lr = args.lr, betas=(0.5,0.999))
    optD = torch.optim.Adam(netD.parameters(), lr = args.lr, betas=(0.5,0.999))
    
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
        writer.writerow(['step', 'lossD', 'lossG', 'D_real', 'D_fake'])
    
    for epoch in range(start_epoch, args.epochs):
        netD.train()
        netG.train()
        t0 = time.time()
        
        for x in loader:
            x = x.to(dev, non_blocking = True)
            
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
            
            if step%100==0:
                p_real = torch.sigmoid(d_real).mean().item()
                p_fake = torch.sigmoid(d_fake).mean().item()
                
                writer.writerow([step, lossD.item(), lossG.item(), p_real, p_fake])
                log.flush()
                
                print(f"epoch {epoch} step {step} lossD {lossD:.2f} lossG {lossG:.2f} D_real {p_real:.2f} D_fake {p_fake:.2f}")
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