import argparse, csv, time
from pathlib import Path
import torch
import torchvision.utils as vu
from torch.utils.data import DataLoader

from data import CelebA64
from models.vae import VAE
from models.utils import seed_all
from losses import VAE_loss

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
    
    #2 setup, fixed seed, GPU, out dir
    
    seed_all(0)
    dev = "cuda" if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    
    #3 Data: batches of bs,3,64,64 of [-1,1] images
    loader = DataLoader(CelebA64(args.data_root, n = args.n), batch_size = args.bs, shuffle = True, num_workers = 2, pin_memory = (dev == 'cuda'), drop_last = True)
    
    #4 Model on the device and the optimizer that will update its weights
    
    model = VAE(z=128).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    #5 Fixed latent vectors so sample grids from different epochs are comparable
    z_path = out/ "z_eval.pt"
    if z_path.exists():
        z_eval = torch.load(z_path)
    else:
        z_eval = torch.randn(64, 128, generator = torch.Generator().manual_seed(1234))
        torch.save(z_eval, z_path)
    z_eval = z_eval.to(dev)
    
    
    #6 resume from last checkpoint if it exists
    
    start_epoch, step = 0,0
    ckpt_path = out / "last.pt"
    
    if ckpt_path.exists() and not args.fresh:
        ck = torch.load(ckpt_path, map_location = dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_epoch, step = ck['epoch']+1, ck['step']
        print(f"Resuming from {ckpt_path}, {start_epoch} epochs already done.")
    
    if start_epoch >= args.epochs:
        print(f"Nothing to do, already trained for {start_epoch} epochs, raise epochs or add --fresh to strat again")
    #7 training loop
    
    log = open(out / 'log.csv', 'a' if step > 0 else 'w', newline = "")
    writer = csv.writer(log)
    if step == 0:
        writer.writerow(['step', 'loss', 'rec', 'kl'])
        
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        for x in loader:
            x = x.to(dev, non_blocking = True)
            x_hat, mu, logvar = model(x)

            loss, rec, kl = VAE_loss(x_hat, x, mu, logvar)
            
            opt.zero_grad(set_to_none = True)
            
            loss.backward()
            
            opt.step()
            
            if step%100 == 0:
                writer.writerow([step, loss.item(), rec.item(), kl.item()])
                log.flush()
                
                print(f"epochs {epoch} step {step} number of data points proceeded {step*args.bs} loss {loss.item():.1f} rec {rec.item():.1f} kl {kl.item():.1f}")
            step+=1
        
        model.eval()
        
        with torch.no_grad():
            samples = model.decode(z_eval)*0.5+0.5
        
        vu.save_image(samples, out / f"samples_ep{epoch:02d}.png", nrow = 8)
        torch.save({"model": model.state_dict(), 'opt': opt.state_dict(), 'epoch' : epoch, 'step' : step}, ckpt_path)
        
        print(f"epoch {epoch} done in {time.time()-t0:.0f} s")
    log.close()
    

if __name__ == "__main__":
    main()