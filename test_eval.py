import torch, torchvision.utils as vu
from pathlib import Path
from cleanfid import fid
from data import CelebA64
import shutil

n = 2000
print("Loading the dataset", n, "images")
ds = CelebA64("data/img_align_celeba", n = n)

print("Dataset loaded")
print("="*50)
perm = torch.randperm(len(ds), generator = torch.Generator().manual_seed(0))

def export(idx, out):
    print("="*50)
    out = Path(out)
    if out.exists():
        print("Directory already exists, removing it...")
        shutil.rmtree(out)
        print("Removed !")
    out.mkdir(parents= True, exist_ok=True)
    print("writing images")
    for j, i in enumerate(idx):
        vu.save_image(ds[i]*0.5 + 0.5, out / f"{j:05d}.png")
    print(f"done writing {len(idx)} images")

export(perm[:n//2], "data/fid_test1")
export(perm[n//2:n], "data/fid_test2")
print("="*50)
print("Fid value between two sets:",fid.compute_fid("data/fid_test1", "data/fid_test2", mode="clean", num_workers=0, device=torch.device("cpu")))