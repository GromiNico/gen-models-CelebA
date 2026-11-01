from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image #to open jpg files


# creation of the CelebA34 class of type Dataset
# it needs len and getitem to have the Dataset type
class CelebA64(Dataset):
    
    #root = folder of JPEG files
    #n to keep only the first n images
    def __init__(self, root, n = None, start = None):
        # sorted sorts based on alphabetical order
        #glob looks for names that follow the specified pattern
        self.files = sorted(Path(root).glob("*.jpg"))[start:n]
        
        #antialias = True blurs the image before sampling and making it 64*64
        # renormalize to [-1,1] to fit the generation
        self.tf = T.Compose([T.CenterCrop(140), T.Resize(64, antialias = True), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert('RGB'))