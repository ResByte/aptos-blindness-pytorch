import pandas as pd 
from PIL import Image 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader 

class AptosDataset(Dataset):
    def __init__(self,
                img_root:str,
                df:pd.DataFrame,
                img_transforms:transforms=None,
                is_train:bool=True
                ):
        
        self.df = df
        
        self.img_root = img_root
        self.img_transforms = img_transforms
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx:int):
        row = self.df.iloc[idx]
        filename = row['id_code']
        target = int(row['diagnosis'])
        img = Image.open(os.path.join(self.img_root, filename+'.png')).convert('RGB')
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        return img, np.asarray(target)