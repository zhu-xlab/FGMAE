import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lmdb
import pdb
import os
from PIL import Image
from skimage.feature import hog
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import matplotlib.pyplot as plt

class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)

class SSL4EORGB(Dataset):

    def __init__(self, root_s2, transforms):
        self.root_s2 = root_s2
        self.transforms = transforms
        #self.feature = feature        
        self.indices = os.listdir(self.root_s2)
        self.length = len(self.indices)

    def __getitem__(self, index):
        sample_id = self.indices[index]
        patch_dir_s2 = os.path.join(self.root_s2, sample_id)
        
        season1 = np.random.choice([0,1,2,3])
        #season1 = 0
            
        fname_s2 = os.listdir(patch_dir_s2)[season1]
        img_s2 = Image.open(os.path.join(patch_dir_s2,fname_s2))
        img_s2 = np.asarray(img_s2) # 264,264,3
        
        '''
        ## very slow, better insert into the model
        if self.feature == 'raw':
            img_feat = img_s2
        elif self.feature == 'HOG':
            #(H, hogImage) = hog(img_s2, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1", visualize=True)
            (H, hogImage0) = hog(img_s2[:,:,0], orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3), block_norm='L2-Hys',visualize=True)
            (H, hogImage1) = hog(img_s2[:,:,1], orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3), block_norm='L2-Hys',visualize=True)
            (H, hogImage2) = hog(img_s2[:,:,2], orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3), block_norm='L2-Hys',visualize=True)
            img_feat = np.stack((hogImage0,hogImage1,hogImage2),-1)
            # outlier
            p2,p98 = np.percentile(img_feat,(0,98))
            img_feat = np.clip(img_feat,p2,p98)
            if not p98-p2 == 0:
                img_feat = (img_feat - p2) / (p98-p2)
            else:
                img_feat = np.zeros(img_feat.shape)
        else:
            raise NotImplementedError
                    
        if self.transforms is not None:                 
            image,feat = self.transform(img_s2,img_feat) # 3,224,224
        else:
            image = TF.to_tensor(img_s2)
            feat = TF.to_tensor(img_feat)
            resize = transforms.Resize(size=(224, 224))
            image = resize(image)
            feat = resize(feat)
        '''
        if self.transforms is not None:
            image = self.transforms(img_s2)
        else:
            image = TF.to_tensor(img_s2)  
            resize = transforms.Resize(size=(224, 224))
            image = resize(image)                        
        return image
        
    def __len__(self):
        return self.length
    '''
    def transform(self, image, mask):

        # Transform to tensor
        image = TF.to_tensor(image) # 3,264,264
        mask = TF.to_tensor(mask) # 1,264,264
        #print(image.shape, mask.shape)
        
        # Random crop
        ratio = random.uniform(0.2, 1.0)
        crop_size = int(224 * ratio)
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(crop_size,crop_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Resize
        resize = transforms.Resize(size=(224, 224))
        image = resize(image)
        mask = resize(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask
    '''