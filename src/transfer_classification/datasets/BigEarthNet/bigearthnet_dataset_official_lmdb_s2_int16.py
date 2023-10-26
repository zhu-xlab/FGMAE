import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lmdb
from tqdm import tqdm

### SSL4EO stats 
S1_MEAN = [-12.59, -20.26]
S1_STD = [5.26, 5.91]

S2A_MEAN = [756.4, 889.6, 1151.7, 1307.6, 1637.6, 2212.6, 2442.0, 2538.9, 2602.9, 2666.8, 2388.8, 1821.5]
S2A_STD = [1111.4, 1159.1, 1188.1, 1375.2, 1376.6, 1358.6, 1418.4, 1476.4, 1439.9, 1582.1, 1460.7, 1352.2]

S2C_MEAN = [ 1612.9, 1397.6, 1322.3, 1373.1, 1561.0, 2108.4, 2390.7, 2318.7, 2581.0, 837.7, 22.0, 2195.2, 1537.4]
S2C_STD = [791.0, 854.3, 878.7, 1144.9, 1127.5, 1164.2, 1276.0, 1249.5, 1345.9, 577.5, 47.5, 1340.0, 1142.9]

### BigEarthNet stats
BANDS_S2A = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

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


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def make_lmdb(dataset, lmdb_file, num_workers=6):
    loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    env = lmdb.open(lmdb_file, map_size=1099511627776)

    txn = env.begin(write=True)
    for index, (sample1, sample2, target) in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        obj = (sample1.tobytes(), sample1.shape, sample2.tobytes(), sample2.shape, target.tobytes())
        txn.put(str(index).encode(), pickle.dumps(obj))
        if index % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()


class LMDBDataset(Dataset):

    def __init__(self, lmdb_file, is_slurm_job=False, transform=None, normalize=False, bands='B12'):
        self.lmdb_file = lmdb_file
        self.transform = transform
        self.is_slurm_job = is_slurm_job
        self.normalize = normalize
        self.bands = bands

        if not self.is_slurm_job:
            self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()['entries']            
        else:
            # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
            self.env = None
            if 'train' in self.lmdb_file:
                self.length = 269695
            elif 'val' in self.lmdb_file:
                self.length = 123723
            elif 'test' in self.lmdb_file:
                self.length = 125866
            else:
                raise NotImplementedError

    def _init_db(self):
        
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        if self.is_slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env is None:
                self._init_db()
        
        with self.env.begin(write=False) as txn:
            data = txn.get(str(index).encode())
            
        #sample_s2_bytes, sample_s2_shape, target_bytes = pickle.loads(data)
        #sample_s2_bytes, sample_s2_shape, sample_s1_bytes, sample_s1_shape, target_bytes = pickle.loads(data)
        sample_s2_bytes, sample_s2_shape, _, _, target_bytes = pickle.loads(data)

        sample = np.frombuffer(sample_s2_bytes, dtype=np.int16).reshape(sample_s2_shape)
        if self.normalize:
            chs = []
            for i, band in enumerate(BANDS_S2A):
                ch = sample[:,:,i]
                ch = normalize(ch, BAND_STATS['mean'][band], BAND_STATS['std'][band])
                chs.append(ch)
            sample = np.stack(chs,-1)
        else:        
            sample = ((sample / 10000.0) * 255).astype(np.uint8)
        #sample_s1 = np.frombuffer(sample_s1_bytes, dtype=np.float32).reshape(sample_s1_shape)

        target = np.frombuffer(target_bytes, dtype=np.float32)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import os
    import argparse
    import time
    import torch
    from torchvision import transforms
    from cvtorchvision import cvtransforms
    import cv2
    import random
    import pdb


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/p/scratch/hai_dm4eo/wang_yi/data/BigEarthNet_LMDB_raw/')
    parser.add_argument('--train_frac', type=float, default=1.0)
    args = parser.parse_args()

    test_loading_time = False
    seed = 42
    

    
    augmentation = [
        cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.ToTensor(),
    ]
    train_transforms = cvtransforms.Compose(augmentation)
    
    train_dataset = LMDBDataset(
        lmdb_file=os.path.join(args.data_dir, 'train_B12_B2.lmdb'),
        transform=train_transforms
    )

    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=0)    
    for idx, (img,target) in enumerate(train_loader):
        if idx>1:
            break
        print(img.shape, img.dtype)