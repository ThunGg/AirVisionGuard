import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import io
from PIL import Image
import torchvision.transforms as transforms
try:
    import mc
except ImportError:
    pass
from typing import List, Tuple, Sequence, Iterable
from dataclasses import dataclass
from tqdm import tqdm
from utils import bin_loader, numeric_sort_key

import pdb

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    label: int
    identity: str


class FaceDataset(Dataset):
    """
    Standard dataset that opens images from paths provided in a list of SampleRecords.
    """
    def __init__(self, samples: Sequence[SampleRecord], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.num_class = len(set(s.label for s in self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with open(sample.image_path, 'rb') as f:
            image = pil_loader(f.read())
        if self.transform:
            image = self.transform(image)
        return image, sample.label


def index_image_folder(dataset_dir: str | os.PathLike, image_extensions: Iterable[str] = ('.jpg', '.jpeg', '.png')) -> Tuple[List[SampleRecord], List[str]]:
    """
    Efficiently index a folder of images organized by identity subdirectories.
    """
    dataset_path = os.path.abspath(dataset_dir)
    allowed_suffixes = {suffix.lower() for suffix in image_extensions}

    identities = sorted(
        [entry.name for entry in os.scandir(dataset_path) if entry.is_dir()],
        key=numeric_sort_key,
    )
    label_mapping = {identity: index for index, identity in enumerate(identities)}
    samples: List[SampleRecord] = []

    for identity in tqdm(identities, desc="Indexing Images"):
        identity_dir = os.path.join(dataset_path, identity)
        with os.scandir(identity_dir) as entries:
            for entry in entries:
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in allowed_suffixes:
                    samples.append(
                        SampleRecord(
                            image_path=entry.path,
                            label=label_mapping[identity],
                            identity=identity,
                        )
                    )

    return samples, identities


class GivenSizeSampler(Sampler):
    '''
    Sampler with given total size, supporting distributed training
    '''
    def __init__(self, dataset, total_size=None, rand_seed=None, sequential=False, silent=False):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
            
        self.rand_seed = rand_seed if rand_seed is not None else 0
        self.dataset = dataset
        self.epoch = 0
        self.sequential = sequential
        self.silent = silent
        
        # Global total size
        global_total_size = total_size if total_size is not None else len(self.dataset)
        # Each replica should have the same number of samples
        self.num_samples = int(np.ceil(global_total_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas # Adjusted global size

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.sequential:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.rand_seed)
            origin_indices = list(torch.randperm(len(self.dataset), generator=g))
        else:
            origin_indices = list(range(len(self.dataset)))
        indices = origin_indices[:]

        # add extra samples to meet self.total_size
        extra = self.total_size - len(origin_indices)
        if not self.silent and self.rank == 0:
            print('Origin Size: {}\tAligned Global Size: {}\tPer-rank Size: {}'.format(len(origin_indices), self.total_size, self.num_samples))
        
        if extra < 0:
            indices = indices[:self.total_size]
        while extra > 0:
            intake = min(len(origin_indices), extra)
            indices += origin_indices[:intake]
            extra -= intake
        
        # slice for the current rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples, "{} vs {}".format(len(indices), self.num_samples)

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class BinDataset(Dataset):
    def __init__(self, bin_file, transform=None):
        self.img_lst, self.lbs = bin_loader(bin_file)
        self.num = len(self.img_lst)
        self.transform = transform

    def __len__(self):
        return self.num

    def _read(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.num)
        try:
            # The img_lst now contains raw bytes, decode on-demand to save memory
            raw_img = self.img_lst[idx]
            img = pil_loader(raw_img)
            return img
        except Exception as err:
            print('Read image[{}] failed ({})'.format(idx, err))
            return self._read()

    def __getitem__(self, idx):
        img = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_labeled_dataset(filelist, prefix):
    img_lst = []
    lb_lst = []
    with open(filelist) as f:
        for x in f.readlines():
            n, lb = x.strip().split(' ')
            lb = int(lb)
            img_lst.append(os.path.join(prefix, n))
            lb_lst.append(lb)
    assert len(img_lst) == len(lb_lst)
    return img_lst, lb_lst

def build_unlabeled_dataset(filelist, prefix):
    img_lst = []
    with open(filelist) as f:
        for x in f.readlines():
            img_lst.append(os.path.join(prefix, x.strip().split(' ')[0]))
    return img_lst


class FileListLabeledDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None, memcached=False, memcached_client=''):
        self.img_lst, self.lb_lst = build_labeled_dataset(filelist, '') # Don't join prefix here
        self.prefix = prefix
        self.num = len(self.img_lst)
        self.transform = transform
        self.num_class = max(self.lb_lst) + 1
        self.initialized = False
        self.memcached = memcached
        self.memcached_client = memcached_client

    def __len__(self):
        return self.num

    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = os.path.join(self.prefix, self.img_lst[idx])
        lb = self.lb_lst[idx]
        try:
            if self.memcached:
                value = mc.pyvector()
                self.mclient.Get(fn, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)
            else:
                img = pil_loader(open(fn, 'rb').read())
            return img, lb
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        if self.memcached:
            self.__init_memcached()
        img, lb = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, lb

class FileListDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None, memcached=False, memcached_client=''):
        self.img_lst = build_unlabeled_dataset(filelist, '') # Don't join prefix here
        self.prefix = prefix
        self.num = len(self.img_lst)
        self.transform = transform
        self.initialized = False
        self.memcached = memcached
        self.memcached_client = memcached_client

    def __len__(self):
        return self.num

    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = os.path.join(self.prefix, self.img_lst[idx])
        try:
            #img = pil_loader(open(fn, 'rb').read())
            if self.memcached:
                value = mc.pyvector()
                self.mclient.Get(fn, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)
            else:
                img = pil_loader(open(fn, 'rb').read())
            return img
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        if self.memcached:
            self.__init_memcached()
        img = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img
