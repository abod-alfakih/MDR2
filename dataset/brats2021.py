import os
from os.path import join
import torch
import random
import torch
import monai
from torch.utils.data import DataLoader, ConcatDataset


import numpy as np
import monai.transforms as transforms
import nibabel as nib
from monai.transforms import EnsureChannelFirstd
from torch.utils.data import Dataset, DataLoader
from .dataset_utils import nib_load, RobustZScoreNormalization

seed=42
def get_brats2021_base_transform():
    base_transform = [
        # Removed EnsureChannelFirstD
        transforms.Orientationd(keys=['flair', 't1', 't1ce', 't2', 'binary_label', 'label'], axcodes="RAS"),
        RobustZScoreNormalization(keys=['flair', 't1', 't1ce', 't2']),
        transforms.ConcatItemsd(keys=['flair', 't1', 't1ce', 't2','binary_label'], name='image', dim=0),
        transforms.DeleteItemsd(keys=['flair', 't1', 't1ce', 't2','binary_label']),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
    ]
    return base_transform


def get_brats2021_train_transform(args, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    monai.utils.set_determinism(seed=seed)

    base_transform = get_brats2021_base_transform()
    data_aug = [
        # crop
        transforms.RandCropByPosNegLabeld(
            keys=["image", 'label'],
            label_key='label',
            spatial_size=[args.patch_size] * 3,
            pos=args.pos_ratio,
            neg=args.neg_ratio,
            num_samples=1),

        # spatial aug
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),

        # intensity aug
        transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

        # other stuff
        transforms.EnsureTyped(keys=["image", 'label']),
    ]

    return transforms.Compose(base_transform + data_aug)

def get_meta_brats2021_train_transform(args, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    monai.utils.set_determinism(seed=seed)

    base_transform = get_brats2021_base_transform()
    data_aug = [
        # crop
        transforms.RandCropByPosNegLabeld(
            keys=["image", 'label'],
            label_key='label',
            spatial_size=[args.patch_size] * 3,
            pos=args.pos_ratio,
            neg=args.neg_ratio,
            num_samples=3),

        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),

        # intensity aug
        transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

        # other stuff
        transforms.EnsureTyped(keys=["image", 'label']),
    ]

    return transforms.Compose(base_transform + data_aug)

def get_brats2021_infer_transform(args,seed=42):
    base_transform = get_brats2021_base_transform()
    infer_transform = [transforms.EnsureTyped(keys=["image", 'label'])]
    return transforms.Compose(base_transform + infer_transform)


####################################################################################################
# dataset
class Adults_BraTS2021Dataset(Dataset):
    def __init__(self, data_root: str, mode: str, case_names: list = [], transforms=None):
        super(Adults_BraTS2021Dataset, self).__init__()

        assert mode in ['train', 'infer'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.case_names = case_names
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple:
        name = self.case_names[index]  # BraTS2021_00000
        base_dir = join(self.data_root, name)  # seg/data/brats21/BraTS2021_00000/BraTS2021_00000


        # Adjust the paths to concatenate the suffix directly to 'name' before adding the extension
        flair_path = os.path.join(base_dir, name + '_flair.nii.gz')
        t1_path = os.path.join(base_dir, name + '_t1.nii.gz')
        t1ce_path = os.path.join(base_dir, name + '_t1ce.nii.gz')
        t2_path = os.path.join(base_dir, name + '_t2.nii.gz')
        binary_path = os.path.join(base_dir, name + '_binaryseg.nii.gz')
        mask_path = os.path.join(base_dir, name + '_seg.nii.gz')

        # Load the imaging data using the nib_load function and convert to float32
        flair = np.array(nib_load(flair_path), dtype='float32')
        t1 = np.array(nib_load(t1_path), dtype='float32')
        t1ce = np.array(nib_load(t1ce_path), dtype='float32')
        t2 = np.array(nib_load(t2_path), dtype='float32')
        binary_label = np.array(nib_load(binary_path), dtype='float32')
        mask = np.array(nib_load(mask_path), dtype='float32')
        # mask = torch.tensor(nib_load(mask_path), dtype=torch.float32)
        # print("FLAIR sample data:", flair[0, 0, :200])  # Print the first ten elements of the first voxel line
        flair = flair[np.newaxis, ...]
        t1 = t1[np.newaxis, ...]
        t1ce = t1ce[np.newaxis, ...]
        t2 = t2[np.newaxis, ...]
        binary_label = binary_label[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        transforms = EnsureChannelFirstd(keys=['flair', 't1', 't1ce', 't2', 'binary_label' ,'label'], channel_dim=0)

        item = {'flair': flair, 't1': t1, 't1ce': t1ce, 't2': t2, 'binary_label':binary_label ,'label': mask}
        item = self.transforms(item)


        if self.mode == 'train':  # train
            item = item[0]  # [0] for RandCropByPosNegLabeld

        return item['image'], item['label'], index, name

    def __len__(self):
        return len(self.case_names)

class BraTS2021Dataset(Dataset):
    def __init__(self, data_root: str, mode: str, case_names: list = [], transforms=None):
        super(BraTS2021Dataset, self).__init__()

        assert mode in ['train', 'infer'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.case_names = case_names
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple:
        name = self.case_names[index]  # BraTS2021_00000
        base_dir = join(self.data_root, name)  # seg/data/brats21/BraTS2021_00000/BraTS2021_00000

        # Adjust the paths to concatenate the suffix directly to 'name' before adding the extension
        flair_path = os.path.join(base_dir, name + '-t2f.nii.gz')
        t1_path = os.path.join(base_dir, name + '-t1n.nii.gz')
        t1ce_path = os.path.join(base_dir, name + '-t1c.nii.gz')
        t2_path = os.path.join(base_dir, name + '-t2w.nii.gz')
        binary_path = os.path.join(base_dir, name + '_binaryseg.nii.gz')
        mask_path = os.path.join(base_dir, name + '-seg.nii.gz')
        # Load the imaging data using the nib_load function and convert to float32
        flair = np.array(nib_load(flair_path), dtype='float32')
        t1 = np.array(nib_load(t1_path), dtype='float32')
        t1ce = np.array(nib_load(t1ce_path), dtype='float32')
        t2 = np.array(nib_load(t2_path), dtype='float32')
        binary_label = np.array(nib_load(binary_path), dtype='float32')
        mask = np.array(nib_load(mask_path), dtype='float32')

        flair = flair[np.newaxis, ...]
        t1 = t1[np.newaxis, ...]
        t1ce = t1ce[np.newaxis, ...]
        t2 = t2[np.newaxis, ...]
        binary_label = binary_label[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        transforms = EnsureChannelFirstd(keys=['flair', 't1', 't1ce', 't2', 'binary_label' ,'label'], channel_dim=0)

        # mask[mask == 3] = 4
        item = {'flair': flair, 't1': t1, 't1ce': t1ce, 't2': t2, 'binary_label': binary_label, 'label': mask}
        item = self.transforms(item)

        if self.mode == 'train':  # train
            item = item[0]  # [0] for RandCropByPosNegLabeld

        return item['image'], item['label'], index, name

    def __len__(self):
        return len(self.case_names)
# import random
# from torch.utils.data import Sampler
#
# class FixedRatioBatchSampler(Sampler):
#     def __init__(self, len_ds1, len_ds2, batch_size, ratio_ds1=0.25):
#         self.len_ds1 = len_ds1
#         self.len_ds2 = len_ds2
#         self.batch_size = batch_size
#
#         self.n1 = int(batch_size * ratio_ds1)
#         self.n2 = batch_size - self.n1
#
#         assert self.n1 > 0, "Dataset-1 contributes zero samples"
#
#     def __iter__(self):
#         idx1 = list(range(self.len_ds1))
#         idx2 = list(range(self.len_ds2))
#
#         random.shuffle(idx1)
#         random.shuffle(idx2)
#
#         p1, p2 = 0, 0
#
#         while p2 + self.n2 <= self.len_ds2:
#             if p1 + self.n1 > self.len_ds1:
#                 random.shuffle(idx1)
#                 p1 = 0  # repeat small dataset
#
#             batch1 = idx1[p1:p1 + self.n1]
#             batch2 = idx2[p2:p2 + self.n2]
#
#             # shift dataset-2 indices
#             batch2 = [i + self.len_ds1 for i in batch2]
#
#             yield batch1 + batch2
#
#             p1 += self.n1
#             p2 += self.n2
#
#     def __len__(self):
#         return self.len_ds2 // self.n2


####################################################################################################
# dataloaders
# def worker_init_fn(worker_id):
#     torch.manual_seed(42)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)
def get_meta_loader(args, case_names: list):
    train_transforms = get_meta_brats2021_train_transform(args,seed)
    train_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='train',
        case_names=case_names,
        transforms=train_transforms)

    return DataLoader(train_dataset, batch_size=16, shuffle=True,
                      drop_last=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g,persistent_workers=True)
def get_meta_loader2(args, case_names: list):
    train_transforms = get_meta_brats2021_train_transform(args,seed)
    train_dataset = Adults_BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='train',
        case_names=case_names,
        transforms=train_transforms)

    return DataLoader(train_dataset, batch_size=16, shuffle=True,
                      drop_last=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g,persistent_workers=True)

def get_pediatric_loader(args, pediatric_case_names: list):
    """
    Returns a DataLoader for pediatric training cases.
    """
    train_transforms = get_brats2021_train_transform(args, seed)
    pediatric_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='train',
        case_names=pediatric_case_names,
        transforms=train_transforms)
    return DataLoader(pediatric_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g, persistent_workers=True)

def get_train_loader(args, case_names: list):
    train_transforms = get_brats2021_train_transform(args, seed)
    train_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='train',
        case_names=case_names,
        transforms=train_transforms)

    return train_dataset


def Adults_get_train_loader(args, case_names: list):
    train_transforms = get_brats2021_train_transform(args, seed)
    train_dataset = Adults_BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='train',
        case_names=case_names,
        transforms=train_transforms)

    return train_dataset

from torch.utils.data import DataLoader, ConcatDataset

# def create_combined_loader(args, train_cases1, train_cases2, ratio_ds1=0.25):
#     # datasets
#     dataset1 = get_train_loader(args, train_cases1)
#     dataset2 = Adults_get_train_loader(args, train_cases2)
#
#     combined_dataset = ConcatDataset([dataset1, dataset2])
#
#     batch_sampler = FixedRatioBatchSampler(
#         len_ds1=len(dataset1),
#         len_ds2=len(dataset2),
#         batch_size=args.batch_size,
#         ratio_ds1=ratio_ds1
#     )
#
#     train_loader = DataLoader(
#         combined_dataset,
#         batch_sampler=batch_sampler,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#         generator=g,
#         persistent_workers=True
#     )
#
#     return train_loader

def create_combined_loader(args, train_cases1, train_cases2):
    # Get datasets
    train_dataset1 = get_train_loader(args, train_cases1)
    train_dataset2 = Adults_get_train_loader(args, train_cases2)

    # Combine datasets
    combined_dataset = ConcatDataset([train_dataset1, train_dataset2])

    # Create a single DataLoader from the combined dataset
    train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=args.num_workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g, persistent_workers=True)

    return train_loader
# #

def get_infer_loader(args, case_names: list):
    infer_transform = get_brats2021_infer_transform(args,seed)
    infer_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='infer',
        case_names=case_names,
        transforms=infer_transform)

    return DataLoader(infer_dataset, batch_size=args.infer_batch_size, shuffle=False,
                      drop_last=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g,persistent_workers=True)



# def get_clean_train_loader(args, case_names: list):
#
#     train_transforms = get_brats2021_train_transform(args)
#     train_dataset = BraTS2021Dataset(
#         data_root=os.path.join(args.clean_data_root, args.dataset),
#         mode='train',
#         case_names=case_names,
#         transforms=train_transforms)
#
#     return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
#                       drop_last=False, num_workers=args.num_workers, pin_memory=True)

def get_clean_train_loader(args, case_names: list):
    train_transforms = get_brats2021_train_transform(args,seed)
    train_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.clean_data_root, args.dataset),
        mode='train',
        case_names=case_names,
        transforms=train_transforms)

    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                      drop_last=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g,persistent_workers=True)


# def Adults_get_train_loader(args, case_names: list):
#     train_transforms = get_brats2021_train_transform(args,seed)
#     train_dataset = Adults_BraTS2021Dataset(
#         data_root=os.path.join(args.data_root, args.dataset),
#         mode='train',
#         case_names=case_names,
#         transforms=train_transforms)
#
#     return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
#                       drop_last=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g,persistent_workers=True)