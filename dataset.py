from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np


def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]

    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])

    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()
    ])
    return data_transforms, gt_transforms


class MVTecDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')

        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Number of images and ground truths do not match"
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        return img, gt, label, img_type


class CAD_SD_Dataset(torch.utils.data.Dataset):

    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')

        self.transform = transform
        self.img_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        return img_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label, os.path.basename(img_path[:-4]), img_type



class VisADataset(torch.utils.data.Dataset):

    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')

        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        all_entries = os.listdir(self.img_path)
        defect_types = [d for d in all_entries if os.path.isdir(os.path.join(self.img_path, d))]

        for defect_type in defect_types:
            # 让代码可以同时查找 .jpg, .JPG, .jpeg, .JPEG 四种后缀名
            img_paths = []
            possible_exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
            for ext in possible_exts:
                img_paths.extend(glob.glob(os.path.join(self.img_path, defect_type, ext)))
            img_paths.sort() # 找到所有文件后统一排序

            if defect_type == 'good':
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                gt_paths = sorted(glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png"))
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

            img_tot_paths.extend(img_paths)


        assert len(img_tot_paths) == len(gt_tot_paths), f"VisADataset: Number of images ({len(img_tot_paths)}) and ground truths ({len(gt_tot_paths)}) do not match"
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)

        return img, gt, label, img_type


# class VisADataset(torch.utils.data.Dataset):
#
#
#     def __init__(self, root, transform, gt_transform, phase):
#         if phase == 'train':
#             self.img_path = os.path.join(root, 'train')
#         else:
#             self.img_path = os.path.join(root, 'test')
#             self.gt_path = os.path.join(root, 'ground_truth')
#
#         self.transform = transform
#         self.gt_transform = gt_transform
#         self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
#
#     def load_dataset(self):
#         img_tot_paths = []
#         gt_tot_paths = []
#         tot_labels = []
#         tot_types = []
#
#         all_entries = os.listdir(self.img_path)
#         defect_types = [d for d in all_entries if os.path.isdir(os.path.join(self.img_path, d))]
#
#         for defect_type in defect_types:
#             img_paths = []
#             possible_exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
#             for ext in possible_exts:
#                 img_paths.extend(glob.glob(os.path.join(self.img_path, defect_type, ext)))
#
#             img_paths = sorted(list(set(img_paths)))
#
#
#             if defect_type == 'good':
#                 gt_tot_paths.extend([0] * len(img_paths))
#                 tot_labels.extend([0] * len(img_paths))
#                 tot_types.extend(['good'] * len(img_paths))
#             else:
#                 gt_paths = sorted(glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png"))
#                 gt_tot_paths.extend(gt_paths)
#                 tot_labels.extend([1] * len(img_paths))
#                 tot_types.extend([defect_type] * len(img_paths))
#
#             img_tot_paths.extend(img_paths)
#
#         assert len(img_tot_paths) == len(
#             gt_tot_paths), f"VisADataset: Number of images ({len(img_tot_paths)}) and ground truths ({len(gt_tot_paths)}) do not match"
#         return img_tot_paths, gt_tot_paths, tot_labels, tot_types
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
#         img = Image.open(img_path).convert('RGB')
#         img = self.transform(img)
#         if gt == 0:
#             gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
#         else:
#             gt = Image.open(gt).convert('L')
#             gt = self.gt_transform(gt)
#         return img, gt, label, img_type