# -*- coding: utf-8 -*-
"""Datasets and the three evaluation protocols.

Protocol 'bam'   : BAMPolyp's per-dataset split, reproduced exactly:
                   sorted(listdir(images)), np.random.seed(42) permutation,
                   80 / 10 / 10 -> train / val / test.
Protocol 'pranet': the standard benchmark (PraNet, MICCAI'20): train on the
                   1450 Kvasir+ClinicDB images, test on Kvasir (100),
                   CVC-ClinicDB (62), CVC-ColonDB (380), ETIS (196), CVC-300 (60).

Two training pipelines:
  'bam'  : BAMPolyp's own PIL augmentation + resize (verbatim semantics).
  'ours' : images cached once at training size, then flips / rot90 / affine /
           colour jitter / blur with OpenCV.
Every eval sample also carries its full-resolution mask, so metrics can be
computed both at the network size (BAMPolyp's metric) and at native size.
"""
from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMG_EXT = ('.jpg', '.png', '.jpeg')

# (glob pattern for the image dir, mask dir name relative to its parent)
BAM_SOURCES = {
    'kvasir':   ('**/Kvasir-SEG/images', 'masks'),
    'clinicdb': ('**/PNG/Original', 'Ground Truth'),
    'colondb':  ('**/CVC-ColonDB/images', 'masks'),
    'polypgen': ('**/PolypGen2021_MultiCenterData_v3/positive/images', 'masks'),
}
PRANET_TESTS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']


def _find_dir(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f'{pattern} not found under {root}')
    return hits[0]


def _mask_for(img_name, mask_dir, mask_files):
    """BAMPolyp's lookup: same stem with a known extension, else prefix match."""
    base = os.path.splitext(img_name)[0]
    for ext in ('.jpg', '.png', '.jpeg', '.tif'):
        p = os.path.join(mask_dir, base + ext)
        if os.path.exists(p):
            return p
    hits = [f for f in mask_files if f.startswith(base)]
    if hits:
        return os.path.join(mask_dir, hits[0])
    raise FileNotFoundError(f'no mask for {img_name}')


def bam_split(data_root, key):
    """Returns {'train'|'val'|'test': [(img_path, mask_path), ...]} exactly as BAMPolyp."""
    pattern, mask_name = BAM_SOURCES[key]
    img_dir = _find_dir(data_root, pattern)
    mask_dir = os.path.join(os.path.dirname(img_dir), mask_name)
    images = sorted(f for f in os.listdir(img_dir) if f.endswith(IMG_EXT))
    rng = np.random.RandomState(42)              # == np.random.seed(42); np.random.permutation
    idx = rng.permutation(len(images))
    n = len(images)
    parts = {'train': idx[:int(0.8 * n)], 'val': idx[int(0.8 * n):int(0.9 * n)],
             'test': idx[int(0.9 * n):]}
    mask_files = os.listdir(mask_dir)
    return {k: [(os.path.join(img_dir, images[i]), _mask_for(images[i], mask_dir, mask_files))
                for i in v] for k, v in parts.items()}


def pranet_split(data_root):
    tr_img = _find_dir(data_root, '**/TrainDataset/image')
    tr_msk = os.path.join(os.path.dirname(tr_img), 'masks')
    names = sorted(f for f in os.listdir(tr_img) if f.endswith(IMG_EXT))
    out = {'train': [(os.path.join(tr_img, f), os.path.join(tr_msk, f)) for f in names]}
    for t in PRANET_TESTS:
        d = _find_dir(data_root, f'**/TestDataset/{t}/images')
        m = os.path.join(os.path.dirname(d), 'masks')
        mf = os.listdir(m)
        out[f'test_{t}'] = [(os.path.join(d, f), _mask_for(f, m, mf))
                            for f in sorted(os.listdir(d)) if f.endswith(IMG_EXT)]
    return out


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _read_rgb(p):
    return np.array(Image.open(p).convert('RGB'))


def _read_mask(p):
    return (np.array(Image.open(p).convert('L')) > 127).astype(np.uint8)


def _normalize(img_u8):
    x = (img_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class EvalSet(Dataset):
    """Image resized to `size` (bilinear, PIL - as BAMPolyp), mask at `size`
    (nearest, >127 - as BAMPolyp) and at native resolution."""

    def __init__(self, pairs, size=256):
        self.pairs, self.size = pairs, size
        self.cache = [None] * len(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        if self.cache[i] is None:
            ip, mp = self.pairs[i]
            img = Image.open(ip).convert('RGB')
            msk = Image.open(mp).convert('L')
            x = TF.normalize(TF.to_tensor(TF.resize(img, (self.size, self.size))),
                             MEAN.tolist(), STD.tolist())
            m_net = (TF.to_tensor(TF.resize(msk, (self.size, self.size),
                                            interpolation=T.InterpolationMode.NEAREST)) > 0.5)
            m_full = torch.from_numpy((np.array(msk) > 127).astype(np.uint8))
            self.cache[i] = (x, m_net[0].to(torch.uint8), m_full, os.path.basename(ip))
        return self.cache[i]


def eval_collate(batch):
    x = torch.stack([b[0] for b in batch])
    m = torch.stack([b[1] for b in batch])
    return x, m, [b[2] for b in batch], [b[3] for b in batch]


class BamTrainSet(Dataset):
    """BAMPolyp's KvasirSEGDataset.__getitem__ (train branch), verbatim semantics."""

    def __init__(self, pairs, size=256):
        self.pairs = pairs
        self.transform = T.Compose([T.Resize((size, size)), T.ToTensor(),
                                    T.Normalize(MEAN.tolist(), STD.tolist())])
        self.target_transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        image = Image.open(ip).convert('RGB')
        mask = Image.open(mp).convert('L')
        if random.random() > 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() > 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image, mask = TF.rotate(image, angle, fill=0), TF.rotate(mask, angle, fill=0)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        x = self.transform(image)
        m = (self.target_transform(mask) > 0.5).long()[0]
        return x, m


class OursTrainSet(Dataset):
    """Cached at `cache_size` (>= train size so random zoom keeps detail)."""

    def __init__(self, pairs, size=256, cache_size=None):
        self.size = size
        cs = cache_size or int(round(size * 1.25))
        self.imgs, self.msks = [], []
        for ip, mp in pairs:
            self.imgs.append(cv2.resize(_read_rgb(ip), (cs, cs), interpolation=cv2.INTER_AREA))
            self.msks.append(cv2.resize(_read_mask(mp), (cs, cs), interpolation=cv2.INTER_NEAREST))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, msk = self.imgs[i], self.msks[i]
        cs, s = img.shape[0], self.size
        # random affine: rotation, zoom, shift; output directly at training size
        ang = random.uniform(-30, 30)
        zoom = random.uniform(0.8, 1.25) * s / cs
        M = cv2.getRotationMatrix2D((cs / 2, cs / 2), ang, zoom)
        M[:, 2] += (s - cs) / 2 + np.array([random.uniform(-0.1, 0.1) * s, random.uniform(-0.1, 0.1) * s])
        img = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        msk = cv2.warpAffine(msk, M, (s, s), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if random.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        k = random.randint(0, 3)
        img, msk = np.rot90(img, k), np.rot90(msk, k)
        img = np.ascontiguousarray(img)
        if random.random() < 0.8:                     # colour jitter in HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-5, 5)) % 180
            hsv[..., 1] *= random.uniform(0.75, 1.25)
            hsv[..., 2] *= random.uniform(0.75, 1.25)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            c = random.uniform(0.8, 1.2)
            img = np.clip((img.astype(np.float32) - img.mean()) * c + img.mean(), 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.5))
        return _normalize(img), torch.from_numpy(np.ascontiguousarray(msk)).long()
