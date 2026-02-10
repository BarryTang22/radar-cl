"""Visualize the first sample from each of the 5 radar datasets."""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from datasets.single import (
    DATA_ROOT, get_mmdrive_folders, get_cube_folders, get_files_from_folders,
    CI4RDataset, DIATDataset
)


def load_dmm_sample():
    files = get_files_from_folders(get_mmdrive_folders('1'))
    data = np.load(files[0]).astype(np.float32)
    return np.log1p(data)  # (21, 77)


def load_drc_sample():
    files = get_files_from_folders(get_cube_folders('1'))
    data = np.load(files[0]).astype(np.float32)
    return np.log1p(data.sum(axis=(1, 2)))  # (T, V, H, R) -> (T, R)


def load_ci4r_sample():
    ds = CI4RDataset(os.path.join(DATA_ROOT, 'ci4r'), '77GHz', normalize='raw_batchnorm')
    tensor, _ = ds[0]
    return tensor.squeeze(0).numpy()  # (128, 128)


def load_radhar_sample():
    root = os.path.join(DATA_ROOT, 'radhar', 'Web_Radhar_Shared_Dataset')
    npz = np.load(os.path.join(root, 'Train_Data_voxels_boxing.npz'), allow_pickle=True)
    voxel = npz['arr_0'][0].astype(np.float32)  # first sample, (60, 10, 32, 32)
    mid = voxel.shape[0] // 2
    return np.log1p(voxel[mid].sum(axis=0))  # sum depth -> (32, 32)


def load_diat_sample():
    root = os.path.join(DATA_ROOT, 'diat', 'DIAT-RadHAR')
    for class_name in DIATDataset.CLASSES:
        class_dir = os.path.join(root, class_name)
        if os.path.isdir(class_dir):
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = Image.open(os.path.join(class_dir, f)).convert('L')
                    return np.array(img, dtype=np.float32)


if __name__ == '__main__':
    samples = [
        ('DMM (21×77)', load_dmm_sample, 'viridis'),
        ('DRC (T×R)', load_drc_sample, 'viridis'),
        ('CI4R (128×128)', load_ci4r_sample, 'viridis'),
        ('RadHAR (32×32)', load_radhar_sample, 'viridis'),
        ('DIAT (grayscale)', load_diat_sample, 'gray'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (title, loader, cmap) in zip(axes, samples):
        data = loader()
        ax.imshow(data, aspect='auto', cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle('First Sample per Dataset', fontsize=14)
    plt.tight_layout()
    plt.show()
