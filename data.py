#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import pandas as pd
from tqdm import tqdm
from glob import glob


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def read_off(file):
    """Parse OFF file and return Nx3 array of vertices."""
    with open(file, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise ValueError('Not a valid OFF file')
        n_verts, n_faces, _ = map(int, f.readline().strip().split())
        verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]
    return np.array(verts, dtype=np.float32)

def load_data_from_csv(partition, csv_path, base_dir, num_points=1024):
    df = pd.read_csv(csv_path)
    df = df[df['split'] == partition]

    all_data = []
    all_label = []

    class_names = sorted(df['class'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Loading {partition} data'):
        file_path = os.path.join(base_dir, row['object_path'])
        cls = row['class']

        try:
            points = read_off(file_path)

            # Randomly sample num_points from mesh vertices
            if len(points) >= num_points:
                choice = np.random.choice(len(points), num_points, replace=False)
            else:
                choice = np.random.choice(len(points), num_points, replace=True)
            sampled = points[choice]

            all_data.append(sampled)
            all_label.append(class_to_idx[cls])

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return np.stack(all_data), np.array(all_label, dtype=np.int64)


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.label = load_data_from_csv(partition, 
                                                   csv_path='archive/metadata_modelnet40.csv',
                                                   base_dir='archive/ModelNet40')
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


class miniBRANCH(Dataset):
    def __init__(self, rgb_dir='miniBRANCH dataset/tree_1_V_0001/Filtered__noGrass/color', depth_dir='miniBRANCH dataset/tree_1_V_0001/Filtered__noGrass/depth', num_points=1024, factor=4, partition='train'):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.rgb_images = sorted(os.listdir(rgb_dir))
        self.depth_images = sorted(os.listdir(depth_dir))
        self.num_points = num_points
        self.factor = factor
        self.partition = partition

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        # Load depth image and convert to point cloud
        depth_path = os.path.join(self.depth_dir, self.depth_images[idx])
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Optional: normalize depth
        depth /= depth.max()

        h, w = depth.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        zz = depth

        # Flatten and sample points
        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)

        # Filter out zero-depth points
        valid = zz.flatten() > 0
        points = points[valid]

        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
        pointcloud = points[indices]

        # Apply DCP-style transform
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(anglex), -np.sin(anglex)],
                       [0, np.sin(anglex), np.cos(anglex)]])
        Ry = np.array([[np.cos(angley), 0, np.sin(angley)],
                       [0, 1, 0],
                       [-np.sin(angley), 0, np.cos(angley)]])
        Rz = np.array([[np.cos(anglez), -np.sin(anglez), 0],
                       [np.sin(anglez), np.cos(anglez), 0],
                       [0, 0, 1]])

        R_ab = Rz.dot(Ry).dot(Rx)
        R_ba = R_ab.T

        t_ab = np.random.uniform(-0.5, 0.5, size=(3,))
        t_ba = -R_ba @ t_ab

        pointcloud1 = pointcloud.T
        pointcloud2 = R_ab @ pointcloud1 + t_ab.reshape(3, 1)

        euler_ab = np.array([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        # Optional: random permutation
        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype(np.float32), pointcloud2.astype(np.float32), \
               R_ab.astype(np.float32), t_ab.astype(np.float32), \
               R_ba.astype(np.float32), t_ba.astype(np.float32), \
               euler_ab.astype(np.float32), euler_ba.astype(np.float32)


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break
