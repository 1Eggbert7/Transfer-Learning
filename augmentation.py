# +
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

import random
import os
import shutil

def get_split_data(directory, ds_percent):
    dest_folder = "sub"
    os.makedirs(dest_folder, exist_ok=True)
    
    for folder in os.listdir(dest_folder):
        f_path = os.path.join(dest_folder, folder)
        for file in os.listdir(f_path):
            os.remove(os.path.join(f_path, file))
        os.rmdir(f_path)
    
    for sub_name in os.listdir(directory):
        sub_path = os.path.join(directory, sub_name)
        
        if os.path.isdir(sub_path):
            dest_sub_folder = os.path.join(dest_folder, sub_name)
            os.makedirs(dest_sub_folder, exist_ok=True)
            
            files = os.listdir(sub_path)
            num = int(len(files) * ds_percent)
            selection = random.sample(files, num)
            
            for fn in selection:
                source = os.path.join(sub_path, fn)
                dest = os.path.join(dest_sub_folder, fn)
                shutil.copy2(source, dest)
    
    return dest_folder



def augmented_ds(data_path,
                 flip_h=True,
                 flip_v=True,
                 color=True, num_color=1,
                 crop=True, num_crop=1,
                 rotate=True, num_rotate=1,
                 scale=True, num_scale=1,
                 ds_percent=0.1
                ):
    
    
    all_tf = []
    sub_path = get_split_data(data_path, ds_percent)

    # always apply basic transform
    basic_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.485, 0.56, 0.406], [0.229, 0.224, 0.225]
        )
    ])
    basic_ds = datasets.ImageFolder(data_path, basic_tf)
    all_tf.append(basic_ds)
    
    # horizontal flip
    if flip_h:
        flip_h_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        flip_h_ds = datasets.ImageFolder(sub_path, flip_h_tf)
        all_tf.append(flip_h_ds)
        
    # vertical flip
    if flip_v:
        flip_v_tf = transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        flip_v_ds = datasets.ImageFolder(sub_path, flip_v_tf)
        all_tf.append(flip_v_ds)
        
    # random color augmentations
    if color:
        color_ds = []
        for i in range(num_color):
            color_tf = transforms.Compose([
                transforms.ColorJitter(
                    brightness=(0, 1),
                    contrast=(0, 1),
                    saturation=(0, 1),
                    hue=(-0.5, 0.5)
                ),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            color_ds.append(datasets.ImageFolder(sub_path, color_tf))
        all_tf.extend(color_ds)
    
    # random crops
    if crop:
        crop_ds = []
        for i in range(num_crop):
            crop_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(size=(224,224)),
                transforms.ToTensor()
            ])
            crop_ds.append(datasets.ImageFolder(sub_path, crop_tf))
        all_tf.extend(crop_ds)
    
    # random rotations
    if rotate:
        rotate_ds = []
        for i in range(num_rotate):
            rotate_tf = transforms.Compose([
                transforms.RandomRotation(degrees=random.randint(0, 360)),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            rotate_ds.append(datasets.ImageFolder(sub_path, rotate_tf))
        all_tf.extend(rotate_ds)
        
    # random scaling
    # scale
    if scale:
        scale_ds = []
        for i in range(num_scale):
            scale_tf = transforms.Compose([
                transforms.Resize(random.randint(256, 350)),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            scale_ds.append(datasets.ImageFolder(sub_path, scale_tf))
        all_tf.extend(scale_ds)
        
    # concatenate to one dataset
    total_ds = torch.utils.data.ConcatDataset(all_tf)
    
    return total_ds
        
def random_augmentations(data_path, num=1, prob=0.5, ds_percent=0.1):
    # always apply basic transform
    
    all_tf = []
    basic_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.485, 0.56, 0.406], [0.229, 0.224, 0.225]
        )
    ])
    basic_ds = datasets.ImageFolder(data_path, basic_tf)
    all_tf.append(basic_ds)
    
    sub_path = get_split_data(data_path, ds_percent)
 
    for i in range(num):
        comp = []
        if random.random() < prob:
            comp.append(transforms.RandomHorizontalFlip(p=1))
        if random.random() < prob:
            comp.append(transforms.RandomVerticalFlip(p=1))
        if random.random() < prob:
            comp.append(transforms.ColorJitter(
                brightness=(0,1),
                contrast=(0,1),
                saturation=(0,1),
                hue=(-0.5,0.5)
            ))
        if random.random() < prob:
            comp.append(transforms.Resize(256))
            comp.append(transforms.RandomCrop(size=(224,224)))
        if random.random() < prob:
            comp.append(transforms.RandomRotation(degrees=random.randint(0,360)))
        if random.random() < prob:
            comp.append(transforms.Resize(random.randint(256,360)))
            comp.append(transforms.CenterCrop(224))

        comp.append(transforms.Resize(256))
        comp.append(transforms.CenterCrop(224))
        comp.append(transforms.ToTensor())

        tf = transforms.Compose(comp)
        all_tf.append(datasets.ImageFolder(sub_path, tf))
    
    total_ds = torch.utils.data.ConcatDataset(all_tf)
    
    return total_ds
    
    
# -

    
