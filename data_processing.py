import torch
from torchvision import datasets, transforms


def load_images(dir, shuffle=True, batch_size=16):
    data = datasets.ImageFolder(dir, transform=preprocess())
    dataloader = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def preprocess(means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    prep = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    return prep


