import numpy as np
import torch
import torchvision
from torchvision import datasets


def parse_line(line):
    parts = line.strip().split()
    label = int(float(parts[0]))

    one_hot = np.zeros(10, dtype=float)
    features = np.zeros(12, dtype=float)

    for part in parts[1:]:
        idx, value = part.split(":")
        idx = int(idx)
        value = float(value)
        if 1 <= idx <= 10:
            one_hot[idx - 1] = value
        else:
            features[idx - 11] = value

    return np.concatenate(([label], one_hot, features))


def process_file(file_path):
    with open(file_path, 'r') as file:
        data = [parse_line(line) or line in file]
        data = np.array(data)
        np.save(file_path, data)
    return data


def load_ijcnn1(file_path):
    """function to import and preprocess IJCNN dataset"""
    try:
        # if dataset already processed and saved as numpy array, array is loaded from npy
        data = np.load(file_path + ".npy")

    except FileNotFoundError:
        # if dataset not yet processed and saved as npy, dataset is loaded from text file and processed
        print("Array file not found. Will be constructed from txt")
        data = process_file(file_path)

    return data


def load_mnist(batch_size=32):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
