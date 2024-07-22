import numpy as np
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset


class BinaryClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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
        data = [parse_line(line) for line in file]
        data = np.array(data)
        np.save(file_path, data)
    return data


def load_ijcnn1_to_numpy(file_path):
    """
    Function to import and preprocess IJCNN1 dataset
    if dataset not yet saved as .npy, dataset will be saved in "file_path"
    :param file_path: path to file containing datasets
    :return: numpy array containing dataset
    """
    try:
        # if dataset already processed and saved as numpy array, array is loaded from npy
        data = np.load(file_path + ".npy")

    except FileNotFoundError:
        # if dataset not yet processed and saved as npy, dataset is loaded from text file and processed
        print("Array file not found. Will be constructed from txt")
        data = process_file(file_path)

    return data

def load_ijcnn1_to_dataloader(file_path, batch_size=32, shuffle=False):
    """
    Converts dataset for binary classification (specifically IJCNN1) to torch dataloader
    :param file_path: path to dataset
    :param batch_size: batch size (for processing of data, same meaning as for torch dtaloader object)
    :param shuffle: if True, processing order of data is shuffled to avoid overfitting, default False
    :return: torch dataloader
    """
    data = load_ijcnn1_to_numpy(file_path)
    labels = torch.tensor(data[:, 0], dtype=torch.float32)
    features = torch.tensor(data[:, 1:], dtype=torch.float32)
    torch_dataset = BinaryClassificationDataset(features, labels)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



def load_mnist(batch_size=32, shuffle=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    testloader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
