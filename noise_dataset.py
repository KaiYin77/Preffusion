from torch.utils.data import Dataset
import torch


class NoiseDataset(Dataset):
    def __init__(self, len=6):
        self.len = len
        self.x = torch.randn(len, 1, 60, 4)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]


if __name__ == '__main__':
    dataset = NoiseDataset(1)
    for i in dataset:
        print(i.shape)
