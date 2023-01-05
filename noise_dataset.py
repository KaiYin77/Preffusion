from torch.utils.data import Dataset
import torch


class NoiseDataset(Dataset):
    def __init__(self, len=100):
        self.len = len
        self.x = torch.randn(len, 3, 32, 32)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]


if __name__ == '__main__':
    # test()
    dataset = NoiseDataset(10)
    for i in dataset:
        print(i.shape)
