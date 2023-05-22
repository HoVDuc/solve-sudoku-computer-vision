import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import Resize, Compose, ColorJitter, RandomRotation, ToTensor, GaussianBlur


class DataLoader:

    def __init__(self, path) -> None:
        self.path = path

    def load_data(self):
        transform = Compose([Resize(32),
                             RandomRotation(20),
                             GaussianBlur(3),
                             ColorJitter(hue=.05, saturation=.05),
                             ToTensor()])
        dataset = datasets.ImageFolder(self.path, transform=transform)
        return dataset

    def split_data(self, dataset, test_size):
        n_sample = len(dataset)
        n_val = int(n_sample * test_size)
        n_train = n_sample - n_val
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val])
        return train_set, val_set

    def dataloader(self, dataset, batch_size=32, shuffle=True):
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle)
        return dataloader

    def display_data(self, batch_data):
        plt.figure(figsize=(10, 7))
        for i in range(len(batch_data[0])):
            plt.subplot(8, 8, i+1)
            plt.imshow(batch_data[0][i].numpy().transpose(1, 2, 0))
            plt.title(batch_data[1][i])
        plt.show()


if __name__ == "__main__":
    PATH = "./train_data/dataset/"
    dataload = DataLoader(PATH)
    dataset = dataload.load_data()
    dataload.kfold_split(dataset)
    # train_set, val_set = dataload.split_data(dataset, 0.2)
    # train_loader = dataload.dataloader(train_set)
    # val_loader = dataload.dataloader(val_set)
    # print(len(train_set))
    # print(len(val_set))
    # test = next(iter(train_loader))
    # dataload.display_data(test)

    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                      batch_size=32,
    #                                      shuffle=True)
