import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


class Dataset_noniid(Dataset):
    def __init__(self, data, targets):
        super(Dataset_noniid, self).__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], int(self.targets[index].item())


def get_dataloader_train(batch_size, alpha, n_clients, flag, datasetname, device):
    # 这里的读取有点浪费时间
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    if datasetname == "MNIST":
        dataset = torchvision.datasets.MNIST(root='../',
                                             train=True,download=True,
                                             transform=transform)
    elif datasetname == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root='../',
                                               train=True,download=True,
                                               transform=transform)
    else:
        # 其他数据集暂无
        dataset = None
        assert dataset is not None, "暂无其他数据集"
    labels = np.array(dataset.targets)
    split_idx = split_noniid(labels, alpha=alpha, n_clients=n_clients)
    data_loader = []

    if flag:
        # 绘图
        plt.figure(figsize=(20, 3))
        plt.hist([labels[idc] for idc in split_idx], stacked=True,
                 bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
                 label=["Client {}".format(i + 1) for i in range(n_clients)], rwidth=0.5)
        plt.xticks(np.arange(len(dataset.classes)), dataset.classes)
        plt.legend()
        plt.savefig("data_distribution.png")

    # 这里应该可以继续优化
    sz = [len(dataset)]
    targets = torch.zeros(sz).to(device)
    sz += [i for i in dataset[0][0].size()]
    feature = torch.zeros(sz).to(device)
    for i, v in enumerate(dataset):
        feature[i] = v[0]
        targets[i] = v[1]
    client_nums = []
    total = 0
    for i in split_idx:
        total += len(i)
        client_nums.append(len(i))
        data_loader.append(
            DataLoader(dataset=Dataset_noniid(feature[i], targets[i]),
                       batch_size=batch_size,
                       shuffle=True), )
    return data_loader, client_nums, total


def get_dataloader_test(batch_size, datasetname):
    # 测试集
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    if datasetname == "MNIST":
        dataset = torchvision.datasets.MNIST(root='../', train=False, transform=transform)
    elif datasetname == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root="../", train=False, transform=transform)
    else:
        # 其他数据集暂无
        dataset = None
        assert dataset is not None, "暂无其他数据集"
    return DataLoader(dataset=dataset, batch_size=batch_size)


def split_noniid(train_labels, alpha, n_clients):
    # 将数据集切分成Non-iid, alpha控制程度，越高代表越均匀
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs

