# 由于数据集中每个用户使用一个数据集，所以数据均为iid，只需要将每个domain的数据集赋给用户即可
import copy
import numpy
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def data_iid(dataset,num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def load_FCparas(premodel="resnet"):
    if premodel == "alexnet":
        state_dict = torch.load("../data/alexnet_caffe.pth.tar")
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
    elif premodel == "resnet":
        state_dict = torch.load("../data/resnet50-19c8e357.pth")
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]

    return state_dict

class Loader_dataset(data.Dataset):
    def __init__(self, path, tranforms = None):
        self.path = path
        self.dataset = datasets.ImageFolder(path, transform=tranforms)
        self.length = self.dataset.__len__()
        self.transform = tranforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data, label = self.dataset.__getitem__(idx)
        return data, label

def get_loaders(args, client):
    path_root = args.path_root
    trans0 = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([225, 225]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    for i in range(3):
        train_path[i] = path_root + client[i] + '/train'
        train_datas[i] = Loader_dataset(path=train_path[i], tranforms=trans0)
        train_loaders[i] = DataLoader(train_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)

        valid_path[i] = path_root + client[i] + '/test'
        valid_datas[i] = Loader_dataset(path=valid_path[i], tranforms=trans1)
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    target_path = path_root + client[3] + '/test'
    target_data = Loader_dataset(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    return train_loaders, valid_loaders, target_loader

if __name__ == "__main__":
    data_1 = torch.tensor(np.random.normal(0, 10, (100, 50)))
    data_2 = torch.tensor(np.random.normal(10, 10, (100, 50)))

    print("MMD Loss:", mmd(data_1, data_2))

    data_1 = torch.tensor(np.random.normal(0, 10, (100, 50)))
    data_2 = torch.tensor(np.random.normal(0, 9, (100, 50)))

    print("MMD Loss:", mmd(data_1, data_2))