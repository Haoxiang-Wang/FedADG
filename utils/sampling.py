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
    # state_dictC = copy.deepcopy(state_dict)
    # del state_dict["classifier.fc6.weight"]
    # del state_dict["classifier.fc6.bias"]
    # del state_dict["classifier.fc7.weight"]
    # del state_dict["classifier.fc7.bias"]
    # del state_dict["classifier.fc8.weight"]
    # del state_dict["classifier.fc8.bias"]
    #
    # del state_dictC["features.conv1.weight"]
    # del state_dictC["features.conv1.bias"]
    # del state_dictC["features.conv2.weight"]
    # del state_dictC["features.conv2.bias"]
    # del state_dictC["features.conv3.weight"]
    # del state_dictC["features.conv3.bias"]
    # del state_dictC["features.conv4.weight"]
    # del state_dictC["features.conv4.bias"]
    # del state_dictC["features.conv5.weight"]
    # del state_dictC["features.conv5.bias"]
    # del state_dictC["classifier.fc8.weight"]
    # del state_dictC["classifier.fc8.bias"]
    return state_dict
    # return state_dict, state_dictC

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

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 倍率参数
    kernel_num: 取不同高斯核的数量
    fix_sigma: 表示是否使用固定的标准差，不同高斯核的sigma值

		return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
						矩阵，表达形式:
						[	K_ss K_st
							K_ts K_tt ]
							多个核矩阵之和
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
    # 当不同的时候，就需要乘上上面的M矩阵
    return loss


def get_loaders(args, client):
    path_root = args.path_root
    trans0 = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),  # 0.3的概率随机旋转，默认为0.5
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([225, 225]),  # 虽然尺寸会变小，但不会裁剪图片
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
    #  计算两个分布的距离 data_1 & data_2
    data_1 = torch.tensor(np.random.normal(0, 10, (100, 50)))
    data_2 = torch.tensor(np.random.normal(10, 10, (100, 50)))

    print("MMD Loss:", mmd(data_1, data_2))

    data_1 = torch.tensor(np.random.normal(0, 10, (100, 50)))
    data_2 = torch.tensor(np.random.normal(0, 9, (100, 50)))

    print("MMD Loss:", mmd(data_1, data_2))