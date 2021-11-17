import sys

import h5py
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class Loader_dataset(data.Dataset):
    def __init__(self, path, tranforms):
        self.path = path
        hdf = h5py.File(self.path, 'r')
        self.length = len(hdf['labels'])   # <KeysViewHDF5 ['images', 'labels']>
        self.transform = tranforms
        hdf.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hdf = h5py.File(self.path, 'r')
        y = hdf['labels'][idx]
        data_pil = Image.fromarray(hdf['images'][idx, :, :, :].astype('uint8'), 'RGB')
        hdf.close()
        data = self.transform(data_pil)
        return data, torch.tensor(y).long().squeeze()-1

def get_pacs_loaders(args, client):
    path_root = '../data/PACS/'
    trans0 = transforms.Compose([transforms.RandomResizedCrop(222, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),  # 0.3的概率随机旋转，默认为0.5
                                 # transforms.RandomRotation((-5, 5)),
                                 transforms.RandomGrayscale(), #! 这个真的很重要了！居然能提高准确率
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([222, 222]),  # 虽然尺寸会变小，但不会裁剪图片
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    for i in range(3):
        train_path[i] = path_root + client[i] + '_train.hdf5'
        train_datas[i] = Loader_dataset(path=train_path[i], tranforms=trans0)
        train_loaders[i] = DataLoader(train_datas[i], args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
        valid_path[i] = path_root + client[i] + '_val.hdf5'
        valid_datas[i] = Loader_dataset(path=valid_path[i], tranforms=trans1)
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
    # target_path = path_root + client[3] + '_train.hdf5'
    target_path = path_root + client[3] + '_test.hdf5'
    target_data = Loader_dataset(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
    return train_loaders, valid_loaders, target_loader