from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init


class Bottleneck(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_downsample=False):
        super(Bottleneck, self).__init__()
        filter1, filter2, filter3 = filters
        self.conv1 = nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(filter1)
        self.conv2 = nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filter2)
        self.conv3 = nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filter3)
        self.relu = nn.ReLU(inplace=True)
        self.is_downsample = is_downsample
        self.parameters()
        if is_downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(filter3))

    def forward(self, X):
        X_shortcut = X
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
 
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)

        X = self.conv3(X)
        X = self.bn3(X)

        if self.is_downsample:
            X_shortcut = self.downsample(X_shortcut)

        X = X + X_shortcut
        X = self.relu(X)
        return X

class ResNet50Model(nn.Module):
    def __init__(self):
        super(ResNet50Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, (64, 64, 256), 3)
        self.layer2 = self._make_layer(256, (128, 128, 512), 4, 2)
        self.layer3 = self._make_layer(512, (256, 256, 1024), 6, 2)
        self.layer4 = self._make_layer(1024, (512, 512, 2048), 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(2048, 1000)
        self.optimizer = None

    def forward(self, input):
        # print("--ResNetModel_1--forward--input.shape={}".format(input.shape))
        X = self.conv1(input)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)

        X = self.avgpool(X)
        X = torch.flatten(X, 1)  # 如果不加这一层 输出就是[batch_size,hidden_size,1,1],加上之后就是[bat_size,hidden_size]
        # X = self.fc(X)
        return X

    def _make_layer(self, in_channels, filters, blocks, stride=1):
        layers = []
        block_one = Bottleneck(in_channels, filters, stride=stride, is_downsample=True)
        layers.append(block_one)
        for i in range(1, blocks):
            layers.append(Bottleneck(filters[2], filters, stride=1, is_downsample=False))

        return nn.Sequential(*layers)


# 生成随机分布 -包括优化器
class GeneDistrNet(nn.Module):
    def __init__(self,input_size,hidden_size, optimizer,lr,momentum,weight_decay):
        super(GeneDistrNet,self).__init__()
        self.num_labels = 7
        # self.gene = nn.Sequential(OrderedDict([
        #     ("fc1", nn.Linear(input_size, input_size)),
        #     ("drop1", nn.Dropout()),
        #     # ("relu1", nn.LeakyReLU()),
        #     ("relu1", nn.ReLU()),
        #
        #     # ("fc2", nn.Linear(4608, 4608)),
        #     # ("drop2", nn.Dropout()),
        #     # ("relu2", nn.ReLU()),
        #     ("fc2", nn.Linear(input_size, hidden_size)),
        #
        #     # add part
        #     # ("drop2", nn.Dropout()),
        #     # ("relu2",nn.LeakyReLU()),
        #     # ("fc3", nn.Linear(hidden_size, hidden_size)),
        #     # ("relu3",nn.ReLU()),
        # ]))
        self.genedistri = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_size+self.num_labels,4096)),
            # ("fc1", nn.Linear(input_size,4096)),
            ("relu1",nn.LeakyReLU()),

            ("fc2",nn.Linear(4096, hidden_size)),
            ("relu2",nn.ReLU()),
        ]))
        self.optimizer = optimizer(self.genedistri.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.G_merge_y = nn.Linear(self.num_labels, input_size,bias=False)
        self.G_merge_z = nn.Linear(input_size, input_size,bias=False)
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                # layer.weight.data.fill_(1)
                # layer.bias.data.zero_()
                init.xavier_uniform_(layer.weight,0.5)
                # layer.bias.data.zero_()

    def forward(self,y,x,device):
        # x = self.G_merge_y(y)+self.G_merge_z(x)
        x = torch.cat([x,y], dim=1)
        x = x.to(device)
        x = self.genedistri(x)
        # x = self.gene(x)
        return x

class feature_extractor(nn.Module):
    def __init__(self, optimizer,lr,momentum,weight_decay, num_classes=5):
        super(feature_extractor,self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(OrderedDict([
            ("conv1",nn.Conv2d(3,96,kernel_size=11,stride=4)),
            ("relu1",nn.ReLU(inplace=True)),
            ("pool1",nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)),
            ("norm1",nn.LocalResponseNorm(5,1.e-4,0.75)),

            ("conv2",nn.Conv2d(96,256,kernel_size=5,padding=2,groups=2)),
            ("relu2",nn.ReLU(inplace=True)),
            ("pool2",nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)),
            ("norm2",nn.LocalResponseNorm(5,1.e-4,0.75)),

            ("conv3",nn.Conv2d(256,384,kernel_size=3,padding=1)),
            ("relu3",nn.ReLU(inplace=True)),

            ("conv4",nn.Conv2d(384,384,kernel_size=3,padding=1,groups=2)),
            ("relu4",nn.ReLU(inplace=True)),

            ("conv5",nn.Conv2d(384,256,kernel_size=3,padding=1,groups=2)),
            ("relu5",nn.ReLU(inplace=True)),
            ("pool5",nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout()),

            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout())
        ]))

        self.optimizer = optimizer(list(self.features.parameters())+list(self.classifier.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight,0.1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.features(x*57.6)
        x = x.view((x.size(0),256*6*6))
        x = self.classifier(x)
        return x

class task_classifier(nn.Module):
    def __init__(self, hidden_size, optimizer, lr, momentum, weight_decay, class_num=5):
        super(task_classifier,self).__init__()
        self.task_classifier = nn.Sequential()
        self.task_classifier.add_module('t1_fc1', nn.Linear(hidden_size, hidden_size))
        self.task_classifier.add_module('t1_fc2', nn.Linear(hidden_size, class_num))
        self.optimizer = optimizer(self.task_classifier.parameters(),
                                   lr=lr, momentum=momentum, weight_decay=weight_decay)
        # self.optimizer = optim.Adam(self.task_classifier.parameters(),lr =lr,weight_decay=weight_decay,amsgrad=True)
        self.initialize_paras()

    def initialize_paras(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight,a=0,mode='fan-out')
            elif isinstance(layer,torch.nn.Linear):
                init.kaiming_normal_(layer.weight)
            elif isinstance(layer,torch.nn.BatchNorm2d) or isinstance(layer,torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = torch.flatten(x, 1)
        y = self.task_classifier(x)
        # y = self.task_classifier(x).view(x.size(0),-1)
        return y

class domain_discriminator(nn.Module):
    def __init__(self,hidden_size, rp_size,optimizer,lr,momentum,weight_decay):
        super(domain_discriminator,self).__init__()
        self.domain_discriminator = nn.Sequential(OrderedDict([
            ("d_fc1", nn.Linear(rp_size, 1024)),
            ("d_relu1",nn.ReLU()),
            ("d_drop1",nn.Dropout()),
            # # ("d_fc2", nn.Linear(1024, 1024)),
            # # ("d_relu2", nn.ReLU(True)),
            ("d_fc2",nn.Linear(1024, 1)),

            # 参考AAE
            # ("d_fc1",nn.Linear(rp_size,1024)),
            # ("d_drop1",nn.Dropout()),
            # ("d_relu1",nn.ReLU()),
            # ("d_fc2", nn.Linear(1024, 1024)),
            # ("d_drop2", nn.Dropout()),
            # ("d_relu2", nn.ReLU()),
            # ("d_fc3",nn.Linear(1024,1)),
        ]))

        self.optimizer = optimizer(self.domain_discriminator.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.initial_params()
        # self.projection = nn.Linear(256*6*6,rp_size,bias=False)
        self.projection = nn.Linear(hidden_size, rp_size,bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight,keepdim=True))

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight,a=0,mode='fan_out')
            elif isinstance(layer,torch.nn.Linear):
                init.kaiming_normal_(layer.weight)
            elif isinstance(layer,torch.nn.BatchNorm2d) or isinstance(layer,torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        feature = x.view(x.size(0),-1)
        feature_proj = self.projection(feature)
        y = self.domain_discriminator(feature_proj)
        # y = self.domain_discriminator(feature)
        # return y  # 如果不用sigmoid的话会出现负数，求损失函数为nan
        return torch.sigmoid(y)  # 如果不用sigmoid的话会出现负数，求损失函数为nan

class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_labels, rp_size):
        super().__init__()
        rp_to_size=rp_size
        # self.features = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1),
        #     nn.Sigmoid(),
        # )
        self.features_pro = nn.Sequential(
            # nn.Linear(rp_size, 1024),
            nn.Linear(rp_size+num_labels, 1024),
            # nn.Linear(hidden_size, rp_size),
            # nn.Linear(rp_size, rp_to_size),
            # nn.Linear(hidden_size, 1024),  # 用于 albation study
            # nn.ReLU(), # 加入num_label后就注释了
            nn.LeakyReLU(),
            # nn.Dropout(), # 加入num_label后就注释了
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        self.projection = nn.Linear(hidden_size, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

        self.optimizer = None
        self.initialize_params()


    def forward(self, y, z):
        # merge = self.D_merge_y(y) + self.D_merge_z(z)
        # new part
        # feature = merge.view(merge.size(0), -1)
        # feature_proj = self.projection(feature)

        # new part1
        feature = z.view(z.size(0), -1)
        # merge = self.projection(feature)
        feature_proj = self.projection(feature)
        feature_proj = torch.cat([feature_proj,y],dim=1)
        # feature_proj = torch.cat([feature,y], dim=1)
        # feature_proj = self.projection(feature_proj)
        # merge = feature_proj + self.D_merge_y1(y)
        logit = self.features_pro(feature_proj)

        # 不考虑混合onehot
        # logit = self.features_pro(feature_proj)
        return logit

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()
