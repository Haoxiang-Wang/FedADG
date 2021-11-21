from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init

# distribution generator
class GeneDistrNet(nn.Module):
    def __init__(self, input_size, hidden_size, optimizer,lr,momentum,weight_decay):
        super(GeneDistrNet,self).__init__()
        self.num_labels = 7
        self.genedistri = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_size+self.num_labels,4096)),
            ("relu1",nn.LeakyReLU()),

            ("fc2",nn.Linear(4096, hidden_size)),
            ("relu2",nn.ReLU()),
        ]))
        self.optimizer = optimizer(self.genedistri.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight,0.5)

    def forward(self,y,x,device):
        x = torch.cat([x,y], dim=1)
        x = x.to(device)
        x = self.genedistri(x)
        return x

# feature extractor
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
# classifier
class task_classifier(nn.Module):
    def __init__(self, hidden_size, optimizer, lr, momentum, weight_decay, class_num=5):
        super(task_classifier,self).__init__()
        self.task_classifier = nn.Sequential()
        self.task_classifier.add_module('t1_fc1', nn.Linear(hidden_size, hidden_size))
        self.task_classifier.add_module('t1_fc2', nn.Linear(hidden_size, class_num))
        self.optimizer = optimizer(self.task_classifier.parameters(),
                                   lr=lr, momentum=momentum, weight_decay=weight_decay)
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
        return y

# discriminator
class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_labels, rp_size):
        super().__init__()
        rp_to_size=rp_size
        self.features_pro = nn.Sequential(
            nn.Linear(rp_size+num_labels, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        self.projection = nn.Linear(hidden_size, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

        self.optimizer = None
        self.initialize_params()


    def forward(self, y, z):
        feature = z.view(z.size(0), -1)
        feature_proj = self.projection(feature)
        feature_proj = torch.cat([feature_proj,y],dim=1)
        logit = self.features_pro(feature_proj)
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