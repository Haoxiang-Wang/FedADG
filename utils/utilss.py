import torch
import math
from torch import nn
from scipy.special import binom
import torch.nn.functional as F


def compare_parameters(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        try:
            if p1.allclose(p2):
                print('equal')
            else:
                print('diff')

        except:
            print('diff layer')


# 标签平滑 考虑存在标签标注错误的情况--事实上在vlcs的数据集中确实存在此情况
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, lbl_set_size, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        # 标签标注正确的概率
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        # 分类的类别
        self.cls = lbl_set_size
        self.dim = dim  # 在softmax中 dim=1表示对每一行求softmax

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))  # 虽然判断为假，但是存在标签误判的概率。-1是因为有一个标签是为真的
            # 将真实标签所在的位置概率替换上
            true_dist.scatter_(1, target.data.unsqueeze(1),self.confidence)
            # scatter_dim=1时填充规则true[i][target[i][j]] = confi[i][j]
        # 因为是多张图片，所以对每一行计算完了之后再求平均，得到总的平均误差
        res = torch.sum(-true_dist * pred, dim=self.dim)
        return torch.mean(res)

class LabelSmoothingLoss1(nn.Module):
    def __init__(self, label_smoothing, lbl_set_size, num_c,args, dim=1):
        super(LabelSmoothingLoss1, self).__init__()
        # 标签标注正确的概率
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        # 分类的类别
        self.cls = lbl_set_size
        self.dim = dim  # 在softmax中 dim=1表示对每一行求softmax
        self.args = args
        # num_c=[379,255,285,184,201,295,449]
        # sum1 = sum(num_c)
        # self.weight = [round(1-i/sum1,4) for i in num_c]
        sum0=sum(num_c)
        weight0 = [round(1-i/sum0,4) for i in num_c]
        sum1 = sum(weight0)
        self.weight = [round(i/sum1,4)+1.15 for i in weight0]

    def forward(self, pred, target):
        # pred = pred.log_softmax(dim=self.dim)
        # with torch.no_grad():
        #     true_dist = pred.data.clone()
            # true_dist = torch.zeros_like(pred)
            # true_dist.fill_(self.smoothing / (self.cls - 1))  # 虽然判断为假，但是存在标签误判的概率。-1是因为有一个标签是为真的
            # 将真实标签所在的位置概率替换上
            # true_dist.scatter_(1, target.data.unsqueeze(1),self.confidence)
            # scatter_dim=1时填充规则true[i][target[i][j]] = confi[i][j]
        # 因为是多张图片，所以对每一行计算完了之后再求平均，得到总的平均误差
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), 1)
        res = torch.sum(-true_dist * pred, dim=self.dim)
        ratio = torch.tensor([self.weight[i] for i in target]).to(self.args.device)
        # print(ratio)
        # exit(0)
        return torch.mean(ratio*res)
        # return torch.mean(res)

## Copied from https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):  # 相对于tensor的opti自带warmup代码，torch需要自己实现

    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, init_lr=1e-7, after_scheduler=None):
        self.init_lr = init_lr
        assert init_lr > 0, 'Initial LR should be greater than 0.'
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs

        return [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if (self.finished and self.after_scheduler) or self.total_epoch == 0:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)