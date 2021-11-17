import sys
from math import pi

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import optim
from scipy.interpolate import make_interp_spline

from models.test import test1, test2
from utils.utilss import LabelSmoothingLoss, GradualWarmupScheduler


class localTrain(object):
    def __init__(self, fetExtrac, classifier, generator, discri, train_loader, valid_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()
        self.classifier = classifier.cuda()
        self.generator = generator.cuda()
        self.discri = discri.cuda()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.opti_encoder = optim.SGD(fetExtrac.parameters(), args.lr1, args.momentum,weight_decay=args.weight_dec)
        self.opti_task = optim.SGD(list(self.fetExtrac.parameters())+list(self.classifier.parameters()), args.lr0, args.momentum,weight_decay=args.weight_dec)
        self.bce_loss = torch.nn.BCELoss()
        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=5)

        # scheduler to update learning rate
        afsche_fet = optim.lr_scheduler.ReduceLROnPlateau(self.opti_encoder, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_fet = GradualWarmupScheduler(self.opti_encoder, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_fet)
        #
        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                           threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                                after_scheduler=afsche_task)
        afsche_gen = optim.lr_scheduler.ReduceLROnPlateau(self.generator.optimizer, factor=args.factor,
                                                          patience=args.patience,threshold=args.lr_threshold,
                                                          min_lr=1e-7)
        self.sche_gene = GradualWarmupScheduler(self.generator.optimizer, total_epoch=args.ite_warmup,
                                                after_scheduler=afsche_gen)
        afsche_dis = optim.lr_scheduler.ReduceLROnPlateau(self.discri.optimizer, factor=args.factor,
                                                          patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_dis = GradualWarmupScheduler(self.discri.optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_dis)

    def train(self):
        best_model_dict = {}

        # 先训练分类 5 epoch--仅第一次训练
        if self.args.current_epoch != -1:
            self.fetExtrac.train()
            self.classifier.train()
            for i in range(5):
                print('\r{}/5'.format(i + 1), end='')
                for t, batch in enumerate(self.train_loader):
                    x, y = batch
                    x = x.to(self.args.device)
                    y = y.to(self.args.device)
                    self.opti_task.zero_grad()
                    # self.fetExtrac.optimizer.zero_grad()
                    # self.classifier.optimizer.zero_grad()
                    feature = self.fetExtrac(x)
                    pre = self.classifier(feature)
                    loss_cla = self.lossFunc(pre, y)
                    loss_cla.backward()
                    self.opti_task.step()
                    # self.fetExtrac.optimizer.step()
                    # self.classifier.optimizer.step()
                acc = test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device)
                self.sche_task.step(i,1.-acc)

        # 全局训练
        g_ac = 0.
        ac = [0.]
        for i in range(self.args.epochs):
            print('\r{}/{}'.format(i + 1, self.args.epochs), end='')
            for t, batch in enumerate(self.train_loader):
                self.train_step(batch)
                torch.cuda.empty_cache()
                # update lr
            valid_acc = test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device)
            # valid_acc = test2(self.fetExtrac, self.classifier, self.valid_loader, self.args.device,label=5)
            ac.append(valid_acc)
            self.sche_dis.step(i+1, 1.-ac[-1])
            # self.sche_fet.step(i+1, 1.-ac[-1])
            self.sche_gene.step(i+1, 1.-ac[-1])
            self.sche_task.step(i + 5 + 1, 1. - ac[-1])
            if valid_acc >= g_ac:
                best_model_dict['F'] = self.fetExtrac.state_dict()
                best_model_dict['C'] = self.classifier.state_dict()
                best_model_dict['G'] = self.generator.state_dict()
                best_model_dict['D'] = self.discri
                g_ac = valid_acc
        loc_w = [best_model_dict['F'], best_model_dict['C'], best_model_dict['G']]
        return g_ac, loc_w, best_model_dict['D']

    def train_step(self, batch):
        alpha = 0.85
        x, y = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)

        # laplace - by_MMD
        # sample_inLap = np.random.laplace(0, 1, y.size(0) * self.args.hidden_size)
        # sample_inLap_ts = torch.tensor(sample_inLap).float()
        # realz = sample_inLap_ts.view([y.size(0), self.args.hidden_size]).to(self.args.device)

        randomn = torch.rand(y.size(0), self.args.input_size).to(self.args.device)
        # randomn = torch.rand(self.args.hidden_size)
        # randomn = randomn.view(-1, self.args.hidden_size).to(self.args.device)

        self.fetExtrac.train()
        self.classifier.train()
        self.generator.train()
        self.discri.train()

        # # train classifier
        # # self.fetExtrac.optimizer.zero_grad()
        # # self.classifier.optimizer.zero_grad()
        # self.opti_task.zero_grad()
        # feature = self.fetExtrac(x)
        # pre = self.classifier(feature)
        # loss_cla = self.lossFunc(pre, y)
        # loss_cla.backward()
        # self.opti_task.step()
        # # self.fetExtrac.optimizer.step()
        # # self.classifier.optimizer.step()

        # train discriminator
        self.fetExtrac.eval()
        self.discri.optimizer.zero_grad()
        fakez = self.fetExtrac(x).detach()
        # realz = self.sample_guassian_mixture(fakez.size(0), y).to(self.args.device)
        # realz = self.generator(self.sample_guassian_mixture(fakez.size(0), y).to(self.args.device))
        y_onehot = torch.zeros(y.size(0), self.args.num_labels).to(self.args.device)
        y_onehot.scatter_(1, y.view(-1, 1), 0.6).to(self.args.device)
        # target_ones = torch.ones(x.size(0), 1).to(self.args.device)
        # target_zeros = torch.zeros(x.size(0), 1).to(self.args.device)

        realz = self.generator(y=y_onehot,x=randomn,device=self.args.device).detach()
        # realz = self.generator(randomn).detach()
        # z_true_pred = self.discri(y_onehot, realz)
        # z_fake_pred = self.discri(y_onehot, fakez)
        # loss_discri = self.bce_loss(z_true_pred, target_ones) + self.bce_loss(z_fake_pred, target_zeros)
        loss_discri = -torch.mean(torch.pow(self.discri(y_onehot, realz), 2) + torch.pow(1-self.discri(y_onehot, fakez),2))
        loss_discri.backward()
        self.discri.optimizer.step()
        self.discri.eval()

        # train encoder
        # self.opti_encoder.zero_grad()
        # fakez = self.fetExtrac(x)
        # # new part
        # # z_fake_pred = self.discri(y_onehot, fakez)
        # # loss_enc = self.bce_loss(z_fake_pred, target_ones)
        # loss_enc = -torch.mean(torch.pow(self.discri(y_onehot, fakez), 2))
        # loss_enc.backward()
        # self.opti_encoder.step()

        self.fetExtrac.train()
        self.opti_task.zero_grad()
        fakez = self.fetExtrac(x)
        pre = self.classifier(fakez)
        loss_cla = self.lossFunc(pre, y)
        loss_enc = torch.mean(torch.pow(1 - self.discri(y_onehot, fakez), 2))
        loss_cla = alpha * loss_cla + (1 - alpha) * loss_enc
        loss_cla.backward()
        self.opti_task.step()


        # train generator
        self.generator.train()
        self.generator.optimizer.zero_grad()
        # realz = self.generator(self.sample_guassian_mixture(fakez.size(0), y).to(self.args.device))
        realz = self.generator(y=y_onehot,x=randomn,device=self.args.device).detach()
        # z_true_pred = self.discri(y_onehot, realz)
        # loss_gene = self.bce_loss(z_true_pred, target_zeros)
        loss_gene = torch.mean(torch.pow(1-self.discri(y_onehot, realz), 2))
        loss_gene.backward()
        self.generator.optimizer.step()


    # local module experiment
    def sample_guassian_mixture(self, batch_size, labels):
        x_var = 0.5
        y_var = 0.05
        x = torch.randn(batch_size, self.args.hidden_size // 2).mul(x_var).to(self.args.device)
        y = torch.randn(batch_size, self.args.hidden_size // 2).mul(y_var).to(self.args.device)
        shift = 1.4
        r = labels.type(torch.float32).mul(2 * pi / self.args.num_labels)
        sin_r = r.sin().view(-1, 1).to(self.args.device)
        cos_r = r.cos().view(-1, 1).to(self.args.device)
        new_x = x.mul(cos_r) - y.mul(sin_r)
        new_y = x.mul(sin_r) + y.mul(cos_r)
        new_x += shift * cos_r
        new_y += shift * sin_r
        return torch.cat([new_x, new_y], 1)

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        :return: sum(kernel_val):多个高斯核矩阵之和
        """
        row_matrix = int(source.size()[0]) + int(target.size()[0])  # 矩阵的行数
        total = torch.cat([source, target], dim=0)  # 将source和target按列方向合并
        # 将total复制n+m份
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (row_matrix ** 2 - row_matrix)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        # fix_sigma为中值,kernel_mul为倍数 取kernel_num个bandwidth值 如sigma=1,得到[0.25,0.5,1,2,4]
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    # 实验发现加了mmd的损失函数也没办法减小两个分布之间的距离
    def mmd_rbf(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):  # sigma:σ
        """
        实验时发现source/target数据要在0-1之间，否则mmd失效
        :param source: 源数据 (n*len(x))
        :param target: 目标数据 (n*len(y)) 一般来说是源向目标靠近
        :param kernel_mul:
        :param kernel_num: 不同高斯核的数量
        :param fix_sigma: 不同高斯核的sigma值
        :return:  the MMD(loss) of source and target 因为一般n=m,所以L矩阵不加入计算
        """
        batch_size = int(source.size()[0])  # 一般默认source and target的Batch_size相同
        kernels = self.gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        mmd = torch.mean(XX + YY - XY - YX)
        return mmd