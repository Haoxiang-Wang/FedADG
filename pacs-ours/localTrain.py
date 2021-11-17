import sys
import torch
import numpy as np
from torch import optim
from models.test import test1, test2
from utils.layers import SinkhornDistance
from utils.utilss import LabelSmoothingLoss, GradualWarmupScheduler, LabelSmoothingLoss1


class localTrain(object):
    def __init__(self, fetExtrac, classifier, generator, discri, train_loader, valid_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.to(self.args.device)
        self.classifier = classifier.to(self.args.device)
        self.generator = generator.to(self.args.device)
        self.discri = discri.to(self.args.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.opti_encoder = optim.SGD(self.fetExtrac.parameters(), args.lr1, args.momentum,weight_decay=args.weight_dec)
        self.opti_task = optim.SGD(list(self.fetExtrac.parameters())+list(self.classifier.parameters()), args.lr0, args.momentum,weight_decay=args.weight_dec)
        # self.bce_loss = torch.nn.BCELoss()
        # self.lossFunc = torch.nn.CrossEntropyLoss()
        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7)
        # self.lossFunc = LabelSmoothingLoss1(args.label_smoothing, lbl_set_size=7,num_c=args.weigh_class_num,args=args)
        # self.tain_enc_epoch = round(args.epochs/2)
        self.fetExtrac.optimizer = optim.SGD(self.fetExtrac.parameters(), args.lr0, args.momentum,weight_decay=args.weight_dec)
        self.discri.optimizer = optim.SGD(self.discri.parameters(), args.lr1, args.momentum,weight_decay=args.weight_dec)

        afsche_fet = optim.lr_scheduler.ReduceLROnPlateau(self.opti_encoder, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_fet = GradualWarmupScheduler(self.opti_encoder, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_fet)
        #
        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_task)
        #
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
        # self.sinkhorn = SinkhornDistance(eps=0.1,max_iter=100,reduction=None)
        # self.projection = torch.nn.Linear(4096, 2048, bias=False)
        # with torch.no_grad():
        #     self.projection.to(args.device)
        #     self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))
        # denum = 2
        # self.schessss_task = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opti_task, denum, eta_min=1e-10)
        # self.sche_fet = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opti_encoder, denum, eta_min=1e-10)
        # self.sche_gene = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.generator.optimizer, denum, eta_min=1e-10)
        # self.sche_dis = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.discri.optimizer, denum, eta_min=1e-10)

    def train(self):
        ac = [0.]
        best_model_dict = {}
        self.before_train(trainId='FC')
        # 全局训练
        ac = [0.]
        global_los= [0., 0., 0., 0.]
        for i in range(self.args.epochs):
            # torch.cuda.empty_cache()
            loss_cla, loss_discri, loss_enc, loss_gene = 0., 0., 0., 0.
            print('\r{}/{}'.format(i + 1, self.args.epochs), end='')
            for t, batch in enumerate(self.train_loader):
                loss_cla1, loss_discri1, loss_enc1, loss_gene1 =self.train_step(t, batch)
                loss_cla+=loss_cla1
                loss_discri+=loss_discri1
                loss_enc += loss_enc1
                loss_gene+=loss_gene1

            # update lr
            # print("losscla:{:.2f},lossdiscri:{:.2f},lossgene:{:.2f}".format(loss_cla,loss_discri,loss_gene))
            valid_acc = test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device)
            # valid_acc = test2(self.fetExtrac, self.classifier, self.valid_loader, self.args.device,mode='r')
            # print(valid_acc)
            ac.append(valid_acc)
            self.sche_dis.step(i+1, 1. - ac[-1])
            self.sche_gene.step(i+1, 1. - ac[-1])
            self.sche_task.step(i+self.args.i_epochs+1, 1.-ac[-1])
            # self.sche_fet.step(i+1, 1. - ac[-1])
            # self.sche_dis.step()
            # self.sche_fet.step()
            # self.sche_gene.step()
            # self.sche_task.step()

            if ac[-1] >= np.max(ac):
                best_model_dict['F'] = self.fetExtrac.state_dict()
                best_model_dict['C'] = self.classifier.state_dict()
                best_model_dict['G'] = self.generator.state_dict()
                best_model_dict['D'] = self.discri
                global_los = [loss_cla, loss_discri, loss_enc, loss_gene]
        loc_w = [best_model_dict['F'], best_model_dict['C'], best_model_dict['G']]

        # loc_w = [self.fetExtrac.state_dict(), self.classifier.state_dict(), self.generator.state_dict()]
        # best_model_dict['D'] = self.discri
        return np.max(ac), loc_w, best_model_dict['D'], global_los

    def train_step(self, t, batch):
        alpha = 0.85
        x, y = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)

        # laplace - by_MMD (1/2**0.5)
        # sample_inLap = np.random.laplace(0, 1/(2 ** 0.5), y.size(0)*self.args.hidden_size)
        # sample_inLap_ts = torch.tensor(sample_inLap).float()
        # realz = sample_inLap_ts.view([y.size(0), self.args.hidden_size]).to(self.args.device)
        #
        # # uniform -test(-1,1)
        # sample_uniform = np.random.uniform(-1,1,y.size(0) * self.args.hidden_size)
        # sample_uniform = torch.tensor(sample_uniform).float()
        # realz = sample_uniform.view([y.size(0), self.args.hidden_size]).to(self.args.device)

        # gaussian - test (0,1)
        # sample_guassian=torch.normal(mean=0.,std=1.,size=(y.size(0),self.args.hidden_size)) # 离散型
        # sample_guassian = torch.randn(y.size(0)*self.args.hidden_size) # 连续型
        # realz = sample_guassian.view([y.size(0), self.args.hidden_size]).to(self.args.device)

        # batch_size*hidden_size
        # random = np.random.uniform(-1,1,y.size(0) * self.args.input_size)
        # random = torch.tensor(random).float()
        # randomn = torch.rand(self.args.input_size).to(self.args.device)
        randomn = torch.rand(y.size(0), self.args.input_size).to(self.args.device)

        # randomn = torch.rand(self.args.input_size)
        # randomn = randomn.view(-1, self.args.input_size).to(self.args.device)

        self.fetExtrac.train()
        self.classifier.train()
        self.generator.train()
        self.discri.train()

        # Wasserstain distance
        # self.opti_encoder.zero_grad()
        # self.generator.optimizer.zero_grad()
        # fakez = self.fetExtrac(x)
        # realz = self.generator(randomn)
        # fakez = fakez.view(fakez.size(0),-1)
        # fakez = self.projection(fakez)
        # realz = realz.view(realz.size(0),-1)
        # realz = self.projection(realz)
        # fgdist, _p, _c = self.sinkhorn(fakez.to("cpu"), realz.to("cpu"))
        # fgdist.to(self.args.device)
        # fgdist.backward()
        # self.opti_encoder.step()
        # self.generator.optimizer.step()
        # #
        # return loss_cla.item(), -1, fgdist,-1

        # train discriminator

        # self.generator.eval()
        self.fetExtrac.eval()
        self.discri.optimizer.zero_grad()
        y_onehot = torch.zeros(y.size(0), self.args.num_labels).to(self.args.device)
        y_onehot.scatter_(1, y.view(-1, 1), 0.5).to(self.args.device)
        # print(y_onehot)
        fakez = self.fetExtrac(x).detach()
        realz = self.generator(y_onehot,randomn,self.args.device).detach()
        # z_true_pred = self.discri(y_onehot, realz)
        # z_fake_pred = self.discri(y_onehot, fakez)
        # target_ones = torch.ones(x.size(0), 1).to(self.args.device)
        # target_zeros = torch.zeros(x.size(0), 1).to(self.args.device)
        loss_discri = torch.mean(torch.pow(1 - self.discri(y_onehot, realz), 2) + torch.pow(self.discri(y_onehot, fakez), 2))
        # loss_discri = -torch.mean(torch.pow(self.discri(y_onehot, realz), 2) + torch.pow(1 - self.discri(y_onehot, fakez), 2))
        loss_discri.backward()
        self.discri.optimizer.step()
        self.discri.eval()

        # # train encoder
        # self.fetExtrac.train()
        # self.opti_encoder.zero_grad()
        # fakez = self.fetExtrac(x)
        # # new part
        # # z_fake_pred = self.discri(y_onehot, fakez)
        # # loss_enc = self.bce_loss(z_fake_pred, target_ones)
        # # loss_enc = -torch.mean(torch.pow(self.discri(y_onehot, fakez), 2))
        # loss_enc = torch.mean(torch.pow(1-self.discri(y_onehot, fakez), 2))
        # loss_enc.backward()
        # self.opti_encoder.step()
        #
        # # train classifier
        # self.opti_task.zero_grad()
        # feature = self.fetExtrac(x)
        # pre = self.classifier(feature)
        # # loss_cla = self.lossFunc(pre, y)
        # loss_cla = self.lossFunc(pre, y)
        # loss_cla.backward()
        # self.opti_task.step()

        self.fetExtrac.train()
        self.opti_task.zero_grad()
        fakez = self.fetExtrac(x)
        pre = self.classifier(fakez)
        loss_cla = self.lossFunc(pre, y)
        loss_enc = torch.mean(torch.pow(1-self.discri(y_onehot, fakez), 2))
        loss_cla = alpha*loss_cla + (1-alpha)*loss_enc
        loss_cla.backward()
        self.opti_task.step()
        # print(loss_cla)
        # train generator
        self.generator.train()
        self.generator.optimizer.zero_grad()
        realz = self.generator(y_onehot,randomn,self.args.device)
        # z_true_pred = self.discri(y_onehot, realz)
        # loss_gene = self.bce_loss(z_true_pred, target_zeros)
        # loss_gene = torch.mean(torch.pow(1-self.discri(y_onehot, realz), 2))
        loss_gene = torch.mean(torch.pow(self.discri(y_onehot, realz), 2))
        loss_gene.backward()
        self.generator.optimizer.step()

        # loss_gene = loss_cla
        return loss_cla.item(), loss_discri.item(), loss_enc.item(), loss_gene.item()

    def before_train(self,trainId = 'FC'):
        if trainId == 'FC':
                ac=[0.]
                self.fetExtrac.train()
                self.classifier.train()
                for i in range(self.args.i_epochs):
                    print('\r{}/{}'.format(i + 1,self.args.i_epochs), end='')
                    for t, batch in enumerate(self.train_loader):
                        x, y = batch
                        x = x.to(self.args.device)
                        y = y.to(self.args.device)

                        self.opti_task.zero_grad()
                        feature = self.fetExtrac(x)
                        pre = self.classifier(feature)
                        loss_cla = self.lossFunc(pre, y)
                        loss_cla.backward()
                        self.opti_task.step()
                    ac.append(test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device))
                    self.sche_task.step(i, 1. - ac[-1])
        elif trainId == 'D': # 先训练D
            self.fetExtrac.eval()
            self.generator.eval()
            self.discri.train()
            for i in range(self.args.i_epochs):
                print('\r{}/{}'.format(i + 1, self.args.i_epochs), end='')
                for t, batch in enumerate(self.train_loader):
                    x, y = batch
                    x = x.to(self.args.device)
                    y = y.to(self.args.device)

                    randomn = torch.rand(y.size(0), self.args.input_size)
                    randomn = randomn.view(-1, self.args.input_size).to(self.args.device)
                    self.discri.optimizer.zero_grad()
                    y_onehot = torch.zeros(y.size(0), self.args.num_labels).to(self.args.device)
                    y_onehot.scatter_(1, y.view(-1, 1), 1).to(self.args.device)
                    fakez = self.fetExtrac(x).detach()
                    realz = self.generator(y_onehot, randomn).detach()

                    loss = torch.mean(torch.pow(1 - self.discri(y_onehot, realz), 2) + torch.pow(self.discri(y_onehot, fakez), 2))
                    loss.backward()
                    self.discri.optimizer.step()
                self.sche_dis.step()
        else:
            pass
        # torch.cuda.empty_cache()

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