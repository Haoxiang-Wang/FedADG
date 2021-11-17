import sys
import torch
import numpy as np
from torch import optim
from models.test import test1
from utils.utilss import LabelSmoothingLoss, GradualWarmupScheduler


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
        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7)
        self.fetExtrac.optimizer = optim.SGD(self.fetExtrac.parameters(), args.lr0, args.momentum,weight_decay=args.weight_dec)
        self.discri.optimizer = optim.SGD(self.discri.parameters(), args.lr1, args.momentum,weight_decay=args.weight_dec)

        afsche_fet = optim.lr_scheduler.ReduceLROnPlateau(self.opti_encoder, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_fet = GradualWarmupScheduler(self.opti_encoder, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_fet)
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
        ac = [0.]
        best_model_dict = {}
        self.before_train(trainId='FC')
        global_los= [0., 0., 0., 0.]
        for i in range(self.args.epochs):
            loss_cla, loss_discri, loss_enc, loss_gene = 0., 0., 0., 0.
            print('\r{}/{}'.format(i + 1, self.args.epochs), end='')
            for t, batch in enumerate(self.train_loader):
                loss_cla1, loss_discri1, loss_enc1, loss_gene1 =self.train_step(t, batch)
                loss_cla+=loss_cla1
                loss_discri+=loss_discri1
                loss_enc += loss_enc1
                loss_gene+=loss_gene1

            # update learning rate
            valid_acc = test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device)
            ac.append(valid_acc)
            self.sche_dis.step(i+1, 1. - ac[-1])
            self.sche_gene.step(i+1, 1. - ac[-1])
            self.sche_task.step(i+self.args.i_epochs+1, 1.-ac[-1])

            if ac[-1] >= np.max(ac):
                best_model_dict['F'] = self.fetExtrac.state_dict()
                best_model_dict['C'] = self.classifier.state_dict()
                best_model_dict['G'] = self.generator.state_dict()
                best_model_dict['D'] = self.discri
                global_los = [loss_cla, loss_discri, loss_enc, loss_gene]
        loc_w = [best_model_dict['F'], best_model_dict['C'], best_model_dict['G']]

        return np.max(ac), loc_w, best_model_dict['D'], global_los

    def train_step(self, t, batch):
        alpha = 0.85
        x, y = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)

        randomn = torch.rand(y.size(0), self.args.input_size).to(self.args.device)

        self.fetExtrac.train()
        self.classifier.train()
        self.generator.train()
        self.discri.train()

        # training discriminator
        self.fetExtrac.eval()
        self.discri.optimizer.zero_grad()
        y_onehot = torch.zeros(y.size(0), self.args.num_labels).to(self.args.device)
        y_onehot.scatter_(1, y.view(-1, 1), 0.7).to(self.args.device)
        fakez = self.fetExtrac(x).detach()
        realz = self.generator(y_onehot,randomn,self.args.device).detach()
        loss_discri = torch.mean(torch.pow(1 - self.discri(y_onehot, realz), 2) + torch.pow(self.discri(y_onehot, fakez), 2))
        loss_discri.backward()
        self.discri.optimizer.step()
        self.discri.eval()

        # training feature extractor and classifier
        self.fetExtrac.train()
        self.opti_task.zero_grad()
        fakez = self.fetExtrac(x)
        pre = self.classifier(fakez)
        loss_cla = self.lossFunc(pre, y)
        loss_enc = torch.mean(torch.pow(1-self.discri(y_onehot, fakez), 2))
        loss_cla = alpha*loss_cla + (1-alpha)*loss_enc
        loss_cla.backward()
        self.opti_task.step()

        # training distribution generator
        self.generator.train()
        self.generator.optimizer.zero_grad()
        realz = self.generator(y_onehot,randomn,self.args.device)
        loss_gene = torch.mean(torch.pow(self.discri(y_onehot, realz), 2))
        loss_gene.backward()
        self.generator.optimizer.step()

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
        elif trainId == 'D':  # training D
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
