import torch
from torch import optim
from models.test import test1
from utils.utilss import LabelSmoothingLoss, GradualWarmupScheduler


# training of client
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
        self.opti_task = optim.SGD(list(self.fetExtrac.parameters())+list(self.classifier.parameters()), args.lr0,
                                   args.momentum,weight_decay=args.weight_dec)
        self.bce_loss = torch.nn.BCELoss()
        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=5)

        # scheduler to update learning rate
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
                                                          patience=args.patience, threshold=args.lr_threshold,
                                                          min_lr=1e-7)
        self.sche_dis = GradualWarmupScheduler(self.discri.optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_dis)

    def train(self):
        best_model_dict = {}

        # training F and C ( E0 = 5 )
        self.fetExtrac.train()
        self.classifier.train()
        for i in range(5):
            print('\r{}/5'.format(i + 1), end='')
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
            acc = test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device)
            self.sche_task.step(i, 1. - acc)

        # local training (E1)
        g_ac = 0.
        ac = [0.]
        for i in range(self.args.epochs):
            print('\r{}/{}'.format(i + 1, self.args.epochs), end='')
            for t, batch in enumerate(self.train_loader):
                self.train_step(batch)
                torch.cuda.empty_cache()

            valid_acc = test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device)
            ac.append(valid_acc)
            # update learning rate
            self.sche_dis.step(i+1, 1.-ac[-1])
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

        randomn = torch.rand(y.size(0), self.args.input_size).to(self.args.device)
        self.fetExtrac.train()
        self.classifier.train()
        self.generator.train()
        self.discri.train()

        # training discriminator
        self.fetExtrac.eval()
        self.discri.optimizer.zero_grad()
        fakez = self.fetExtrac(x).detach()
        y_onehot = torch.zeros(y.size(0), self.args.num_labels).to(self.args.device)
        y_onehot.scatter_(1, y.view(-1, 1), 0.6).to(self.args.device)
        realz = self.generator(y=y_onehot,x=randomn,device=self.args.device).detach()
        loss_discri = -torch.mean(torch.pow(self.discri(y_onehot, realz), 2) + torch.pow(1-self.discri(y_onehot, fakez),2))
        loss_discri.backward()
        self.discri.optimizer.step()
        self.discri.eval()

        # training feature extractor and classifier
        self.fetExtrac.train()
        self.opti_task.zero_grad()
        fakez = self.fetExtrac(x)
        pre = self.classifier(fakez)
        loss_cla = self.lossFunc(pre, y)
        loss_enc = torch.mean(torch.pow(1 - self.discri(y_onehot, fakez), 2))
        loss_cla = alpha * loss_cla + (1 - alpha) * loss_enc
        loss_cla.backward()
        self.opti_task.step()

        # training distribution generator
        self.generator.train()
        self.generator.optimizer.zero_grad()
        realz = self.generator(y=y_onehot,x=randomn,device=self.args.device).detach()
        loss_gene = torch.mean(torch.pow(1-self.discri(y_onehot, realz), 2))
        loss_gene.backward()
        self.generator.optimizer.step()