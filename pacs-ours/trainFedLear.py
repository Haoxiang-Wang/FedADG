import argparse
import copy
import sys
import matplotlib.pyplot as plt
import numpy
import random
from torch.optim.lr_scheduler import LambdaLR

from models.Fed import FedAvg
from data_loader import get_pacs_loaders
from Nets import feature_extractor, ResNet50Model, task_classifier
from utils.layers import SinkhornDistance
from utils.sampling import load_FCparas, get_loaders
import torch
import numpy as np
from torch import optim
from models.test import test1, test2
from utils.utilss import LabelSmoothingLoss, GradualWarmupScheduler, LabelSmoothingLoss1
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def args_parser():
    paser = argparse.ArgumentParser()
    paser.add_argument('--batch-size', type=int, default=10, help='batch size for training')
    paser.add_argument('--workers', type=int, default=4, help='number of data-loading workers')
    paser.add_argument('--lr0', type=float, default=0.0001, help='learning rate 0')
    paser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    paser.add_argument('--weight-dec', type=float, default=1e-5, help='0.005weight decay coefficient default 1e-5')
    paser.add_argument('--epochs', type=int, default=10, help='rounds of training')
    paser.add_argument('--current_epoch', type=int, default=1, help='current epoch in training')
    paser.add_argument('--factor', type=float, default=0.2, help='lr decreased factor (0.1)')
    paser.add_argument('--patience', type=int, default=20, help='number of epochs to waut before reduce lr (20)')
    paser.add_argument('--lr-threshold', type=float, default=1e-4, help='lr schedular threshold')
    paser.add_argument('--ite-warmup', type=int, default=100, help='LR warm-up iterations (default:500)')
    paser.add_argument('--label_smoothing', type=float, default=0.02, help='the rate of wrong label(default:0.2)')
    paser.add_argument('--hidden_size', type=int, default=4096, help='the size of hidden feature')  # 4096-alex
    paser.add_argument('--num_labels', type=int, default=7, help='the categories of labels')  # pacs-7
    paser.add_argument('--global_epochs', type=int, default=23, help='the num of global train epochs')
    paser.add_argument('--path_root', type=str, default='../data/PACS/', help='the root of dataset')
    args = paser.parse_args()
    return args

class localTrain(object):
    def __init__(self, fetExtrac, classifier, train_loader, valid_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.to(args.device)
        self.classifier = classifier.to(args.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=7)
        self.opti_task = optim.SGD(list(self.fetExtrac.parameters()) + list(self.classifier.parameters()), args.lr0,
                                   args.momentum, weight_decay=args.weight_dec)
        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                           threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                                after_scheduler=afsche_task)

    def train(self):
        ac = [0.]
        best_model_dict = {}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opti_task, 2, eta_min=1e-7)
        # server execute
        ac=[0.]
        self.fetExtrac.train()
        self.classifier.train()
        for i1 in range(self.args.epochs):
            print('\r{}/{}'.format(i1 + 1, self.args.epochs), end='')
            for t1, batch in enumerate(self.train_loader):
                x, y = batch
                x = x.to(self.args.device)
                y = y.to(self.args.device)

                # trainning feature extractor and classifier
                self.opti_task.zero_grad()
                feature = self.fetExtrac(x)
                pre = self.classifier(feature)
                loss_cla = self.lossFunc(pre, y)
                loss_cla.backward()
                self.opti_task.step()
            valid_acc = test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device)
            ac.append(valid_acc)
            self.sche_task.step(i1,1.-ac[-1])
            if ac[-1] >= np.max(ac):
                best_model_dict['F'] = self.fetExtrac.state_dict()
                best_model_dict['C'] = self.classifier.state_dict()
        loc_w = [best_model_dict['F'], best_model_dict['C']]
        return np.max(ac), loc_w

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(1) if torch.cuda.is_available() else 'cpu')
    numpy.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)

    local_runs = 1
    client = ['photo', 'art_painting', 'cartoon', 'sketch']
    for iteration in range(4 * local_runs):
        torch.cuda.empty_cache()
        if iteration == local_runs:
            client = ['photo', 'cartoon', 'sketch', 'art_painting']
        elif iteration == 2 * local_runs:
            client = ['art_painting', 'sketch', 'photo', 'cartoon']
        elif iteration == 3 * local_runs:
            client = ['art_painting', 'cartoon', 'sketch', 'photo']
        print('\nIteration {}'.format(iteration))

        train_loaders, valid_loaders, target_loader = get_loaders(args, client)

        # initialize global models
        global_fetExtrac = feature_extractor(optim.SGD, args.lr0, args.momentum, args.weight_dec)
        global_fetExtrac.load_state_dict(load_FCparas("alexnet"), strict=False)
        global_fetExtrac.optimizer = optim.SGD(global_fetExtrac.parameters(), args.lr0, args.momentum,
                                               weight_decay=args.weight_dec)
        global_classifier = task_classifier(args.hidden_size, optim.SGD, args.lr0, args.momentum, args.weight_dec,
                                            class_num=args.num_labels)

        # iteration training
        models_global = []
        model_best_paras, best_acc = {}, 0.
        for t in range(args.global_epochs):
            print('global train epoch: %d ' % (t + 1))
            w_locals, avg_ac = [], 0.
            # client update
            for i in range(3):
                print('train domain {}/3'.format(i + 1))
                local_f = copy.deepcopy(global_fetExtrac)
                local_c = copy.deepcopy(global_classifier)
                trainer = localTrain(local_f, local_c, train_loaders[i], valid_loaders[i], args)
                acc, w = trainer.train()
                w_locals.append(w)
                avg_ac += acc
            models_global.clear()
            models_global = FedAvg(w_locals)
            avg_ac /= 3.
            if avg_ac > best_acc:
                model_best_paras['F'] = models_global[0]
                model_best_paras['C'] = models_global[1]
                best_acc = avg_ac
            global_fetExtrac.load_state_dict(models_global[0])
            global_classifier.load_state_dict(models_global[1])

        # test on target domain
        global_fetExtrac.load_state_dict(model_best_paras['F'])
        global_classifier.load_state_dict(model_best_paras['C'])
        acc_target = 0.  
        for testi in range(10):
            acc_target += test1(global_fetExtrac, global_classifier, target_loader, args.device)
        acc_target /= 10
        print(acc_target.item())