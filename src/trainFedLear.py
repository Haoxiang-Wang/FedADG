import argparse
import copy
import sys
from models.Fed import FedAvg
from models.Nets import feature_extractor, ResNet50Model, task_classifier
from utils.sampling import load_FCparas, get_loaders
import torch
import numpy as np
from torch import optim
from models.test import test1
from utils.utilss import LabelSmoothingLoss, GradualWarmupScheduler


def args_parser():
    paser = argparse.ArgumentParser()
    paser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    paser.add_argument('--workers', type=int, default=4, help='number of data-loading workers')
    paser.add_argument('--lr0', type=float, default=0.001, help='learning rate 0')
    paser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    paser.add_argument('--weight-dec', type=float, default=1e-5, help='0.005weight decay coefficient default 1e-5')
    paser.add_argument('--rp-size', type=int, default=1024, help='Random Projection size')
    paser.add_argument('--epochs', type=int, default=10, help='rounds of training')
    paser.add_argument('--current_epoch', type=int, default=1, help='current epoch in training')
    paser.add_argument('--factor', type=float, default=0.2, help='lr decreased factor (0,1)')
    paser.add_argument('--patience', type=int, default=20, help='number of epochs to waut before reduce lr (20)')
    paser.add_argument('--lr-threshold', type=float, default=1e-4, help='lr schedular threshold')
    paser.add_argument('--ite-warmup', type=int, default=500, help='LR warm-up iterations (default:500)')
    paser.add_argument('--label_smoothing', type=float, default=0.2, help='the rate of wrong label(default:0.2)')
    paser.add_argument('--hidden_size', type=int, default=4096, help='the size of hidden feature')  # 4096-alex 2048-res
    paser.add_argument('--num_labels', type=int, default=5, help='the categories of labels')  # vlcs-5
    paser.add_argument('--global_epochs', type=int, default=20, help='the num of global train epochs')
    paser.add_argument('--path_root', type=str, default='../data/VLCS/', help='the root of dataset')
    args = paser.parse_args()
    return args

class localTrain(object):
    def __init__(self, fetExtrac, classifier, train_loader, valid_loader, args):
        self.args = args
        self.fetExtrac = fetExtrac.cuda()
        self.classifier = classifier.cuda()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=5)
        self.opti_task = optim.SGD(list(self.fetExtrac.parameters()) + list(self.classifier.parameters()), args.lr0,
                                   args.momentum, weight_decay=args.weight_dec)
        afsche_task = optim.lr_scheduler.ReduceLROnPlateau(self.opti_task, factor=args.factor, patience=args.patience,
                                                           threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_task = GradualWarmupScheduler(self.opti_task, total_epoch=args.ite_warmup,
                                                after_scheduler=afsche_task)

    def train(self):
        ac = [0.]
        best_model_dict = {}
        # global execute
        for i in range(self.args.epochs):
            print('\r{}/{}'.format(i + 1, self.args.epochs), end='')
            for t, batch in enumerate(self.train_loader):
                x, y = batch
                x = x.to(self.args.device)
                y = y.to(self.args.device)

                self.fetExtrac.train()
                self.classifier.train()

                # train classifier
                self.opti_task.zero_grad()
                feature = self.fetExtrac(x)
                pre = self.classifier(feature)
                loss_cla = self.lossFunc(pre, y)
                loss_cla.backward()
                self.opti_task.step()

            valid_acc = test1(self.fetExtrac, self.classifier, self.valid_loader, self.args.device)
            ac.append(valid_acc)
            self.sche_task.step(i,1.-ac[-1])
            if ac[-1] >= np.max(ac):
                best_model_dict['F'] = self.fetExtrac.state_dict()
                best_model_dict['C'] = self.classifier.state_dict()
        loc_w = [best_model_dict['F'], best_model_dict['C']]
        dx = sys.getsizeof(loc_w)+sys.getsizeof(np.max(ac))
        return np.max(ac), loc_w

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(10)
    local_runs = 1
    client = ['Caltech101', 'LabelMe', 'VOC2007', 'SUN09']
    for iteration in range(4*local_runs):
        torch.cuda.empty_cache()
        if iteration == local_runs:
            client = ['SUN09', 'Caltech101', 'VOC2007', 'LabelMe']
        elif iteration == 2*local_runs:
            client = ['SUN09', 'LabelMe', 'VOC2007', 'Caltech101']
        elif iteration == 3*local_runs:
            client = ['SUN09', 'LabelMe', 'Caltech101', 'VOC2007']
        print('\nIteration {}'.format(iteration))
        train_loaders, valid_loaders, target_loader = get_loaders(args, client)

        # initialize the global models
        global_fetExtrac = feature_extractor(optim.SGD, args.lr0, args.momentum, args.weight_dec)
        global_fetExtrac.load_state_dict(load_FCparas("alexnet"), strict=False)
        global_classifier = task_classifier(args.hidden_size, optim.SGD, args.lr0, args.momentum, args.weight_dec)

        # iteration training
        models_global = []
        model_best_paras, best_acc = {}, 0.
        for t in range(args.global_epochs):
            print('global train epoch: %d ' % (t + 1))
            args.current_epoch = t+1
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

        # test on traget domains
        global_fetExtrac.load_state_dict(model_best_paras['F'])
        global_classifier.load_state_dict(model_best_paras['C'])
        acc_target = 0.
        for testi in range(10):
            acc_target += test1(global_fetExtrac, global_classifier, target_loader, args.device)
        acc_target /= 10
        print(acc_target.item())