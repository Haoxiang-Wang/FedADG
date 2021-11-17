import argparse
import copy
import sys
import torch
from torch import optim

from models.Fed import FedAvg, FedAvg1
from models.Nets import feature_extractor, ResNet50Model, task_classifier, GeneDistrNet, domain_discriminator, \
    Discriminator
from models.test import test1
from src.localTrain import localTrain
from utils.sampling import load_FCparas, get_loaders


def args_parser():
    paser = argparse.ArgumentParser()
    paser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    paser.add_argument('--workers', type=int, default=4, help='number of data-loading workers')
    paser.add_argument('--lr0', type=float, default=0.001, help='learning rate 0')
    paser.add_argument('--lr1', type=float, default=0.0007, help='learning rate 1')
    paser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    paser.add_argument('--weight-dec', type=float, default=1e-5, help='0.005weight decay coefficient default 1e-5')
    paser.add_argument('--rp-size', type=int, default=1024, help='Random Projection size')
    paser.add_argument('--epochs', type=int, default=10, help='rounds of training')
    paser.add_argument('--current_epoch', type=int, default=1, help='current epoch in training')
    paser.add_argument('--factor', type=float, default=0.2, help='lr decreased factor (0.1)')  # 0.3
    paser.add_argument('--patience', type=int, default=20, help='number of epochs to waut before reduce lr (20)')
    paser.add_argument('--lr-threshold', type=float, default=1e-4, help='lr schedular threshold')
    paser.add_argument('--ite-warmup', type=int, default=500, help='LR warm-up iterations (default:500)')
    paser.add_argument('--label_smoothing', type=float, default=0.2, help='the rate of wrong label(default:0.2)')
    paser.add_argument('--input_size', type=int, default=2048, help='the size of hidden feature')
    paser.add_argument('--hidden_size', type=int, default=4096, help='the size of hidden feature')  # 4096-alex
    paser.add_argument('--num_labels', type=int, default=5, help='the categories of labels')  # vlcs-5
    paser.add_argument('--global_epochs', type=int, default=20, help='the num of global train epochs')
    paser.add_argument('--pin', type=bool, default=True, help='pin-memory')
    paser.add_argument('--path_root', type=str, default='../data/VLCS/', help='the root of dataset')
    args = paser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(10)

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

        train_loaders, valid_loaders, target_loader = get_loaders(args, client)

        # initialize the global model
        global_fetExtrac = feature_extractor(optim.SGD, args.lr0, args.momentum, args.weight_dec)
        global_fetExtrac.load_state_dict(load_FCparas("alexnet"), strict=False)
        global_classifier = task_classifier(args.hidden_size, optim.SGD, args.lr0, args.momentum, args.weight_dec)
        global_generator = GeneDistrNet(args.input_size,args.hidden_size, optim.SGD, args.lr1, args.momentum, args.weight_dec)

        local_d = []
        for i in range(3):  # local discriminator
            global_discri = Discriminator(args.hidden_size, args.num_labels)
            global_discri.optimizer = optim.SGD(global_discri.parameters(), args.lr1, args.momentum,
                                                weight_decay=args.weight_dec)
            local_d.append(global_discri)

        # server excutes
        models_global = []
        model_best_paras, best_acc, best_id = {}, 0., 0
        for t in range(args.global_epochs):
            print('global train epoch: %d ' % (t + 1))
            args.current_epoch = t+1
            w_locals, avg_ac = [], 0.
            for i in range(3):
                print('train domain {}/3'.format(i + 1))
                local_f = copy.deepcopy(global_fetExtrac)
                local_c = copy.deepcopy(global_classifier)
                local_g = copy.deepcopy(global_generator)
                trainer = localTrain(local_f, local_c, local_g, local_d[i], train_loaders[i], valid_loaders[i], args)
                acc, w, wd = trainer.train()
                w_locals.append(w)
                local_d[i] = wd
                avg_ac += acc
                torch.cuda.empty_cache()
            models_global.clear()
            # aggregation
            models_global = FedAvg(w_locals)
            avg_ac /= 3.
            if avg_ac > best_acc:
                model_best_paras['F'] = models_global[0]
                model_best_paras['C'] = models_global[1]
                best_acc = avg_ac
                best_id = t+1
            global_fetExtrac.load_state_dict(models_global[0])
            global_classifier.load_state_dict(models_global[1])
            global_generator.load_state_dict(models_global[2])


        # test on target domain
        global_fetExtrac.load_state_dict(model_best_paras['F'])
        global_classifier.load_state_dict(model_best_paras['C'])
        acc_target = 0.
        for testi in range(10):
            acc_target += test1(global_fetExtrac, global_classifier, target_loader, args.device)
        acc_target /= 10
        print(acc_target.item())