import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs',type=int,default=10, help="rounds of training")
    parser.add_argument('--batch-size',type=int,default=32,metavar='N',help='input batch size for training (default:64)')
    parser.add_argument('--num-users',type=int,default=4,help="number of users: K")
    parser.add_argument('-lr-task',type=float,default=0.001, metavar='LR',help="Learning rate (default: 0.0001)")  # 测试过程如果lr = 0.01或者更大会严重影响avg后的准确率
    parser.add_argument('-lr-domain',type=float,default=0.001, metavar='LR', help="Learning rate (default:0.01/0.0002)")
    parser.add_argument('--momentum-task', type=float, default=0.9, metavar='m', help="SGD momentum (default: 0.9)")
    parser.add_argument('--momentum-domain', type=float, default=0.9, metavar='m', help="SGD momentum (default: 0.9)")
    parser.add_argument('--weight-dec', type=float, default=0.005, metavar='weight_dec', help='Weight decay coefficient (default: 1e-5/0.00001')
    parser.add_argument('--n-runs',type=int,default=3, metavar='runs',help='number of repetition (default: 3)')
    parser.add_argument('--fl-runs',type=int, default=20, metavar='runs',help='number of FL aggregation')
    parser.add_argument('--fl-run',type=int, default=0, metavar='runs',help='current number of FL aggregation')

    # model arguments
    parser.add_argument('--data_vlcs',type=str,default='./data/VLCS/', metavar='Path', help='Data path')
    parser.add_argument('--vlcs_s1', type=str, default='LabelMe', metavar='Path', help='Path to source1 file')
    parser.add_argument('--vlcs_s2', type=str, default='Caltech101', metavar='Path', help='Path to source2 file')
    parser.add_argument('--vlcs_s3', type=str, default='VOC2007', metavar='Path', help='Path to source3 file')
    parser.add_argument('--target', type=str, default='SUN09', metavar='Path', help='Path to target data')
    # parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. if None, training start from scratch')
    # parser.add_argument('--checkpoint-path', type=str, default='./',metavar='Path', help='path for checkpointing')
    parser.add_argument('--seed',type=int, default=10, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--alpha',type=float,default=0.80,metavar='alpha', help='balance losses to train F. Within [0,1]')
    parser.add_argument('--rp-size',type=int,default=3500,metavar='rp', help='Random projection size. Should be smaller than 3500/4096')
    parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='used in label smoothing (default: 0.2)')
    parser.add_argument('--workers',type=int, default=2, help='number of data loading workers')
    # parser.add_argument('--save-every',type=int, default=5,metavar='N',help='how many epochs to wait before logging training status. (default: 5)')
    parser.add_argument('--no-cuda',action='store_true',default=False,help='disable GPU use')
    parser.add_argument('--save-ckpt',type=bool,default=False, help='save checkpoint')
    parser.add_argument('--load-ckpt',type=bool,default=False, help='load checkpoint')

    # for schedular --warm up the lr which in optimizer
    parser.add_argument('--patience', type=int, default=60, metavar='N', help='number of epochs to wait before reducing lr (default: 20)')
    parser.add_argument('--factor',type=float,default=0.3, metavar='f', help='used in schedular, LR decreadr factor (default: 0.1)')
    parser.add_argument('--lr-threshold', type=float, default=1e-4, metavar='LRthrs', help='used in schedular LR (default:1e-4)')
    parser.add_argument('--warmup-its',type=int,default=300,metavar='w',help='LR warm-up iterations (default:500)')

    # other arguments
    parser.add_argument('--dataset',type=str,default='vlcs', help="name of dataset")
    parser.add_argument('--num-classes',type=int,default=5, help="number of classes")
    parser.add_argument('--FL',type=bool,default=False, help='weather train FL to compare')
    parser.add_argument('--all_clients',action='store_true', help='aggregation over all clients')

    args = parser.parse_args()
    return args


# classes = ['bird', 'car', 'chair', 'dog', 'person']
# num_train, num_valid = [891, 1672, 2067], [424, 797, 985]