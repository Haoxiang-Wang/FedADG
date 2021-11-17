import os

import copy
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn.functional as F


# test on target domain
def test1(fetExtrac, classifier, valid_loader, device):
    fetExtrac = fetExtrac.eval()
    classifier = classifier.eval()
    num_correct = 0.
    num_all = 0.
    with torch.no_grad():
        fetExtrac = fetExtrac.to(device)
        classifier = classifier.to(device)
        for t, batch in enumerate(valid_loader, 0):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            feature = fetExtrac(x)

            label_out = classifier(feature)
            pre = torch.max(label_out, 1)[1].data.squeeze()
            num_correct += (pre == y).sum()
            num_all += x.size(0)
    return (num_correct*1.0/num_all)