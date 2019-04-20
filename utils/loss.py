from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import nn


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
    

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def __call__(self, anchor, positive, nagative, normalize_feature=False):
        if normalize_feature:
            anchor = normalize(anchor, axis=-1)
            positive = normalize(positive, axis=-1)
            nagative = normalize(nagative, axis=-1)
        loss = self.triplet_loss(anchor, positive, nagative)
        # print(loss)
        return loss
