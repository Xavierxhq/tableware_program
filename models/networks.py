# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .resnet_builder import ResNetBuilder


def get_baseline_model(last_stride=1, model_path=None, layers=50, num_of_classes=0):
    model = ResNetBuilder(last_stride, model_path, layers=layers, num_of_classes=num_of_classes)
    optim_policy = model.get_optim_policy()
    return model, optim_policy
