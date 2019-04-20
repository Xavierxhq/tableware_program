import time, random, os
import torch
from torch.utils.data import DataLoader
from utils.meters import AverageMeter
from utils.feature_util import FeatureUtil
from utils.transforms import TrainTransform
import cv_global as G
from datasets.data_loader import ImageDataset, dataloader_collate_fn


def train_using_metriclearning(model, optimizer, criterion, epoch, train_root, train_pictures,
                            batch_size,
                            distance_dict=None,
                            class_to_nearest_class=None
                            ):
    start = time.time()
    model.train()

    losses = AverageMeter()
    is_add_margin = False
    if train_pictures is None:
        train_pictures = os.listdir(train_root)

    # set global variables for dataloader usage
    G.set_value('train_root', train_root)
    G.set_value('train_pictures', train_pictures)
    G.set_value('distance_dict', distance_dict)
    G.set_value('class_to_nearest_class', class_to_nearest_class)

    dataset = ImageDataset([os.path.join(train_root, x) for x in train_pictures],
                           transform=TrainTransform(G.WIDTH, G.HEIGHT))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=dataloader_collate_fn)

    i = 1
    for anchor_ls, positive_ls, negative_ls in train_loader:
        anchor_ls = model(anchor_ls.cuda())
        positive_ls = model(positive_ls.cuda())
        negative_ls = model(negative_ls.cuda())

        loss = criterion(anchor_ls, positive_ls, negative_ls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        if losses.val < 1e-5:
            is_add_margin = True
        # reset for next training "batch"
        if i % 10 == 0:
            print('Epoch: {}[{}/{}]\t'
                  'Loss {:.6f} ({:.6f})\t'
                  .format(epoch, i, len(train_loader),
                          losses.val, losses.mean))
        i += 1

    time_token = time.time() - start
    param_group = optimizer.param_groups
    print('Epoch: [{}]\tEpoch Time {:.1f} s\tLoss {:.6f}\t'
          'Lr {:.2e}'
          .format(epoch, time_token, losses.mean, param_group[0]['lr']))
    return is_add_margin
    

def train_using_metriclearning_with_inception3(model, optimizer, criterion, epoch, train_root, train_pictures, prefix,
                             distance_dict=None,
                             class_to_nearest_class=None):
    start = time.time()
    model.train()

    losses = AverageMeter()
    is_add_margin = False
    feature_util = FeatureUtil(G.WIDTH, G.HEIGHT)
    if train_pictures is None:
        train_pictures = os.listdir(train_root)
    log_freq = int( len(train_pictures) / 6 )

    anchor_ls, positive_ls, negative_ls = [], [], []
    for i, picture_path in enumerate(train_pictures):
        cls_idx = picture_path.split('_')[-1][:-4]

        anchor_input = feature_util.get_proper_input(os.path.join(train_root, picture_path), ls_form=True)
        anchor_ls.append(anchor_input)

        hard_sample = random.randint(1, 2) % 2 == 0  # decide if use random sample or hard sample
        if hard_sample and distance_dict is not None and class_to_nearest_class is not None:
            random_int = random.randint(0, 39)
            random_int = min(random_int, len(distance_dict[cls_idx]) - 1)
            p_input_pic = distance_dict[cls_idx][random_int][0] if len(
                distance_dict[cls_idx][random_int]) > 0 else picture_path
        else:
            p_pictures = [x for x in train_pictures if x.split('_')[-1][:-4] == cls_idx and x != picture_path]
            random.shuffle(p_pictures)
            p_input_pic = p_pictures[0] if len(p_pictures) > 0 else picture_path
        p_input = feature_util.get_proper_input(os.path.join(train_root, p_input_pic), ls_form=True)
        positive_ls.append(p_input)

        if hard_sample and distance_dict is not None and class_to_nearest_class is not None:
            n_pictures = [x for x in train_pictures if x.split('_')[-1][:-4] == class_to_nearest_class[cls_idx]]
            if len(n_pictures) == 0:
                n_pictures = [x for x in train_pictures if x.split('_')[-1][:-4] != cls_idx]
        else:
            n_pictures = [x for x in train_pictures if x.split('_')[-1][:-4] != cls_idx]
        random.shuffle(n_pictures)
        n_input_pic = n_pictures[0]
        n_input = feature_util.get_proper_input(os.path.join(train_root, n_input_pic), ls_form=True)
        negative_ls.append(n_input)

        if ((i+1) == len(train_pictures)) or ((i+1) % 128 == 0):

            anchor_ls = torch.Tensor(anchor_ls)
            positive_ls = torch.Tensor(positive_ls)
            negative_ls = torch.Tensor(negative_ls)

            anchor_ls = anchor_ls.cuda()
            positive_ls = positive_ls.cuda()
            negative_ls = negative_ls.cuda()

            anchor_ls = model(anchor_ls)
            positive_ls = model(positive_ls)
            negative_ls = model(negative_ls)

            loss = criterion(anchor_ls, positive_ls, negative_ls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            if losses.val < 1e-5:
                is_add_margin = True

            """
                reset for next training "batch"
            """
            anchor_ls, positive_ls, negative_ls = [], [], []

        if (i + 1) % log_freq == 0:
            print('Epoch: {}[{}/{}]\t'
                  'Loss {:.6f} ({:.6f})\t'
                  .format(epoch, i + 1, len(train_pictures),
                          losses.val, losses.mean))

    time_token = time.time() - start

    param_group = optimizer.param_groups
    print('Epoch: [{}]\tEpoch Time {:.1f} s\tLoss {:.6f}\t'
          'Lr {:.2e}'
          .format(epoch, time_token, losses.mean, param_group[0]['lr']))
    return is_add_margin
