import time, random, os
from utils.meters import AverageMeter
from tester import get_proper_input
import torch


def train_for_fine_tune(model, optimizer, criterion, epoch, train_root, train_pictures):
    start = time.time()
    model.train()

    losses = AverageMeter()
    is_add_margin = False
    if train_pictures is None:
        train_pictures = os.listdir(train_root)
    log_freq = len(train_pictures)

    anchor_ls, positive_ls, negative_ls = [], [], []
    taken_ls, taken_count = [], 0
    for i, picture_path in enumerate(train_pictures):
        cls_id = picture_path.split('_')[-1][:-4]

        # if the picture from class that has been taken care of, skip
        if cls_id in taken_ls:
            continue

        taken_ls.append(cls_id)
        pics = [x for x in train_pictures if cls_id == x.split('_')[-1][:-4]]
        for i in range(len(pics)):
            p_input = get_proper_input(os.path.join(train_root, pics[i]), ls_form=True)

            for j in range(i + 1, len(pics)):
                # get anchor
                anchor_ls.append(p_input)

                # get positive
                p_input = get_proper_input(os.path.join(train_root, pics[j]), ls_form=True)
                positive_ls.append(p_input)

                # get negative
                n_pics = [x for x in train_pictures if cls_id != x.split('_')[-1][:-4]]
                random.shuffle(n_pics)
                n_input = get_proper_input(os.path.join(train_root, n_pics[5]), ls_form=True)
                negative_ls.append(n_input)

                taken_count += 1
                # to see if update model
                if taken_count % 128 == 0 or taken_count == len(train_pictures):
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

                    # reset for next training "batch"
                    anchor_ls, positive_ls, negative_ls = [], [], []

                # to see if print status
                if taken_count % log_freq == 0:
                    print('Epoch: [{}][{}]\t'
                          'Loss {:.6f} ({:.6f})\t'
                          .format(epoch, taken_count,
                                  losses.val, losses.mean))

    time_token = time.time() - start

    param_group = optimizer.param_groups
    print('Epoch: [{}]\tEpoch Time {:.1f} s\tLoss {:.6f}\t'
          'Lr {:.2e}'
          .format(epoch, time_token, losses.mean, param_group[0]['lr']))
    return is_add_margin
