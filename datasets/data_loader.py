from __future__ import print_function, absolute_import

import random, os
from torch.utils.data import Dataset
from utils.feature_util import FeatureUtil


# def read_image(img_path):
#     """Keep reading image until succeed.
#     This can avoid IOError incurred by heavy IO process."""
#     got_img = False
#     while not got_img:
#         try:
#             img = Image.open(img_path).convert('RGB')
#             got_img = True
#         except IOError:
#             print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#             exit(-1)
#             pass
#     return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform,
                distance_dict=None,
                class_to_nearest_class=None,
                width=300,
                height=300):
        """ Args:
            dataset: should be the list of complete paths
        """
        self.dataset = dataset
        self.transform = transform
        self.distance_dict = distance_dict
        self.class_to_nearest_class = class_to_nearest_class
        self.feature_util = FeatureUtil(width, height)

    def __getitem__(self, item):
        img_p = self.dataset[item]
        pid = img_p.split('_')[-1].split('.')[0]
        p_path = os.path.basename(img_p)
        img = self.feature_util.get_proper_input(img_p, self.transform, ls_form=True)
        if self.distance_dict is None and self.class_to_nearest_class is None:
            return img, pid, p_path

        dirname = os.path.dirname(img_p)
        hard_sample = random.randint(1, 2) % 2 == 0
        if hard_sample:
            random_int = random.randint(0, 39)
            random_int = min(random_int, len(self.distance_dict[pid]) - 1)
            p_input_pic = self.distance_dict[pid][random_int][0] if len(self.distance_dict[pid][random_int]) > 0 else picture_path
        else:
            p_pictures = [x for x in self.dataset if x.split('_')[-1].split('.')[0] == pid]
            random.shuffle(p_pictures)
            p_input_pic = p_pictures[0]
        p_img = self.feature_util.get_proper_input(os.path.join(dirname, os.path.basename(p_input_pic)),
                                                    self.transform, ls_form=True)

        if hard_sample:
            n_pictures = [x for x in self.dataset if x.split('_')[-1].split('.')[0] == self.class_to_nearest_class[pid]]
            if len(n_pictures) == 0:
                n_pictures = [x for x in self.dataset if x.split('_')[-1].split('.')[0] != pid]
        else:
            n_pictures = [x for x in self.dataset if x.split('_')[-1].split('.')[0] != pid]
        random.shuffle(n_pictures)
        n_input_pic = n_pictures[0]
        n_img = self.feature_util.get_proper_input(os.path.join(os.path.dirname(img_p), os.path.basename(n_input_pic)),
                                                    self.transform, ls_form=True)
        return img, p_img, n_img

    def __len__(self):
        return len(self.dataset)


# def dataloader_collate_fn(batch):
#     anchor_ls, anchor_label_ls, anchor_path_ls = zip(*batch)
#     positive_ls, negative_ls = [], []
#     feature_util = FeatureUtil(G.WIDTH, G.HEIGHT)

#     train_root = G.get_value('train_root')
#     train_pictures = G.get_value('train_pictures')
#     distance_dict = G.get_value('distance_dict')
#     class_to_nearest_class = G.get_value('class_to_nearest_class')

#     for cls_idx, picture_path in zip(anchor_label_ls, anchor_path_ls):
#         hard_sample = random.randint(1, 2) % 2 == 0  # decide if use random sample or hard sample
#         if hard_sample and distance_dict is not None and class_to_nearest_class is not None:
#             random_int = random.randint(0, 39)
#             random_int = min(random_int, len(distance_dict[cls_idx]) - 1)
#             p_input_pic = distance_dict[cls_idx][random_int][0] if len(
#                 distance_dict[cls_idx][random_int]) > 0 else picture_path
#         else:
#             p_pictures = [x for x in train_pictures if x.split('_')[-1][:-4] == cls_idx and x != picture_path]
#             random.shuffle(p_pictures)
#             p_input_pic = p_pictures[0] if len(p_pictures) > 0 else picture_path
#         p_input = feature_util.get_proper_input(os.path.join(train_root, os.path.basename(p_input_pic)),
#                                                 TrainTransform(G.WIDTH, G.HEIGHT),
#                                                 ls_form=True)
#         positive_ls.append(p_input)

#         if hard_sample and distance_dict is not None and class_to_nearest_class is not None:
#             n_pictures = [x for x in train_pictures if x.split('_')[-1][:-4] == class_to_nearest_class[cls_idx]]
#             if len(n_pictures) == 0:
#                 n_pictures = [x for x in train_pictures if x.split('_')[-1][:-4] != cls_idx]
#         else:
#             n_pictures = [x for x in train_pictures if x.split('_')[-1][:-4] != cls_idx]
#         random.shuffle(n_pictures)
#         n_input_pic = n_pictures[0]
#         n_input = feature_util.get_proper_input(os.path.join(train_root, os.path.basename(n_input_pic)),
#                                                 TrainTransform(G.WIDTH, G.HEIGHT),
#                                                 ls_form=True)
#         negative_ls.append(n_input)

#     anchor_ls = torch.Tensor(anchor_ls)
#     positive_ls = torch.Tensor(positive_ls)
#     negative_ls = torch.Tensor(negative_ls)
#     return anchor_ls, positive_ls, negative_ls
