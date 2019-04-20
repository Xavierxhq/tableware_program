import os, time, shutil, random
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils.file_util import pickle_write, pickle_read
from utils.feature_util import FeatureUtil
from utils.transforms import TestTransform
from datasets.data_loader import ImageDataset
from trainers.train_util import load_model


class Analyzer(object):
    def __init__(self, sample_file_dir, test_dir, prefix, WIDTH=None, HEIGHT=None):
        self.sample_file_dir = sample_file_dir
        self.test_dir = test_dir
        self.prefix = prefix
        if not os.path.exists(os.path.join('model', self.prefix, 'visualize_folder')):
            os.makedirs( os.path.join('model', self.prefix, 'visualize_folder') )
        if WIDTH is not None:
            self.WIDTH = WIDTH
        if HEIGHT is not None:
            self.HEIGHT = HEIGHT
        self.feature_util = FeatureUtil(WIDTH, HEIGHT)

    def analysis_for_hard_sample(self, model, test_pictures=None):
        if test_pictures == None:
            test_pictures = os.listdir(self.test_dir)

        self._get_avg_feature_for_all([model], test_pictures=test_pictures)

        all_test_pkls_dir = './results/temp/%s_all_test_pkls' % self.prefix.split('/')[0]
        avg_feature_dict = self._calc_true_avg_feature(all_test_pkls_dir)
        _, class_to_nearest_class = self._calc_exter_class_distance(avg_feature_dict)
        distance_dict = self._calc_inter_distance(all_test_pkls_dir)

        for classid, d in distance_dict.items():
            _d = sorted(d.items(), key=lambda x: x[1])
            distance_dict[classid] = _d[-40:]
        return distance_dict, class_to_nearest_class

    def analysis_for_exter_class_overlap(self, model_path, model, WIDTH, HEIGHT, top=12):
        if model is None:
            model, _ = load_model(model_path)
        model.eval()
        # use 100 images/class to calculate the class center
        test_pictures = self._get_training_set_list(100)
        self._get_avg_feature_for_all([model], test_pictures=test_pictures)

        all_test_pkls_dir = './results/temp/%s_all_test_pkls' % self.prefix.split('/')[0]
        feature_map_for_all = self._calc_true_avg_feature(all_test_pkls_dir)
        exter_class_distance_dict, _ = self._calc_exter_class_distance(feature_map_for_all)
        distance_dict = self._calc_inter_distance(all_test_pkls_dir, feature_map_for_all)
        variance_dict = self._calc_variance_each_class(distance_dict)

        # use exter class distance + class variance to calculate the overlap rate of class pairs
        overlap_rate_dict = {}
        variance_dict_len = len(variance_dict.keys())
        for class_id in range(1, variance_dict_len+2):
            for class_id_second in range(class_id+1, variance_dict_len+1):
                key = '%d-%d' % (class_id, class_id_second)
                if key not in exter_class_distance_dict:
                    key = '%s-%s' % (class_id_second, class_id)
                if key not in exter_class_distance_dict:
                    continue
                overlap_rate = (variance_dict[str(class_id)] + variance_dict[str(class_id_second)] - exter_class_distance_dict[key]) / exter_class_distance_dict[key]
                overlap_rate_dict[key] = overlap_rate
        overlap_rate_dict_ls = sorted(overlap_rate_dict.items(), key=lambda x: x[1])
        overlap_rate_dict_ls.reverse()
        _exter_class_top = []
        for key, _ in overlap_rate_dict_ls[:top]:
            first_id = key.split('-')[0]
            second_id = key.split('-')[1]
            if first_id not in _exter_class_top:
                _exter_class_top.append(first_id)
            if second_id not in _exter_class_top:
                _exter_class_top.append(second_id)
        return _exter_class_top, overlap_rate_dict_ls

    def _calc_true_avg_feature(self, feature_pkls_dir):
        t1 = time.time()
        feature_pkls = [x for x in os.listdir(feature_pkls_dir) if 'features.pkl' in x]
        avg_feature_dict = {}
        for pkl in feature_pkls:
            features = pickle_read(os.path.join(feature_pkls_dir, pkl))
            features = list(features.values())
            _avg_feature = np.zeros(shape=features[0].shape)
            for _feature in features:
                _feature = _feature.cpu().detach().numpy()
                _avg_feature += _feature
            _avg_feature /= len(features)
            classid = pkl.split('_')[0]
            avg_feature_dict[classid] = torch.FloatTensor(_avg_feature)

        pickle_write('./results/temp/%s_true_avg_feature_for_each_class.pkl'%self.prefix.split('/')[0], avg_feature_dict)
        print('Time for _calc_true_avg_feature: %.1f s' % (time.time() - t1))
        return avg_feature_dict

    def _calc_exter_class_distance(self, avg_feature_dict):
        """
        Calculate the exter distance for each center vector, the data will be saved and organized as:
        {
            'id-id': distance,
            ...: ...
        }
        """
        t1 = time.time()
        id_feature_ls = [(_id, _feature) for _id, _feature in avg_feature_dict.items()]
        exter_class_distance_dict, class_to_nearest_class = {}, {}
        for _i in range(len(id_feature_ls)):
            classid, feature = id_feature_ls[_i]
            nearest_id, neareast_d = None, 1e6
            for _second_classid, _second_feature in id_feature_ls[_i+1:]:
                _d = self.feature_util.dist(feature, _second_feature)
                _key = classid + '-' + _second_classid
                exter_class_distance_dict[_key] = _d
                if neareast_d > _d:
                    neareast_d = _d
                    nearest_id = _second_classid
            class_to_nearest_class[classid] = nearest_id

        pickle_write('./results/temp/%s_exter_class_distances.pkl'%self.prefix.split('/')[0], exter_class_distance_dict)
        print('Time for _calc_exter_class_distance: %.1f s' % (time.time() - t1))
        return exter_class_distance_dict, class_to_nearest_class

    def _calc_inter_distance(self, feature_map_dir, avg_feature_dict=None):
        """
        Calculate the inter distance inside each class, the data will be saved and organized as:
        {
            'classid': {
                'xxx.png': distance,
                ...: ...
            },
            ...: ...
        }
        """
        t1 = time.time()
        distance_dict = {}

        avg_feature_dict = pickle_read('./results/temp/%s_true_avg_feature_for_each_class.pkl'%self.prefix.split('/')[0])
        for pkl in [x for x in os.listdir(feature_map_dir) if 'features.pkl' in x]:
            classid = pkl.split('_')[0]
            distance_dict[classid] = {}
            for _filename, _feature in pickle_read(os.path.join(feature_map_dir, pkl)).items():
                distance_dict[classid][_filename] = self.feature_util.dist(avg_feature_dict[classid], _feature)
        pickle_write('./results/temp/%s_inter_class_distances.pkl'%self.prefix.split('/')[0], distance_dict)
        print('Time for _calc_inter_distance: %.1f s' % (time.time() - t1))
        return distance_dict

    def _calc_variance_each_class(self, inter_distance=None):
        """
        Calculate the variance of each class, the data will be saved and organized as:
        {
            'classid': variance,
            ...: ...
        }
        """
        t1 = time.time()
        variance_dict = {}
        if inter_distance is None:
            inter_distance = pickle_read('./results/temp/%s_inter_class_distances.pkl'%self.prefix.split('/')[0])
        for classid, distances in inter_distance.items():
            count = len(distances.keys())
            sum_d = .0
            for _, d in distances.items():
                sum_d += d
            variance_dict[classid] = sum_d / count
        pickle_write('./results/temp/%s_variance_each_class.pkl'%self.prefix.split('/')[0], variance_dict)
        print('Time for _calc_variance_each_class: %.1f s' % (time.time() - t1))
        return variance_dict

    def _get_avg_feature_for_all(self, models, test_pictures=None):
            """
                Prediction for all pictures in given test_dir
            """
            pkls_dir = os.path.join('results', 'temp', self.prefix.split('/')[0] + '_all_test_pkls')
            if os.path.exists(pkls_dir):
                shutil.rmtree(pkls_dir)

            if test_pictures is None:
                test_pictures = os.listdir(self.test_dir)
            dataset = ImageDataset([os.path.join(self.test_dir, x) for x in test_pictures],
                                   transform=TestTransform(self.WIDTH, self.HEIGHT))
            test_loader = DataLoader(dataset, batch_size=64, num_workers=2, pin_memory=True)
            index = 0
            for f_ls, l_ls, p_ls in test_loader:
                f_ls = models[0](torch.Tensor(f_ls).cuda())
                for f, l, p in zip(f_ls, l_ls, p_ls):
                    self.feature_util.write_feature_map(l, f, p, pkls_dir, weight=1.0)
                    index += 1
                    if index % 2000 == 0:
                        print('Process', index, 'images.')

    def _get_training_set_list(self, train_limit=10,
                            random_training_set=False,
                            special_classes=None):
        ls = []
        one_level_order = False

        for label in os.listdir(self.test_dir):
            if '.png' or '.jpg' in label:
                one_level_order = True
                break
            images = os.listdir( os.path.join(self.test_dir, label) )
            random.shuffle(images)
            for i in [label + '/' + x for x in images[:train_limit]]:
                ls.append(i)

        if one_level_order and random_training_set:
            ls = os.listdir(self.test_dir)
            ls = ls[:10000]
            print('Fresh training set generated randomly. with', len(ls), 'images randomly.')
        elif one_level_order:
            class_count_dict = {}
            images = os.listdir(self.test_dir)
            random.shuffle(images)
            for image in images:
                cls_idx = image.split('_')[-1][:-4]
                true_train_limit = (train_limit+50) if (special_classes is not None and cls_idx in special_classes) else train_limit
                if cls_idx in class_count_dict and class_count_dict[cls_idx] == true_train_limit:
                    continue
                if cls_idx not in class_count_dict:
                    class_count_dict[cls_idx] = 1
                else:
                    class_count_dict[cls_idx] += 1
                ls.append(image)
            print('Fresh training set generated. with', len(ls), 'images selected.')
        random.shuffle(ls)
        return ls
