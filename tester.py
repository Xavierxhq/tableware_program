import os, time, shutil, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.file_util import pickle_write, pickle_read
from trainers.train_util import load_model
from utils.feature_util import FeatureUtil
from utils.transforms import TestTransform
from datasets.data_loader import ImageDataset


class Tester(object) :
    def __init__(self, model, sample_file_dir, test_dir, prefix, input_w, input_h,
                sample_num_each_cls=5,
                model=None,
                tablewares_mapping_path=None):
        self.model = model
        self.sample_file_dir = sample_file_dir
        self.test_dir = test_dir
        self.prefix = prefix
        self.input_w = input_w
        self.input_h = input_h
        self.sample_num_each_cls = sample_num_each_cls
        self.mapping = None
        if tablewares_mapping_path is not None :
            self.mapping = pickle_read(tablewares_mapping_path)
        self.feature_util = FeatureUtil(input_w, input_h)
        self.feature_map = None

    def set_feature_map(self, f):
        self.feature_map = f

    def evaluate_single_file(self, feature, feature_map=None):
        if feature_map is None:
            feature_map = self.feature_map
        result_dict = {}
        for k, v in feature_map.items():
            result_dict[k] = self.feature_util.dist(y1=feature, y2=v)

        result_rank_ls = sorted(result_dict.items(), key=lambda d: d[1])
        pred_l, min_d = result_rank_ls[0][0], result_rank_ls[0][1]
        return pred_l, min_d

    def get_feature_map_average(self) :
        t1 = time.time()

        if self.feature_map is None or force:
            map_d = os.path.join('results', 'temp',, self.prefix + '_feature_pkls')
            if os.path.exists(map_d):
                shutil.rmtree(map_d)
            os.makedirs(map_d)

            for l in os.listdir(self.sample_file_dir):
                full_d = os.path.join(self.sample_file_dir, l)
                for name in os.listdir(full_d):
                    full_f = os.path.join(full_d, name)
                    f = self.feature_util.get_feature(full_f, self.model, TestTransform(self.input_w, self.input_h))
                    self.feature_util.write_feature_map(l, f, name, map_d)

            self.feature_map = self.feature_util.calc_avg_feature_map(map_d)
            print('Time for get_feature_map_average: %.1f s' % (time.time() - t1))
        return self.feature_map

    def predict_pictures(self, test_pictures=None):
        pkls_dir = os.path.join('results', 'temp', self.prefix + '_all_test_pkls')
        if os.path.exists(pkls_dir):
            shutil.rmtree(pkls_dir)

        all_count, positive_count = 0, 0
        predict_dict, class_count_dict = {}, {}

        if test_pictures is None:
            test_pictures = os.listdir(self.test_dir)
        dataset = ImageDataset([os.path.join(self.test_dir, x) for x in test_pictures],
                               transform=TestTransform(self.input_w, self.input_h))
        test_loader = DataLoader(dataset, batch_size=64, num_workers=2, pin_memory=True)
        for f_ls, l_ls, p_ls in test_loader:
            f_ls = self.models[0](torch.Tensor(f_ls).cuda())
            for f, l, p in zip(f_ls, l_ls, p_ls):
                self.feature_util.write_feature_map(l, f, p, pkls_dir, weight=1.0)
                pred_l, min_d = self.evaluate_single_file(f)
                pred_k = l + '-' + pred_l
                if pred_k not in predict_dict:
                    predict_dict[pred_k] = []
                predict_dict[pred_k].append(p)
                if l not in class_count_dict:
                    class_count_dict[l] = 0
                class_count_dict[l] += 1

                if l == pred_l:
                    positive_count += 1
                all_count += 1
                if all_count % 1000 == 0:
                    print('All:', all_count, ', Positive:', positive_count)
        print('test finished. all:', all_count, ', positive:', positive_count)

        pickle_write('./results/temp/%s_predict_label_dict_%s.pkl' % (self.prefix, seen), predict_dict) # store the prediction of each picture
        pickle_write('./results/temp/%s_class_count_dict_%s.pkl' % (self.prefix, seen), class_count_dict) # store the count of each class
        return positive_count / (all_count + 1e-12)

    def evaluate_with_models(self) :
        self.model.eval()
        self.get_feature_map_average()
        return self.predict_pictures()

    def predict(self, model, feature_map, picture_path):
        model.eval()
        if type(feature_map) == str:
            feature_map = pickle_read(feature_map)
        feature = self.feature_util.get_feature(picture_path, model, TestTransform(self.input_w, self.input_h))
        pred_l, min_d = self.evaluate_single_file(feature, feature_map)
        return pred_l, min_d

    # def evaluate_single_file(self, feature, feature_map,
    #                         need_transform=False,
    #                         use_tableware=False,
    #                         cls_idx=""):
    #     if need_transform:
    #         feature = torch.FloatTensor(feature)
    #     result_dict = {}
    #     for k, v in feature_map.items():
    #         if type(v) == dict:
    #             continue
    #         _feature = torch.FloatTensor(v)
    #         result_dict[k] = self.feature_util.dist(y1=feature, y2=_feature)

    #     for k, v in result_dict.items():
    #         for i in np.nditer(result_dict[k]):
    #             result_dict[k] = float(str(i))

    #     result_rank_ls = sorted(result_dict.items(), key=lambda d: d[1])

    #     if use_tableware :
    #         tablewares = self.mapping[cls_idx]
    #         for i in range(len(result_rank_ls)) :
    #             for j in self.mapping[result_rank_ls[i][0]] :
    #                 if j in tablewares :
    #                     return result_rank_ls[i][0], result_rank_ls[i][1]
    #     return result_rank_ls[0][0], result_rank_ls[0][1]

    # def prepare_base_picture_for_class(self, save_dir_path, picture_pool,
    #                                 force_refresh=True):
    #     if os.path.exists(save_dir_path) and force_refresh:
    #         shutil.rmtree(save_dir_path)
    #         os.makedirs(save_dir_path)
    #     elif os.path.exists(save_dir_path):
    #         print('base pictures exist, no operation needed.')
    #         return

    #     sample_list, copy_file_name_list, sample_num_dict = [], [], {}

    #     for i in os.listdir(picture_pool):
    #         class_index = i.split('.')[0].split('_')[-1]
    #         if class_index not in sample_num_dict:
    #             sample_num_dict[class_index] = 1
    #         elif sample_num_dict[class_index] == self.sample_num_each_cls:
    #             continue
    #         else:
    #             sample_num_dict[class_index] += 1
    #         sample_list.append(os.path.join(picture_pool, i))
    #         copy_file_name_list.append(i)
    #     for i in range(len(sample_list)):
    #         class_index = copy_file_name_list[i].split('.')[0].split('_')[-1]
    #         save_dir = os.path.join(save_dir_path, class_index)
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         save_path = os.path.join(save_dir, copy_file_name_list[i])
    #         shutil.copyfile(sample_list[i], save_path)
    #     print('base samples prepared.', self.sample_num_each_cls, 'for 1 class')

    # def prepare_base_picture_from_labeled_pool(self, save_dir_path, name_dict, picture_pool,
    #                                 force_refresh=True):
    #     if os.path.exists(save_dir_path) and force_refresh:
    #         shutil.rmtree(save_dir_path)
    #         os.makedirs(save_dir_path)
    #     for label in os.listdir(picture_pool):
    #         os.makedirs(os.path.join(save_dir_path, name_dict[label]))
    #         images = os.listdir(os.path.join(picture_pool, label))
    #         for image_path in images[:self.sample_num_each_cls]:
    #             shutil.copy(os.path.join(picture_pool, label,image_path),
    #                         os.path.join(save_dir_path, name_dict[label], image_path))
    #     print('base samples prepared.', self.sample_num_each_cls, 'for 1 class')

    # def evaluate_with_models(self, seen='none', force_refresh_base=False,
    #                          weight_ls=None, use_tableware=False) :
    #     if use_tableware and self.mapping is None :
    #         print("Tableware mapping file does not exist.")
    #         return

    #     self.prepare_base_picture_for_class(picture_pool=self.test_dir,
    #                                 save_dir_path=self.sample_file_dir,
    #                                 force_refresh=force_refresh_base)
    #     for model in self.models:
    #         model.eval()
    #     if weight_ls is None or len(self.models) == 1:
    #         weight_ls = [1.0 for _ in self.models]
    #     feature_map = self.get_feature_map_average(weight_ls=weight_ls)
    #     _accuracy = self.predict_pictures(feature_map, seen=seen, use_tableware=use_tableware)
    #     return _accuracy

    # def test_with_classifier(self, model, testloader):
    #     model.eval()

    #     correct_count, all_count = 0, 0
    #     for _x, _y in testloader:
    #         _x = Variable(_x)
    #         _y = Variable(_y)
    #         _x, _y = _x.cuda(), _y.cuda()

    #         output = model(_x)
    #         pred_y = torch.max(output, 1)[1]
    #         correct_count += (pred_y == _y).sum()
    #         all_count += _y.size(0)
    #     acc = float(correct_count) / all_count
    #     print('Test accuracy with classifier: %.4f(%d/%d)' % (acc, correct_count, all_count))
    #     return acc

    # def normalize_test_pictures(self, threshold_h=100, threshold_l=20):
    #     need_normalization = False

    #     class_count_dict, keep_normalized_picture_ls = {}, []
    #     test_pictures = os.listdir(self.test_dir)
    #     random.shuffle(test_pictures)
    #     for image_path in test_pictures:
    #         cls_idx = image_path.split('.')[0].split('_')[-1]
    #         if cls_idx in class_count_dict:
    #             if class_count_dict[cls_idx] == threshold_h:
    #                 need_normalization = True
    #                 continue
    #             class_count_dict[cls_idx] += 1
    #         else:
    #             class_count_dict[cls_idx] = 1
    #         keep_normalized_picture_ls.append(image_path)

    #     exclude_class_ls = []
    #     for cls_idx, cls_count in class_count_dict.items():
    #         if cls_count < threshold_l:
    #             need_normalization = True
    #             exclude_class_ls.append(cls_idx)
    #             print('exclude class:', cls_idx, ', with pictures:', cls_count)

    #     if not need_normalization:
    #         print('test pictures are balanced enough.')
    #         return

    #     keep_normalized_picture_ls = [x for x in keep_normalized_picture_ls if x.split('.')[0].split('_')[-1] not in exclude_class_ls]
    #     exclude_pictures = set(test_pictures) - set(keep_normalized_picture_ls)

    #     test_exclude_dir = '/home/ubuntu/Program/Dish_recognition/dataset/test_exclude/%6.6f' % time.time()
    #     if not os.path.exists(test_exclude_dir):
    #         os.makedirs(test_exclude_dir)
    #     for image in exclude_pictures:
    #         shutil.move( os.path.join(self.test_dir, image),
    #                     os.path.join(test_exclude_dir, image) )

    #     print('test pictures are made more balanced for testing. now get pictures:', len(os.listdir(self.test_dir)))
    #     keep_classes_set = set(class_count_dict.keys()) - set(exclude_class_ls)
    #     keep_test_classes = [x for x in keep_classes_set]
    #     print('Testing', len(keep_test_classes), 'classes.', keep_test_classes)


if __name__ == '__main__':

    """
        Example of evaluate a model
    """
    model, _ = load_model('/home/ubuntu/Program/xhq/TablewareFinetunePro-V3/model/model_fine-tuned.tar')
    tester = Tester(model_path = None,
                    model=model,
                    test_dir = '/home/ubuntu/Program/xhq/TablewareFinetunePro-V3/test_set',
                    sample_file_dir = '/home/ubuntu/Program/xhq/TablewareFinetunePro-V3/base_sample',
                    prefix='tableware',
                    input_w=300,
                    input_h=300)
    acc = tester.evaluate_with_models()
    print('Accuracy:', acc)
