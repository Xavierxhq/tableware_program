import os, time, re, shutil, random
import numpy as np
import torch
from torch.autograd import Variable
from utils.file_util import pickle_write, pickle_read
from trainers.train_util import load_model
from utils.feature_util import FeatureUtil


class Tester(object) :
    def __init__(self, model_path, sample_file_dir, test_dir, prefix, input_w, input_h,
                sample_num_each_cls=5,
                model=None,
                tablewares_mapping_path=None):
        self.models, _ = load_model(model_path) if model is None else model
        self.models = [self.models]
        self.sample_file_dir = sample_file_dir
        self.test_dir = test_dir
        self.prefix = prefix
        self.sample_num_each_cls = sample_num_each_cls
        self.mapping = None
        if tablewares_mapping_path is not None :
            self.mapping = pickle_read(tablewares_mapping_path)
        self.feature_util = FeatureUtil(input_w, input_h)

    def evaluate_single_file(self, file_feature, feature_map,
                            need_transform=False,
                            use_tableware=False,
                            cls_idx=""):
        if need_transform:
            file_feature = torch.FloatTensor(file_feature)
        result_dict = {}
        for k, v in feature_map.items():
            if type(v) == dict:
                continue
            _feature = torch.FloatTensor(v)
            result_dict[k] = self.feature_util.dist(y1=file_feature, y2=_feature)

        for k, v in result_dict.items():
            for i in np.nditer(result_dict[k]):
                result_dict[k] = float(str(i))

        result_rank_ls = sorted(result_dict.items(), key=lambda d: d[1])

        if use_tableware :
            tablewares = self.mapping[cls_idx]
            for i in range(len(result_rank_ls)) :
                for j in self.mapping[result_rank_ls[i][0]] :
                    if j in tablewares :
                        return result_rank_ls[i][0], result_rank_ls[i][1]
        return result_rank_ls[0][0], result_rank_ls[0][1]

    def get_feature_map_average(self, weight_ls=None, ignore_limit=False) :
        t1 = time.time()

        if not os.path.exists(self.sample_file_dir):
            print('You should prepare base samples first, call prepare_base_picture_for_class may help.')
            return None

        feature_map_dir = os.path.join('results', 'temp', self.prefix + '_feature_pkls')
        if os.path.exists(feature_map_dir):
            shutil.rmtree(feature_map_dir)
        os.makedirs(feature_map_dir)

        name_to_id_dict = pickle_read('./constants/mapping_reverse_dict.pkl')

        for i, model in enumerate(self.models):
            class_count_dict = {}
            for cls_name in os.listdir(self.sample_file_dir):
                ground_truth_label = name_to_id_dict[cls_name] if cls_name in name_to_id_dict else cls_name

                if ground_truth_label not in class_count_dict:
                    class_count_dict[ground_truth_label] = 0

                dir_full_path = os.path.join(self.sample_file_dir, cls_name)  # open the directory in order.

                for file_name in os.listdir(dir_full_path):
                    if ignore_limit or class_count_dict[ground_truth_label] < self.sample_num_each_cls:
                        file_full_path = os.path.join(dir_full_path, file_name)
                        feature_on_gpu = self.feature_util.get_feature(file_full_path, model)
                        self.feature_util.write_feature_map(label=ground_truth_label,
                                        feature=feature_on_gpu,
                                        file_name=file_name,
                                        feature_map_dir=feature_map_dir,
                                        weight=weight_ls[i])
                        class_count_dict[ground_truth_label] += 1

        avg_feature_map = self.feature_util.calc_avg_feature_map(feature_map_dir, len(self.models))
        print('Time for get_feature_map_average: %.1f s' % (time.time() - t1))
        return avg_feature_map

    def prepare_base_picture_for_class(self, save_dir_path, picture_pool,
                                    force_refresh=True):
        if os.path.exists(save_dir_path) and force_refresh:
            shutil.rmtree(save_dir_path)
            os.makedirs(save_dir_path)
        elif os.path.exists(save_dir_path):
            print('base pictures exist, no operation needed.')
            return

        sample_list, copy_file_name_list, sample_num_dict = [], [], {}

        for i in os.listdir(picture_pool):
            class_index = i.split('.')[0].split('_')[-1]
            if class_index not in sample_num_dict:
                sample_num_dict[class_index] = 1
            elif sample_num_dict[class_index] == self.sample_num_each_cls:
                continue
            else:
                sample_num_dict[class_index] += 1
            sample_list.append(os.path.join(picture_pool, i))
            copy_file_name_list.append(i)
        for i in range(len(sample_list)):
            class_index = copy_file_name_list[i].split('.')[0].split('_')[-1]
            save_dir = os.path.join(save_dir_path, class_index)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, copy_file_name_list[i])
            shutil.copyfile(sample_list[i], save_path)
        print('base samples prepared.', self.sample_num_each_cls, 'for 1 class')

    def prepare_base_picture_from_labeled_pool(self, save_dir_path, name_dict, picture_pool,
                                    force_refresh=True):
        if os.path.exists(save_dir_path) and force_refresh:
            shutil.rmtree(save_dir_path)
            os.makedirs(save_dir_path)
        for label in os.listdir(picture_pool):
            os.makedirs(os.path.join(save_dir_path, name_dict[label]))
            images = os.listdir(os.path.join(picture_pool, label))
            for image_path in images[:self.sample_num_each_cls]:
                shutil.copy(os.path.join(picture_pool, label,image_path),
                            os.path.join(save_dir_path, name_dict[label], image_path))
        print('base samples prepared.', self.sample_num_each_cls, 'for 1 class')

    def predict_pictures(self, feature_map, seen, weight_ls,
                        test_pictures=None, use_tableware=False):
        pkls_dir = os.path.join('results', 'temp', self.prefix + '_all_test_pkls')
        if os.path.exists(pkls_dir):
            shutil.rmtree(pkls_dir)

        all_count, positive_count, pred_to_unseen = 0, 0, 0
        predict_dict, class_count_dict = {}, {}
        if test_pictures is None:
            test_pictures = os.listdir(self.test_dir)

        for i in test_pictures:
            file_path = os.path.join(self.test_dir, i)
            cls_idx = re.split('_', file_path)[-1][:-4] # accroding to the directory name
            all_count += 1

            feature_on_gpu = None
            for weight_index, model in enumerate(self.models) :
                _f = self.feature_util.get_feature(file_path, model)

                _zero = torch.Tensor([[.0 for _ in range(2048)]]).cuda()
                _multiplier = torch.Tensor([[weight_ls[weight_index] for _ in range(2048)]]).cuda()
                _zero = torch.addcmul(_zero, 1, _f, _multiplier)

                if type(_zero) is not torch.Tensor:
                    print('Expected torch.Tensor, got', type(feature_on_gpu))
                    exit(200)
                if feature_on_gpu is None:
                    feature_on_gpu = np.zeros(shape=_zero.shape)
                feature_on_gpu += _zero.cpu().detach().numpy()
            feature_on_gpu = torch.FloatTensor(feature_on_gpu).cuda()
            self.feature_util.write_feature_map(cls_idx, feature_on_gpu,
                                file_path.split('/')[-1],
                                pkls_dir,
                                weight=1.0)

            pred_label, min_distance = self.evaluate_single_file(feature_on_gpu, feature_map, use_tableware=use_tableware, cls_idx=cls_idx)

            pred_key = cls_idx + '-' + pred_label
            if pred_key not in predict_dict:
                predict_dict[pred_key] = [file_path.split('/')[-1]]
            else:
                predict_dict[pred_key].append(file_path.split('/')[-1])
            if cls_idx not in class_count_dict:
                class_count_dict[cls_idx] = 1
            else:
                class_count_dict[cls_idx] += 1

            if cls_idx == pred_label:  # compute the correct num of the class
                positive_count += 1
            if all_count % 1000 == 0:
                print('For now, all:', all_count, ', positive:', positive_count)
        print('all:', all_count, ', positive:', positive_count)

        pickle_write('./results/temp/%s_predict_label_dict_%s.pkl' % (self.prefix, seen), predict_dict) # store the prediction of each picture
        pickle_write('./results/temp/%s_class_count_dict_%s.pkl' % (self.prefix, seen), class_count_dict) # store the count of each class
        return positive_count / (all_count + 1e-12)

    def evaluate_with_models(self, seen='none', force_refresh_base=False,
                             balance_testset=False, weight_ls=None, use_tableware=False) :
        if use_tableware and self.mapping is None :
            print("Tableware mapping file does not exist.")
            return

        self.prepare_base_picture_for_class(picture_pool=self.test_dir,
                                    save_dir_path=self.sample_file_dir,
                                    force_refresh=force_refresh_base)
        for model in self.models:
            model.eval()
        if weight_ls is None or len(self.models) == 1:
            weight_ls = [1.0 for _ in self.models]
        feature_map = self.get_feature_map_average(weight_ls=weight_ls)
        _accuracy = self.predict_pictures(feature_map, seen=seen, weight_ls=weight_ls, use_tableware=use_tableware)
        return _accuracy

    def test_with_classifier(self, model, testloader):
        model.eval()

        correct_count, all_count = 0, 0
        for _x, _y in testloader:
            _x = Variable(_x)
            _y = Variable(_y)
            _x, _y = _x.cuda(), _y.cuda()

            output = model(_x)
            pred_y = torch.max(output, 1)[1]
            correct_count += (pred_y == _y).sum()
            all_count += _y.size(0)
        acc = float(correct_count) / all_count
        print('Test accuracy with classifier: %.4f(%d/%d)' % (acc, correct_count, all_count))
        return acc

    def predict(self, model, feature_map, picture_path):
        model.eval()
        if type(feature_map) == str:
            feature_map = pickle_read(feature_map)
        feature = self.feature_util.get_feature(picture_path)
        pred_label, min_distance = self.evaluate_single_file(feature, feature_map)
        return pred_label, min_distance

    def normalize_test_pictures(self, threshold_h=100, threshold_l=20):
        need_normalization = False

        class_count_dict, keep_normalized_picture_ls = {}, []
        test_pictures = os.listdir(self.test_dir)
        random.shuffle(test_pictures)
        for image_path in test_pictures:
            cls_idx = image_path.split('.')[0].split('_')[-1]
            if cls_idx in class_count_dict:
                if class_count_dict[cls_idx] == threshold_h:
                    need_normalization = True
                    continue
                class_count_dict[cls_idx] += 1
            else:
                class_count_dict[cls_idx] = 1
            keep_normalized_picture_ls.append(image_path)

        exclude_class_ls = []
        for cls_idx, cls_count in class_count_dict.items():
            if cls_count < threshold_l:
                need_normalization = True
                exclude_class_ls.append(cls_idx)
                print('exclude class:', cls_idx, ', with pictures:', cls_count)

        if not need_normalization:
            print('test pictures are balanced enough.')
            return

        keep_normalized_picture_ls = [x for x in keep_normalized_picture_ls if x.split('.')[0].split('_')[-1] not in exclude_class_ls]
        exclude_pictures = set(test_pictures) - set(keep_normalized_picture_ls)

        test_exclude_dir = '/home/ubuntu/Program/Dish_recognition/dataset/test_exclude/%6.6f' % time.time()
        if not os.path.exists(test_exclude_dir):
            os.makedirs(test_exclude_dir)
        for image in exclude_pictures:
            shutil.move( os.path.join(self.test_dir, image),
                        os.path.join(test_exclude_dir, image) )

        print('test pictures are made more balanced for testing. now get pictures:', len(os.listdir(self.test_dir)))
        keep_classes_set = set(class_count_dict.keys()) - set(exclude_class_ls)
        keep_test_classes = [x for x in keep_classes_set]
        print('Testing', len(keep_test_classes), 'classes.', keep_test_classes)


if __name__ == '__main__':

    """
        Example of evaluate a model
    """
    tester = Tester(model_path = '/home/ubuntu/Program/Dish_recognition/program/model/chawdoe-hardsample-o-o-run-1-augmentation/a_o_o_8/121_' \
	                  'resnet_metric_conv0.05.tar',
                    test_dir = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_test/',
                    sample_file_dir = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_base_sample_5',
                    prefix='augmentation_o_o',
                    input_w=300,
                    input_h=300)
    acc = tester.evaluate_with_models()
    print('Accuracy:', acc)
