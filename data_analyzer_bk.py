import os, time, shutil
import torch
import numpy as np
from utils.file_util import pickle_write, pickle_read, write_csv
from tester import dist, get_feature, get_feature_map_average, predict_pictures
from trainers.train_util import load_model
from datasets.prepare_dataset import get_training_set_list


def _write_feature_map(label, feature, file_name, feature_map_dir, weight=1.0):
    _zero = torch.Tensor([[.0 for i in range(2048)]]).cuda()
    _multiplier = torch.Tensor([[weight for i in range(2048)]]).cuda()
    _zero = torch.addcmul(_zero, 1, feature, _multiplier)

    if not os.path.exists(feature_map_dir):
        os.makedirs(feature_map_dir)
    feature_map_name = os.path.join(feature_map_dir, '%s_features.pkl' % label)
    if not os.path.exists(feature_map_name):
        obj = {
            file_name: _zero
        }
    else:
        obj = pickle_read(feature_map_name)
        obj[file_name] = _zero
    pickle_write(feature_map_name, obj)
    return _zero


class DataAnalyzer(object):

    def __init__(self, sample_file_dir, test_dir, num_of_classes, prefix, WIDTH=None, HEIGHT=None):
        self.sample_file_dir = sample_file_dir
        self.test_dir = test_dir
        self.num_of_classes = num_of_classes
        self.prefix = prefix
        pkl_path = './constants/o_id_to_name.pkl' if self.num_of_classes > 50 else './constants/n_id_to_name.pkl'
        self.id_name_dict = pickle_read(pkl_path)
        if not os.path.exists(os.path.join('model', self.prefix, 'visualize_folder')):
            os.makedirs( os.path.join('model', self.prefix, 'visualize_folder') )
        if WIDTH is not None:
            self.WIDTH = WIDTH
        if HEIGHT is not None:
            self.HEIGHT = HEIGHT

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
                _d = dist(feature, _second_feature)
                _key = classid + '-' + _second_classid
                exter_class_distance_dict[_key] = _d
                if neareast_d > _d:
                    neareast_d = _d
                    nearest_id = _second_classid
            class_to_nearest_class[classid] = nearest_id

        pickle_write('./results/temp/%s_exter_class_distances.pkl'%self.prefix.split('/')[0], exter_class_distance_dict)
        print('Time for _calc_exter_class_distance: %.1f s' % (time.time() - t1))
        return exter_class_distance_dict, class_to_nearest_class

    def visualize_exter_class_distance(self, exter_class_distance_dict):
        data_dict = {
            '00DISTANCE': ['(%d)%s'%(i,self.id_name_dict[str(i)]) for i in range(1, self.num_of_classes+1)]
        }
        for first_id in range(1, self.num_of_classes+1):
            data_dict['(%02d)%s'%(first_id,self.id_name_dict[str(first_id)])] = []
            for second_id in range(1, self.num_of_classes+1):
                if first_id == second_id:
                    distance = 0
                else:
                    key = '%d-%d' % (second_id, first_id)
                    if key not in exter_class_distance_dict:
                        key = '%d-%d' % (first_id, second_id)
                    distance = exter_class_distance_dict[key]
                data_dict['(%02d)%s'%(first_id,self.id_name_dict[str(first_id)])].append(distance)
        _save_csv = os.path.join('model', self.prefix, 'visualize_folder', '%s_exter_class_distance_seen.csv'%self.prefix.split('/')[0])
        write_csv(_save_csv, data_dict)

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
                distance_dict[classid][_filename] = dist(avg_feature_dict[classid], _feature)
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

    def visualize_class_variance(self, variance_dict, class_acc=None):
        class_count_dict = pickle_read('./results/temp/%s_class_count_dict_seen.pkl'%self.prefix.split('/')[0])
        data_dict = {
            '0CLASS': ['%s'%(self.id_name_dict[str(i)]) for i in range(1, self.num_of_classes+1)],
            '1VARIANCE': ['%f'%variance_dict[str(i)] for i in range(1, self.num_of_classes+1)],
            '2IMAGES_COUNT': ['%d'%class_count_dict[str(i)] for i in range(1, self.num_of_classes+1)]
        }
        if class_acc is not None:
            data_dict['3ACCURACY'] = ['%f'%class_acc[str(i)] for i in range(1, self.num_of_classes+1)]
        _save_csv = os.path.join('model', self.prefix, 'visualize_folder', '%s_class_variance_seen.csv'%self.prefix.split('/')[0])
        write_csv(_save_csv, data_dict)

    def calc_top3_error(self, predict_dict_path, class_count_dict_path, seen, epoch):
        # calc top 3 error
        predict_dict = pickle_read(predict_dict_path)
        class_count_dict = pickle_read(class_count_dict_path)
        id_name = pickle_read('/home/ubuntu/Program/Dish_recognition/program/constants/mapping_dict.pkl')
        for i in range(1, 95):
            key = '%d-%d' % (i, i)
            if key in predict_dict:
                acc = len(predict_dict[key]) / class_count_dict[str(i)]
                if acc < .8:
                    # start to calc the top 3 error
                    with open( os.path.join('results', 'data_analyze', '%s_top3_error.txt' % self.prefix.split('/')[0]), 'ab+' ) as f:
                        acc_dict = {}
                        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        c = '\r\n\r\n%s (%s)\r\nEposh[%d] Top 3 Error of Class: %d (Dish: %s)\r\n' % (time_str, seen, epoch, i, id_name[str(i)])
                        f.write(c.encode())
                        for j in range(1, 95):
                            if i == j:
                                continue
                            key = '%d-%d' % (i, j)
                            if key in predict_dict:
                                acc = len(predict_dict[key]) / class_count_dict[str(i)] * 100
                                acc_dict[str(j)] = acc
                        acc_dict = sorted(acc_dict.items(), key=lambda x : x[1])
                        acc_dict.reverse()
                        for cls_idx, acc in acc_dict[:3]:
                            c = '\tClass: %s (Dish: %s) : %.2f\r\n' % (cls_idx, id_name[cls_idx], acc)
                            f.write(c.encode())

    def calc_predict_result(self, predict_dict_path, class_count_dict_path):
        # calc predict result
        predict_dict = pickle_read(predict_dict_path)
        class_count_dict = pickle_read(class_count_dict_path)
        data_dict = {
            '00ACCURACY': [self.id_name_dict[str(x)] for x in range(1, self.num_of_classes+1)]
        }
        class_acc = {}
        for i in range(1, self.num_of_classes+1):
            data_dict['(%02d)%s'%(i,self.id_name_dict[str(i)])] = []
            for j in range(1, self.num_of_classes+1):
                key = '%d-%d' % (j, i)
                if key in predict_dict:
                    acc = len(predict_dict[key]) / class_count_dict[str(j)]
                    if i == j:
                        class_acc[str(i)] = acc
                else:
                    acc = 0.0
                data_dict['(%02d)%s'%(i,self.id_name_dict[str(i)])].append(acc)
        _save_csv = os.path.join('model', self.prefix, 'visualize_folder', '%s_prediction_class_pairs.csv'%self.prefix.split('/')[0])
        write_csv(_save_csv, data_dict)
        return class_acc

    def get_acc(self, positive_num, class_count_dict, threshold_l, threshold_h) :
        cnt_r, cnt_a, cnt_c = 0, 0, 0
        class_count_min = threshold_h + 1

        for i in class_count_dict:
            if class_count_dict[i] < threshold_l:
                continue
            if class_count_dict[i] < class_count_min:
                class_count_min = class_count_dict[i]

        threshold_l = class_count_min
        threshold_h = 5 * class_count_min
        for i in class_count_dict:
            if class_count_dict[i] >= threshold_l and class_count_dict[i] <= threshold_h:
                cnt_c += 1
                cnt_r += positive_num[i]
                cnt_a += class_count_dict[i]
                continue
            if class_count_dict[i] > threshold_h:
                cnt_c += 1
                cnt_a += threshold_h
                cnt_r += threshold_h * positive_num[i] / class_count_dict[i]
                continue
            print("ignore:", i)
        print("use", cnt_c, "class")
        return cnt_r / cnt_a

    def analysis_for_inter_exter_acc(self, model_path, WIDTH, HEIGHT, model_type=None):
        model, _ = load_model(model_path)
        model.eval()

        if model_type is not None:
            if model_type == 'resnet':
                WIDTH = HEIGHT = 224
            elif model_type == 'inception3':
                WIDTH = HEIGHT = 299
        feature_map = get_feature_map_average([model],
                                              sample_num_each_cls=5,
                                              sample_file_dir=self.sample_file_dir,
                                              temp_prefix=self.prefix.split('/')[0],
                                              weight_ls=[1.0],
                                              WIDTH=WIDTH,
                                              HEIGHT=HEIGHT)
        acc = predict_pictures(feature_map,
                               [model],
                               [i for i in range(1, self.num_of_classes)],
                               test_dir=self.test_dir,
                               temp_prefix=self.prefix.split('/')[0],
                               seen='seen',
                               weight_ls=[1.0],
                               WIDTH=WIDTH,
                               HEIGHT=HEIGHT)
        print('accuracy in analysis_for_inter_exter_acc:', '%.2f'%(acc*100))

        class_acc = self.calc_predict_result(class_count_dict_path='./results/temp/%s_class_count_dict_seen.pkl'%self.prefix.split('/')[0],
                            predict_dict_path='./results/temp/%s_predict_label_dict_seen.pkl'%self.prefix.split('/')[0])

        all_test_pkls_dir = './results/temp/%s_all_test_pkls' % self.prefix.split('/')[0]

        feature_map_for_all = self._calc_true_avg_feature(all_test_pkls_dir)
        exter_class_distance_dict, _ = self._calc_exter_class_distance(feature_map_for_all)
        distance_dict = self._calc_inter_distance(all_test_pkls_dir)
        variance_dict = self._calc_variance_each_class(distance_dict)

        self.visualize_exter_class_distance(exter_class_distance_dict)
        self.visualize_class_variance(variance_dict, class_acc)

    def _get_avg_feature_for_all(self, models, weight_ls, WIDTH, HEIGHT, test_pictures=None):
        """
            Prediction for all pictures in given test_dir
        """
        pkls_dir = os.path.join('results', 'temp', self.prefix.split('/')[0] + '_all_test_pkls')
        if os.path.exists(pkls_dir):
            shutil.rmtree(pkls_dir)

        if test_pictures is None:
            test_pictures = os.listdir(self.test_dir)

        for index, i in enumerate(test_pictures):
            file_path = os.path.join(self.test_dir, i)
            cls_idx = file_path.split('_')[-1][:-4] # accroding to the directory name

            feature_on_gpu = None
            for weight_index, model in enumerate(models):
                _f = get_feature(file_path, model, WIDTH, HEIGHT)

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
            _write_feature_map(cls_idx, feature_on_gpu,
                                file_path.split('/')[-1],
                                pkls_dir,
                                weight=1.0)
            if (index+1) % 2000 == 0:
                print('Process', index+1, 'images.')

    def analysis_for_exter_class_overlap(self, model_path, model, WIDTH, HEIGHT, top=12):
        if model is None:
            model, _ = load_model(model_path)
        model.eval()
        # use 100 images/class to calculate the class center
        test_pictures = get_training_set_list(self.test_dir,
                                           train_limit=100,
                                           random_training_set=False)
        self._get_avg_feature_for_all([model],
                               weight_ls=[1.0],
                               WIDTH=WIDTH,
                               HEIGHT=HEIGHT,
                               test_pictures=test_pictures)

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

    def analysis_for_exter_class_overlap_v2(self, model_path, model, WIDTH, HEIGHT, top=12):
        #change some parameter
        if model is None:
            model, _ = load_model(model_path)
        model.eval()
        # use 100 images/class to calculate the class center
        test_pictures = get_training_set_list(self.test_dir,
                                           train_limit=100,
                                           random_training_set=False)
        self._get_avg_feature_for_all([model],
                               weight_ls=[1.0],
                               WIDTH=WIDTH,
                               HEIGHT=HEIGHT,
                               test_pictures=test_pictures)

        all_test_pkls_dir = './results/temp/%s_all_test_pkls' % self.prefix.split('/')[0]
        feature_map_for_all = self._calc_true_avg_feature(all_test_pkls_dir)
        exter_class_distance_dict, _ = self._calc_exter_class_distance(feature_map_for_all)
        distance_dict = self._calc_inter_distance(all_test_pkls_dir, feature_map_for_all)
        variance_dict = self._calc_variance_each_class(distance_dict)

        # use exter class distance + class variance to calculate the overlap rate of class pairs
        overlap_rate_dict = {}
        variance_dict_len = len(variance_dict.keys())
        for class_id in range(1, variance_dict_len+1):
            for class_id_second in range(class_id+1, variance_dict_len+1):
                key = '%d-%d' % (class_id, class_id_second)
                if key not in exter_class_distance_dict:
                    key = '%s-%s' % (class_id_second, class_id)
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

    def analysis_for_hard_sample(self, model, test_pictures=None):
        if test_pictures == None:
            test_pictures = os.listdir(self.test_dir)

        self._get_avg_feature_for_all([model],
                               weight_ls=[1.0],
                               WIDTH=self.WIDTH,
                               HEIGHT=self.HEIGHT,
                               test_pictures=test_pictures)

        all_test_pkls_dir = './results/temp/%s_all_test_pkls' % self.prefix.split('/')[0]
        avg_feature_dict = self._calc_true_avg_feature(all_test_pkls_dir)
        _, class_to_nearest_class = self._calc_exter_class_distance(avg_feature_dict)
        distance_dict = self._calc_inter_distance(all_test_pkls_dir)

        for classid, d in distance_dict.items():
            _d = sorted(d.items(), key=lambda x: x[1])
            distance_dict[classid] = _d[-40:]
        return distance_dict, class_to_nearest_class

    def analysis_for_hard_sample_v2(self, model, test_pictures=None):
        #choose overlap rate as reference but not exter_class_distance
        if test_pictures == None:
            test_pictures = os.listdir(self.test_dir)

        self._get_avg_feature_for_all([model],
                               weight_ls=[1.0],
                               WIDTH=self.WIDTH,
                               HEIGHT=self.HEIGHT,
                               test_pictures=test_pictures)

        all_test_pkls_dir = './results/temp/%s_all_test_pkls' % self.prefix.split('/')[0]
        avg_feature_dict = self._calc_true_avg_feature(all_test_pkls_dir)
        #_, class_to_nearest_class = self._calc_exter_class_distance(avg_feature_dict)
        _, overlap_rate_dict_ls = self.analysis_for_exter_class_overlap(model_path = None,
                                                                            model = model,
                                                                            WIDTH = self.WIDTH,
                                                                            HEIGHT = self.HEIGHT)
        class_to_nearest_class = {}
        nearest_overlap = {}
        for _id, rate in overlap_rate_dict_ls:
            [first_id, second_id] = _id.split('-')
            if nearest_overlap.get(first_id) is None:
                nearest_overlap[first_id] = rate
                class_to_nearest_class[first_id] = second_id
            elif nearest_overlap[first_id] < rate:
                nearest_overlap[first_id] = rate
                class_to_nearest_class[first_id] = second_id

            if nearest_overlap.get(second_id) is None:
                nearest_overlap[second_id] = rate
                class_to_nearest_class[second_id] = first_id
            elif nearest_overlap[second_id] < rate:
                nearest_overlap[second_id] = rate
                class_to_nearest_class[second_id] = first_id
        distance_dict = self._calc_inter_distance(all_test_pkls_dir)

        for classid, d in distance_dict.items():
            _d = sorted(d.items(), key=lambda x: x[1])
            distance_dict[classid] = _d[-40:]
        return distance_dict, class_to_nearest_class

    def get_accuracy_for_every_class(self, pkl_path, seen='none'):
        predict_label_dict = pickle_read('./results/temp/%s_predict_label_dict_%s.pkl' % (self.prefix.split('/')[0], seen))
        class_count_dict = pickle_read('./results/temp/%s_class_count_dict_%s.pkl' % (self.prefix.split('/')[0], seen))

        class_count_dict = sorted(class_count_dict.items(), key=lambda x : x[1])
        accuracy_for_every_class = []
        mapping = pickle_read('./constants/mapping_dict.pkl')

        for classid, classcount in class_count_dict:
            key = '%s-%s' % (classid, classid)
            acc = len(predict_label_dict[key]) / classcount if key in predict_label_dict else 0
            accuracy_for_every_class.append({
                'id': classid,
                'accuracy': acc,
                'count': classcount
            })
        pickle_write(pkl_path, accuracy_for_every_class)
        return accuracy_for_every_class

    def visualize_acc_for_every_class(self, pkl_path, acc_for_every_class=None):
        if acc_for_every_class is None:
            acc_for_every_class = pickle_read(pkl_path)
        data_dict = {
            '1CLASS': [],
            '2COUNT': [],
            '3ACCURACY': []
        }
        for obj in acc_for_every_class:
            data_dict['1CLASS'].append(obj['id'])
            data_dict['2COUNT'].append(obj['count'])
            data_dict['3ACCURACY'].append(obj['accuracy']*100)
        write_csv('%s.csv'%pkl_path, data_dict)


if __name__ == '__main__':

    model_path = None
    model_path = '/home/ubuntu/Program/Dish_recognition/program/model/chawdoe-hardsample-o-o-run-1-augmentation/a_o_o_8/121_' \
	                  'resnet_metric_conv0.05.tar'
    prefix = 'augmentation_o_n'
    analyzer = DataAnalyzer(sample_file_dir='/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_base_sample_5/',
                            test_dir='/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_test/',
                            num_of_classes=42, # ? num of class + 1?
                            prefix=prefix)
    analyzer.analysis_for_inter_exter_acc(model_path=model_path, WIDTH=300, HEIGHT=300)

    pass
