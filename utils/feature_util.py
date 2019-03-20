import os, time
import torch
import numpy as np
from datasets import data_loader
from utils.transforms import TestTransform
from torch.autograd import Variable
from utils.file_util import pickle_write, pickle_read


class FeatureUtil(object):
	def __init__(self, WIDTH, HEIGHT):
		self.WIDTH = WIDTH
		self.HEIGHT = HEIGHT

	def dist(self, y1, y2):
		return torch.sqrt(torch.sum(torch.pow(y1.cpu() - y2.cpu(), 2))).item()

	def get_proper_input(self, img_path, ls_form=False):
		if not os.path.exists(img_path):
			return None
		pic_data = data_loader.read_image(img_path)
		lst = list()
		test = TestTransform(self.WIDTH, self.HEIGHT)
		if ls_form:
			return np.array(test(pic_data))

		lst.append(np.array(test(pic_data)))
		lst = np.array(lst)
		pic_data = Variable(torch.from_numpy(lst))
		return pic_data

	def get_feature(self, img_path, base_model, use_cuda=True):
		x = self.get_proper_input(img_path)
		if use_cuda:
			x = x.cuda()
		y = base_model(x)
		if use_cuda:
			y = y.cuda()
		return y

	def calc_avg_feature_map(self, feature_map_dir, model_count=1):
		t1 = time.time()

		feature_pkls = [x for x in os.listdir(feature_map_dir) if 'features.pkl' in x]
		avg_feature_dict = {}

		for pkl in feature_pkls:
			features = pickle_read(os.path.join(feature_map_dir, pkl))
			features = list(features.values())
			_avg_feature = np.zeros(shape=features[0].shape)
			for _feature in features:
				_feature = _feature.cpu().detach().numpy()
				_avg_feature += _feature

			divider = len(features) if model_count == 1 else (len(features) / 2)
			_avg_feature /= divider

			classid = pkl.split('_')[0]
			avg_feature_dict[classid] = torch.FloatTensor(_avg_feature)

		prefix = feature_map_dir.split('/')[-1].split('_')[0]
		pickle_write('./results/temp/%s_avg_feature_for_each_class.pkl' % prefix, avg_feature_dict)
		print('Time for _calc_avg_feature_map: %.1f s' % (time.time() - t1))
		return avg_feature_dict

	def write_feature_map(self, label, feature, file_name, feature_map_dir, weight=1.0):
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
