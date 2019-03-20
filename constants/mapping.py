# encoding: utf-8

import os
from utils.file_util import pickle_write, pickle_read


from_old_to_new_name = {
	'A': '大头菜1',
	'B': '炒西洋菜',
	'L': '水煮肉片',
	'No': '炒腐皮',
	'R': '南瓜炒肉',
	'TT': '蟹柳',
	'V': '肠粉',
	'X': '豇豆炒蛋',
	'XX': '青椒炒蛋',
	'Y': '腐皮肉卷',
	'Z': '煎肉饼',
	'白菜': '大白菜',
	'白切鸡': '姜葱白切鸡',
	'包菜': '包菜（辣）',
	'不知名炒腊肠': '瓜片炒腊肠',
	'菜花': '青菜花',
	'炒粿条': '炒河粉',
	'炒鸡蛋': '蛋包青瓜',
	'炒青椒': '青椒炒肉',
	'大鸡块': '黄金鸡块',
	'蛋': '咸鸭蛋',
	'冬瓜肉': '冬瓜炒肉',
	'豆腐蛋黄': '豆腐窝蛋',
	'豆角': '豆角炒肉片',
	'豆角炒肉': '豆角烧肉',
	'番茄蛋': '番茄炒蛋',
	'耗烙': '炸鸡蛋',
	'红烧肉': '腊肉',
	'胡萝卜': '胡萝卜炒蛋',
	'黄瓜': '黄瓜片',
	'黄韭菜': '九王炒蛋',
	'黄块': '水煮蛋',
	'鸡丝': '手撕鸡',
	'鸟蛋': '鹌鹑蛋',
	'排豆腐炒肉': '豆腐串炒肉片',
	'肉卷': '面筋',
	'土豆': '土豆炒肉片',
	'血': '猪鸭血',
	'一根肠': '热狗',
	'一陀肉': '卤味肉饼',
}


def get_mapping_dict():
	return pickle_read('mapping_dict.pkl')


def get_mapping_list():
	return pickle_read('mapping_list.pkl')


def main():
	mapping = get_mapping_dict()
	d = {}
	for _id, _name in mapping.items():
		d[_name] = _id
	pickle_write('./mapping_reverse_dict.pkl', d)
	pass


if __name__ == '__main__':
	main()
	pass
