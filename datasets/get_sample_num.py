import os
import cv2
import pickle
from triplet_tester import pickle_read
import json


def get_new_sample_num():
    sample_path = '/home/ubuntu/Program/select_new_data'
    dir_list = os.listdir(sample_path)
    # mapping_dict = pickle_read('mapping_dict.pkl')
    # print(mapping_dict)
    num_dict = {}
    for dir in dir_list:
        _dir = os.path.join(sample_path, dir)
        num_dict[dir] = len(os.listdir(_dir))
    # i = 0
    f = open('new_class_num.txt', 'w+')
    for k, v in num_dict.items():
        # if v > 25:
            # f.write(k)
            # f.write(','
            # i += 1
        f.write(str(v))
        f.write('\r\n')
    f.close()
    # print(i)
    # print(num_dict)
    return num_dict


def get_old_sample_num():
    rootdir = "/home/ubuntu/Program/Tableware/data/2018043000/样本/样本"
    train_save_dir = "../datas/dishes_dataset/train/"
    test_save_dir = "../datas/dishes_dataset/test_std/"
    mapping_dict = dict()
    train_num_dict = dict()
    test_num_dict = dict()
    class_list = os.listdir(rootdir)

    test_class_list = [i for i in range(0, 54)]

    # should be careful because the index is [0, 53]
    # but the class is [1, 54]

    # prepare training data
    label = 1

    for cls_name in class_list:  # it will in order by index.
        cls_path = os.path.join(rootdir, cls_name)
        file_list_of_cls = os.listdir(cls_path)
        mapping_dict[str(label)] = cls_name
        train_num_dict[str(label)] = 0

        for file_name in file_list_of_cls:
            file_path = os.path.join(cls_path, file_name)
            train_num_dict[str(label)] += 1
        label += 1

    import json
    print(json.dumps(train_num_dict, indent=4))

    f = open('train_num.txt', 'w+')
    for k, v in train_num_dict.items():
        f.write(str(v))
        f.write('\r\n')
    f.close()


if __name__ == "__main__":

    get_new_sample_num()












