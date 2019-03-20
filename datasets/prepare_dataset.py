import os, random, shutil
import cv2
from utils.file_util import pickle_write, pickle_read


# modified
# origin_data_root = r'/home/ubuntu/Program/Dish_recognition/dataset/dataset_O/'
# origin_data_root = r'/home/ubuntu/Program/Dish_recognition/dataset/dataset_N/'
# origin_data_root = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset_light/n_base_sample_5/')

# output_train_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/o_train'
# output_test_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/o_test'
# output_train_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_train'
# output_train_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset_light/n5_train'
# output_test_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_test')
# output_test_unseen_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/test_unseen'

# output_sample_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/o_base_sample_5'
# output_sample_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_base_sample_5'

# output_train_loader = r'/home/ubuntu/Program/Dish_recognition/dataset/train_dataloader'
# output_test_loader = r'/home/ubuntu/Program/Dish_recognition/dataset/test_dataloader'

# distortion_train_path = '/home/ubuntu/Program/Tableware/DataArgumentation/output2'

def copy_files(src_dir, des_dir, pic_list, cls_idx, limit=-1):
    output_dir = os.path.join(des_dir, str(cls_idx))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _num = 0
    for pic in pic_list:
        pic_full_path = os.path.join(src_dir, pic)
        des_full_path = os.path.join(output_dir, pic)
        shutil.copyfile(pic_full_path, des_full_path)
        _num += 1
        if _num == limit:
            return


def clean_old_train_test_data():

    # if os.path.exists(output_train_path):
        # shutil.rmtree(output_train_path)
    if os.path.exists(output_test_path):
        shutil.rmtree(output_test_path)
    if os.path.exists(output_test_unseen_path):
        shutil.rmtree(output_test_unseen_path)
    if os.path.exists(output_sample_path):
        shutil.rmtree(output_sample_path)

    # os.makedirs(output_train_path)
    os.makedirs(output_test_path)
    os.makedirs(output_test_unseen_path)
    os.makedirs(output_sample_path)

    print('old data cleaned.')


def prepare_dataset_train_test(dir_list=None, train_limit=300, test_limit=100, train_classes=None, unseen_test_classes=None):
    if dir_list is None:
        dir_list = os.listdir(origin_data_root)

    train_index, test_index, unseen_test_index = 40000, 10000, 10000
    # pic_sum_limit = 25  # > 25 will be choose.
    # test_num_dict, train_num_dict = {}, {}
    # id_to_name_dict = {}

    # clean_old_train_test_data()

    # first, prepare a training set with 2-level order, test set with 1-level order
    # name_to_id_dict = pickle_read('../constants/mapping_reverse_dict.pkl')
    class_index = 93
    for index, cls_name in enumerate(dir_list):
        if '.zip' in cls_name:
            continue

        dir_full_path = os.path.join(origin_data_root, cls_name)
        pic_list = os.listdir(dir_full_path)

        # if len(pic_list) < pic_sum_limit:
        #     continue
        class_index += 1
        # id_to_name_dict[str(class_index)] = cls_name

        # random.shuffle(pic_list)

        # feature_list = pic_list[:5]
        # copy_files(src_dir=dir_full_path,
        #             des_dir=output_sample_path,
        #             pic_list=feature_list,
        #             cls_idx=str(class_index))
        # train_and_test_list = pic_list[5:]

        # if ( len(pic_list) < pic_sum_limit and train_classes is None ) or (train_classes is not None and cls_name not in train_classes):
        #     print(cls_name, 'not included for training.')

        #     if (unseen_test_classes is not None and cls_name in unseen_test_classes):
        #         for pic in train_and_test_list:
        #             src_path = os.path.join(dir_full_path, pic)
        #             output_name = str(unseen_test_index) + '_' + name_to_id_dict[cls_name] + '.png'
        #             img = cv2.imread(src_path)
        #             cv2.imwrite(os.path.join(output_test_unseen_path, output_name), img)
        #             unseen_test_index += 1

        #     continue

        # if len(train_and_test_list) <= (train_limit + test_limit):
        #     mid = int(len(train_and_test_list)/2)
        #     train_list = train_and_test_list[:mid]
        #     test_list = train_and_test_list[mid:]
        # else:
        # test_list = train_and_test_list[:test_limit]
        # train_list = train_and_test_list[test_limit:train_limit+test_limit]
            # train_list = train_and_test_list[test_limit:]

        # train_final_output_dir = os.path.join(output_train_path, name_to_id_dict[cls_name])
        # if not os.path.exists( train_final_output_dir ):
        #     os.makedirs( train_final_output_dir )

        # train_final_output_dir = output_train_path

        for pic in pic_list:
            src_path = os.path.join(dir_full_path, pic)
            output_name = str(train_index) + '_' + str(class_index) + '.png'
            img = cv2.imread(src_path)
            cv2.imwrite(os.path.join(output_train_path, output_name), img)
            train_index += 1

        # for pic in pic_list:
        #     src_path = os.path.join(dir_full_path, pic)
        #     output_name = str(test_index) + '_' + str(class_index) + '.png'
        #     img = cv2.imread(src_path)
        #     cv2.imwrite(os.path.join(output_test_path, output_name), img)
        #     test_index += 1
    # pickle_write('../constants/n_id_to_name.pkl', id_to_name_dict)
    # pickle_write('../constants/o_id_to_name.pkl', id_to_name_dict)

    print('training & test set prepared.')
    return


def prepare_dataset_different_train_test(dir_list=None, train_limit=300, test_limit=100, train_classes=None,
                               unseen_test_classes=None):
    # this function only use to get different distribution data
    if dir_list is None:
        dir_list = os.listdir(origin_data_root)

    train_index, test_index, unseen_test_index = 10000, 10000, 10000
    pic_sum_limit = 25  # > 25 will be choose.
    test_num_dict, train_num_dict = {}, {}

    clean_old_train_test_data()

    # first, prepare a training set with 2-level order, test set with 1-level order
    name_to_id_dict = pickle_read('../constants/mapping_reverse_dict.pkl')
    for index, cls_name in enumerate(dir_list):
        if '.zip' in cls_name:
            continue

        dir_full_path = os.path.join(origin_data_root, cls_name)
        pic_list = os.listdir(dir_full_path)

        random.shuffle(pic_list)

        feature_list = pic_list[:5]
        copy_files(src_dir=dir_full_path,
                   des_dir=output_sample_path,
                   pic_list=feature_list,
                   cls_idx=name_to_id_dict[cls_name])
        train_and_test_list = pic_list[5:]

        if (len(pic_list) < pic_sum_limit and train_classes is None) or (
                train_classes is not None and cls_name not in train_classes):
            print(cls_name, 'not included for training.')

            if (unseen_test_classes is not None and cls_name in unseen_test_classes):
                for pic in train_and_test_list:
                    src_path = os.path.join(dir_full_path, pic)
                    output_name = str(unseen_test_index) + '_' + name_to_id_dict[cls_name] + '.png'
                    img = cv2.imread(src_path)
                    cv2.imwrite(os.path.join(output_test_unseen_path, output_name), img)
                    unseen_test_index += 1

            continue

        if len(train_and_test_list) <= (train_limit + test_limit):
            mid = int(len(train_and_test_list) / 2)
            # train_list = train_and_test_list[:mid]
            test_list = train_and_test_list[mid:]
        else:
            test_list = train_and_test_list[:test_limit]
            # train_list = train_and_test_list[test_limit:]
        train_final_output_dir = output_train_path
        dir_path = os.path.join(distortion_train_path, cls_name)
        train_list = os.listdir(dir_path)

        for pic in train_list:
            src_path = os.path.join(dir_path, pic)
            output_name = str(train_index) + '_' + name_to_id_dict[cls_name] + '.png'
            img = cv2.imread(src_path)
            cv2.imwrite(os.path.join(train_final_output_dir, output_name), img)
            train_index += 1

        for pic in test_list:
            src_path = os.path.join(dir_full_path, pic)
            output_name = str(test_index) + '_' + name_to_id_dict[cls_name] + '.png'
            img = cv2.imread(src_path)
            cv2.imwrite(os.path.join(output_test_path, output_name), img)
            test_index += 1
    print('training & test set prepared.')
    return


def get_training_set_list(training_dir,
                        train_limit=10,
                        random_training_set=False,
                        special_classes=None):
    ls = []
    one_level_order = False
    
    for label in os.listdir(training_dir):
        if '.png' or '.jpg' in label:
            one_level_order = True
            break
        images = os.listdir( os.path.join(training_dir, label) )
        random.shuffle(images)
        for i in [label + '/' + x for x in images[:train_limit]]:
            ls.append(i)

    if one_level_order and random_training_set:
        ls = os.listdir(training_dir)
        ls = ls[:10000]
        print('Fresh training set generated randomly. with', len(ls), 'images randomly.')
    elif one_level_order:
        class_count_dict = {}
        images = os.listdir(training_dir)
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


def get_training_set_list_practice(training_dir, train_limit=300, random_training_set=False):
    ls = []
    class_count_dict = {}
    images = os.listdir(training_dir)
    random.shuffle(images)
    for image in images:
        cls_idx = image.split('_')[-1][:-4]
        if int(cls_idx) > 93:
            ls.append(image)
            continue
        if cls_idx in class_count_dict and class_count_dict[cls_idx] == train_limit:
            continue
        if cls_idx not in class_count_dict:
            class_count_dict[cls_idx] = 1
        else:
            class_count_dict[cls_idx] += 1
        ls.append(image)
    print('Fresh training set generated. with', len(ls), 'images selected.')
    random.shuffle(ls)
    return ls


def prepare_dataloader(train_root, test_root, clean=True):
    if os.path.exists(output_train_loader) and os.path.exists(output_test_loader) and not clean:
        return

    if os.path.exists(output_train_loader):
        shutil.rmtree(output_train_loader)
    if os.path.exists(output_test_loader):
        shutil.rmtree(output_test_loader)

    for image in os.listdir(train_root):
        cls_idx = image.split('_')[-1][:-4]
        if not os.path.exists( os.path.join(output_train_loader, cls_idx) ):
            os.makedirs( os.path.join(output_train_loader, cls_idx) )
        shutil.copy(os.path.join(train_root, image),
                    os.path.join(output_train_loader, cls_idx, image))

    for image in os.listdir(test_root):
        cls_idx = image.split('_')[-1][:-4]
        if not os.path.exists( os.path.join(output_test_loader, cls_idx) ):
            os.makedirs( os.path.join(output_test_loader, cls_idx) )
        shutil.copy(os.path.join(test_root, image),
                    os.path.join(output_test_loader, cls_idx, image))

    print('training loader and test loader all set.')


def prepare_data_for_one(pictures_pool, limit=100, ignore_limit=False,
                    output_path = r'/home/ubuntu/Program/Dish_recognition/dataset/test'):
    # clean_old_train_test_data()
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    train_dict = pickle_read('./mapping_dict_for_train.pkl')
    train_dict_reverse = {}
    for k, v in train_dict.items():
        train_dict_reverse[v] = k

    start_index = 10000
    for label in os.listdir(pictures_pool):
        label_count = 0
        if label not in train_dict_reverse:
            l = len( os.listdir(os.path.join(pictures_pool, label)) )
            print('skip:', label, ', has count:', l)
            continue
        ground_truth_label = train_dict_reverse[label]
        for image_path in os.listdir(os.path.join(pictures_pool, label)):
            final_name = str(start_index) + '_' + ground_truth_label + '.png'
            img = cv2.imread(os.path.join(pictures_pool, label, image_path))
            cv2.imwrite(os.path.join(output_path, final_name), img)
            start_index += 1
            label_count += 1
            if label_count == limit and not ignore_limit:
                break
    print(start_index - 10000, 'pictures prepared for specific one set.')


def prepare_fine_tune_dataset(fine_tune_root, sample_file_dir):
    if os.path.exists(fine_tune_root):
        shutil.rmtree(fine_tune_root)
    os.makedirs(fine_tune_root)

    index = 100
    for cls_idx in os.listdir(sample_file_dir):
        for image in os.listdir( os.path.join(sample_file_dir, cls_idx) ):
            if image.split('_')[-1][:-4] == cls_idx:
                final_name = image
            else:
                final_name = str(index) + '_' + cls_idx + '.png'
            shutil.copy(os.path.join(sample_file_dir, cls_idx, image),
                        os.path.join(fine_tune_root, final_name))
    print('Fine-tune dataset set.')


def backup_datasets(ls):
    import time

    t = str(time.time())
    for path in ls:
        shutil.copytree(path, path + '_bk_' + t)
        print(path + '_bk_' + t, 'backuped and saved.')


def training_for_94_vs_54():
    
    # copy 5 pictures/class to des_dir, as for base pictures
    # only classes all from test set
    """
        n classes, n base features,
        input picture -> input feature,
        n distances,
        number 9, distance smallest, input picture belong to number 9
    """
    train_index, test_index, unseen_test_index = 10000, 10000, 10000


    # dir_list=os.listdir('/home/ubuntu/Program/fsl/data/train/')
    # name_to_id_dict = pickle_read('../constants/mapping_reverse_dict.pkl')
    # for index, cls_name in enumerate(dir_list):
    #     train_dir_full_path = os.path.join('/home/ubuntu/Program/fsl/data/train/', cls_name)
    #     train_list = os.listdir(train_dir_full_path)
    #     random.shuffle(train_list)
    #      # prapare training set
    #     for pic in train_list:
    #         src_path = os.path.join(train_dir_full_path, pic)
    #         output_name = str(train_index) + '_' + name_to_id_dict[cls_name] + '.png'
    #         img = cv2.imread(src_path)
    #         cv2.imwrite(os.path.join('/home/ubuntu/Program/fsl/data/output_train/', output_name), img)
    #         train_index += 1

    # dir_list=os.listdir('/home/ubuntu/Program/fsl/data/test/')
    # for index, cls_name in enumerate(dir_list):
    #     test_dir_full_path = os.path.join(r'/home/ubuntu/Program/fsl/data/test/', cls_name)
    #     test_list = os.listdir(test_dir_full_path)
    #     random.shuffle(test_list)
    #     feature_list = test_list[:5]
    #     copy_files(src_dir=test_dir_full_path,
    #                 des_dir='/home/ubuntu/Program/fsl/data/sample/',
    #                 pic_list=feature_list,
    #                 cls_idx=name_to_id_dict[cls_name])
    #     # test set
    #     for pic in test_list[5:]:
    #         src_path = os.path.join(test_dir_full_path, pic)
    #         output_name = str(test_index) + '_' + name_to_id_dict[cls_name] + '.png'
    #         img = cv2.imread(src_path)
    #         cv2.imwrite(os.path.join(r'/home/ubuntu/Program/fsl/data/output_test/', output_name), img) # write to the path
    #         test_index += 1
            
    dir_list=os.listdir('/home/ubuntu/Program/fsl/data/未改名test/')
    name_to_id_dict = {
    '豆腐2':101,
    '豆角炒肉':102,
    '干蒸':103,
    '咕噜肉':104,
    '荷兰豆炒肉':105,
    '黑椒热狗':106,
    '鸡腿鸡翅':107,
    '煎蛋':108,
    '辣鸡肉':109,
    '辣土豆丝':110,
    '卤蛋':111,
    '萝卜丝':112,
    '麻婆豆腐':113,
    '排骨':114,
    '切开炸鸡':115,
    '热狗':116,
    '肉饼2':117,
    '肉丸2':118,
    '丝瓜炒蛋':119,
    '酸菜':120,
    '汤圆':121,
    '西湖瓜炒蛋':122,
    '鱼头':123,
    '整鱼':124,
    }
    for index, cls_name in enumerate(dir_list):
        test_dir_full_path = os.path.join(r'/home/ubuntu/Program/fsl/data/未改名test/', cls_name)
        test_list = os.listdir(test_dir_full_path)
        random.shuffle(test_list)
        feature_list = test_list[:5]
        copy_files(src_dir=test_dir_full_path,
                    des_dir='/home/ubuntu/Program/fsl/data/sample/',
                    pic_list=feature_list,
                    cls_idx=name_to_id_dict[cls_name])
        # test set
        for pic in test_list[5:]:
            src_path = os.path.join(test_dir_full_path, pic)
            output_name = str(test_index) + '_' + str(name_to_id_dict[cls_name]) + '.png'
            img = cv2.imread(src_path)
            cv2.imwrite(os.path.join(r'/home/ubuntu/Program/fsl/data/output_test/', output_name), img) # write to the path
            test_index += 1
    
    print('training & test set prepared.')

    """
        train set path
        test set path
        base sample path
    """
    return


if __name__ == '__main__':

    # prepare_dataset_train_test()
    n10_train = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/exp_n5_size/n_train_10/'
    output = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/o-n10_train/'
    images_10 = os.listdir(n10_train)

    index = 40000
    for label in range(1, 43):
        img_10 = [x for x in images_10 if str(label) == x.split('_')[-1][:-4]]
        # print(len(img_10))
        random.shuffle(img_10)
        for image in img_10:
            output_iamge_name = '%d_%d.png' % (index, label+93)
            shutil.copy(os.path.join(n10_train, image), os.path.join(output, output_iamge_name))
            index += 1
    # print(index)
    

    train_classes = [
        '洋葱',
        '日本豆腐',
        '粗海带炒肉',
        '炒饭',
        '煎饺',
        '肠粉',
        '小青椒炒蛋',
        '土豆鸡肉',
        '腐竹炒肉',
        '苦瓜炒肉',
        '大头菜',
        '炒腐皮',
        '蟹柳',
        '苦瓜包肉',
        '南瓜炒肉',
        '煎肉饼',
        '烧麦',
        '鸡蛋卷肉',
        '青椒回锅肉',
        '小白菜',
        '咸鸭蛋',
        '火腿片',
        '豇豆炒蛋',
        '芥菜',
        '西洋菜炒猪肉',
        '细海带',
        '青瓜炒肉',
        '面筋',
        '韭菜炒蛋',
        '炒粉',
        '腊肉',
        '黄瓜豆腐蛋',
        '土豆炒肉片',
        '豆腐串炒肉片',
        '冬瓜炒肉',
        '热狗',
        '鸡米花',
        '荷包蛋',
        '肉丸',
        '豆芽',
        '卤鸡腿',
        '大白菜',
        '豆角炒肉片',
        '九王炒蛋',
        '炒莲耦',
        '蛋包青瓜',
        '包菜（辣）',
        '蛋饺',
        '肉饼',
        '炸鸡蛋',
        '麻辣豆腐',
        '手撕鸡',
        '黄金鸡块',
        '姜葱白切鸡',
        '黑肉块',
        '鸡腿',
        '卤味肉饼',
        '土豆排骨',
        '猪鸭血',
        '红烧',
        '黄瓜片',
        '鱼肉',
        '青菜花',
        '苦瓜炒蛋',
        '南瓜',
        '黄菜花',
        '土豆丝',
        '烧鸭',
        '鸭肉',
        '胡萝卜炒蛋',
        '油菜花炒蘑菇',
        '茄子',
        '水煮蛋',
        '番茄炒蛋',
        '大头菜1',
    ]

    unseen_test_classes = [
        '汤',
        '豆腐窝蛋',
        '青椒炒蛋',
        '炒河粉',
        '花生',
        '豆角烧肉',
        '番薯',
        '饺子',
        '蛋包肉丸',
        '米饭',
        '瓜片炒腊肠',
        '鹌鹑蛋',
        '青椒炒肉',
        '肾',
        '炒西洋菜',
        '块豆腐',
        '梅菜扣肉',
        '腐皮肉卷',
        '水煮肉片',
    ]
    pass
