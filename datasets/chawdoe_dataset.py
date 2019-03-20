import os, random, shutil
import cv2
from utils.file_util import pickle_write, pickle_read

# modified
origin_data_root = r'/home/ubuntu/Program/Tableware/DataArgumentation/output2/'

output_train_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/train'
output_test_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/test'

output_test_unseen_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/test_unseen'
output_sample_path = r'/home/ubuntu/Program/Tableware/DataArgumentation/dataset/base_sample_5'

output_train_loader = r'/home/ubuntu/Program/Dish_recognition/dataset/train_dataloader'
output_test_loader = r'/home/ubuntu/Program/Dish_recognition/dataset/test_dataloader'


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
    if os.path.exists(output_train_path):
        shutil.rmtree(output_train_path)
    if os.path.exists(output_test_path):
        shutil.rmtree(output_test_path)
    if os.path.exists(output_test_unseen_path):
        shutil.rmtree(output_test_unseen_path)
    if os.path.exists(output_sample_path):
        shutil.rmtree(output_sample_path)

    os.makedirs(output_train_path)
    os.makedirs(output_test_path)
    os.makedirs(output_test_unseen_path)
    os.makedirs(output_sample_path)

    print('old data cleaned.')


def prepare_dataset_train_test(dir_list=None, train_limit=300, test_limit=100, train_classes=None,
                               unseen_test_classes=None):
    if dir_list is None:
        dir_list = os.listdir(origin_data_root)

    train_index, test_index, unseen_test_index = 10000, 10000, 10000
    pic_sum_limit = 25  # > 25 will be choose.
    test_num_dict, train_num_dict = {}, {}

    clean_old_train_test_data()

    # first, prepare a training set with 2-level order, test set with 1-level order
    name_to_id_dict = pickle_read('../constants/mapping_reverse_dict.pkl')
    for index, cls_name in enumerate(dir_list):
        if '.zip' in cls_name:  # 点解有 .zip
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
            train_list = train_and_test_list[:mid]
            test_list = train_and_test_list[mid:]
        else:
            test_list = train_and_test_list[:test_limit]
            # train_list = train_and_test_list[test_limit:train_limit+test_limit]
            train_list = train_and_test_list[test_limit:]

        # train_final_output_dir = os.path.join(output_train_path, name_to_id_dict[cls_name])
        # if not os.path.exists( train_final_output_dir ):
        #     os.makedirs( train_final_output_dir )

        train_final_output_dir = output_train_path

        for pic in train_list:
            src_path = os.path.join(dir_full_path, pic)
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


def get_training_set_list(training_dir, train_limit=300, random_training_set=False):
    ls = []
    one_level_order = False

    for label in os.listdir(training_dir):
        if '.png' in label:
            one_level_order = True
            break
        images = os.listdir(os.path.join(training_dir, label))
        random.shuffle(images)
        for i in [label + '/' + x for x in images[:train_limit]]:
            ls.append(i)

    if one_level_order or random_training_set:
        ls = os.listdir(training_dir)
        random.shuffle(ls)
        ls = ls[:10000]
    print('Fresh training set generated. with', len(ls), 'images.')
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
        if not os.path.exists(os.path.join(output_train_loader, cls_idx)):
            os.makedirs(os.path.join(output_train_loader, cls_idx))
        shutil.copy(os.path.join(train_root, image),
                    os.path.join(output_train_loader, cls_idx, image))

    for image in os.listdir(test_root):
        cls_idx = image.split('_')[-1][:-4]
        if not os.path.exists(os.path.join(output_test_loader, cls_idx)):
            os.makedirs(os.path.join(output_test_loader, cls_idx))
        shutil.copy(os.path.join(test_root, image),
                    os.path.join(output_test_loader, cls_idx, image))

    print('training loader and test loader all set.')


def prepare_data_for_one(pictures_pool, limit=100, ignore_limit=False,
                         output_path=r'/home/ubuntu/Program/Dish_recognition/dataset/test'):
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
            l = len(os.listdir(os.path.join(pictures_pool, label)))
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
    if not os.path.join(fine_tune_root):
        os.makedirs(fine_tune_root)

    index = 100
    for cls_idx in os.listdir(sample_file_dir):
        for image in os.listdir(os.path.join(sample_file_dir, cls_idx)):
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
    copy_files(src_dir='/home/ubuntu/Program/Dish_recognition/dataset/correct_pictures/94_classes_1810/',
               des_dir='/home/ubuntu/Program/fsl/94_classes/base_picture/',
               pic_list=None,
               cls_idx=None)
    pass
    train_list = ''
    # prapare training set
    for pic in train_list:
        src_path = os.path.join(dir_full_path, pic)
        output_name = str(train_index) + '_' + name_to_id_dict[cls_name] + '.png'
        img = cv2.imread(src_path)
        cv2.imwrite(os.path.join('', output_name), img)
        train_index += 1

    test_list = None
    # test set
    for pic in test_list:
        src_path = os.path.join(dir_full_path, pic)
        output_name = str(test_index) + '_' + name_to_id_dict[cls_name] + '.png'
        img = cv2.imread(src_path)
        cv2.imwrite(os.path.join('', output_name), img)  # write to the path
        test_index += 1

    print('training & test set prepared.')

    """
        train set path
        test set path
        base sample path
    """
    return


if __name__ == '__main__':
    # training_for_94_vs_54()
    # exit(1000)

    name_to_id_dict = pickle_read('../constants/mapping_reverse_dict.pkl')
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
    print('Training using', len(train_classes), 'classes.')
    for name in train_classes:
        if name not in name_to_id_dict:
            print('Train class lack:', name)
            exit(200)

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
    print('Test using', len(unseen_test_classes), 'classes.')
    for name in unseen_test_classes:
        if name not in name_to_id_dict:
            print('Test class lack:', name)
            exit(200)

    if len(train_classes) != 75 or len(unseen_test_classes) != 19:
        exit(202)
    prepare_dataset_train_test(train_classes=train_classes, unseen_test_classes=unseen_test_classes)
    # prepare_dataset_train_test(train_classes=train_classes, unseen_test_classes=unseen_test_classes)
    # prepare_dataset_train_test(unseen_test_classes=unseen_test_classes)

    # all_count, class_count_dict = 0, {}
    # ls = os.listdir(output_train_path)
    # for image in ls:
    #     cls_idx = image.split('_')[-1][:-4]
    #     if cls_idx not in class_count_dict:
    #         class_count_dict[cls_idx] = 1
    #         all_count += 1
    #         continue
    #     if class_count_dict[cls_idx] >= 300:
    #         class_count_dict[cls_idx] += 1
    #         continue
    #     if class_count_dict[cls_idx] < 300:
    #         class_count_dict[cls_idx] += 1
    #         all_count += 1
    # print('all_count:', all_count)

    # class_count_dict = sorted(class_count_dict.items(), key=lambda x: x[1])
    # print('Classes with >= 300 pics:', len([x for x in class_count_dict if x[1] >= 300]))

    # id_to_name_dict = pickle_read('../constants/mapping_dict.pkl')
    # for i, c in class_count_dict:
    #     print(i, id_to_name_dict[i], c)

    # backup_datasets([output_train_path, output_train_loader, output_test_path, output_test_loader, output_test_unseen_path, output_sample_path])

    pass
