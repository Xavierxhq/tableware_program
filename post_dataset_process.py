"""
    这段代码运行后，
    ./datas/pictures_tobe_proccessed/ 路径的就是需要再一次分配给她们去人工检查的
    ./datas/correct_pictures/ 路径的就是这一次检查后正确的图片
"""
import os, shutil, time, random
from tester import pickle_write


path_feedback = './datas/feedback/'
path_to_cutoff = './datas/pictures_been_cutoff/'
path_to_unlabeled_pictures = './datas/unlabeled_pictures/'
path_tobe_processed = './datas/pictures_tobe_proccessed/'
path_keep_correct_pictures = './datas/correct_pictures/'
path_outlier = './datas/outlier/'
path_tobe_assign = './datas/pictures_tobe_assign'
path_to_base_pictures = './datas/base_pictures_10_nearest/'


def cutoff_processed_pictures_before_manually_check(cutoff_parition=.0):
    if cutoff_parition > .0:
        for label in os.listdir(path_tobe_processed):
            if os.path.exists(os.path.join(path_to_cutoff, label)):
                shutil.rmtree(os.path.join(path_to_cutoff, label))
            os.makedirs(os.path.join(path_to_cutoff, label))
            images = os.listdir(os.path.join(path_tobe_processed, label))
            images = sorted(images, key=lambda i: i)
            for image in images[int(len(images)*cutoff_parition):]:
                shutil.move(os.path.join(path_tobe_processed, label, image), os.path.join(path_to_cutoff, label, image))


def post_process_for_feedback(): # pass test
    # extract out the pictures that pass manual check
    for label in os.listdir(path_feedback):
        if not os.path.exists(os.path.join(path_keep_correct_pictures, label)):
            os.makedirs(os.path.join(path_keep_correct_pictures, label))
        correct_images = os.listdir(os.path.join(path_feedback, label))
        for img in correct_images:
            if '.jpg' not in img:
                continue
            new_true_img = '{}_{}_{}'.format(img.split('_')[0], label, img.split('_')[-2] + '_' + img.split('_')[-1])
            shutil.move(os.path.join(path_tobe_processed, label, new_true_img), os.path.join(path_keep_correct_pictures, label, new_true_img))
    print('feedback pictures be moved to correct directory.')

    # extract out the pictures that fail manual check
    if os.path.exists(path_to_unlabeled_pictures):
        shutil.rmtree(path_to_unlabeled_pictures)
    shutil.copytree(path_tobe_processed, path_to_unlabeled_pictures)
    print('pictures that failed manual check be copied to unlabeled directory.')
    shutil.rmtree(path_tobe_processed) # copy pictures not needed any longer

    # get the pictures from cutoff before manual check
    for label in os.listdir(path_to_cutoff):
        if not os.path.join(os.path.join(path_to_unlabeled_pictures, label)):
            os.makedirs(os.path.join(path_to_unlabeled_pictures, label))
        for image in os.listdir(os.path.join(path_to_cutoff, label)):
            shutil.copy(os.path.join(path_to_cutoff, label, image), os.path.join(path_to_unlabeled_pictures, label, image))
    print('cutoff pictures be copied to unlabeled directory.')
    shutil.rmtree(path_to_cutoff)  # copy pictures not needed any longer


def check_potential_outlier():
    """
        检查那些在一开始被认为是异类的样本，有多少被人工检查后却认为其实是正确的
    """
    count_dict = {}
    for label in os.listdir(path_keep_correct_pictures):
        for image in [x for x in os.listdir(os.path.join(path_keep_correct_pictures, label)) if int(x.split('.')[0]) > 17]:
            if label in count_dict:
                count_dict[label].append(image)
            else:
                count_dict[label] = [image]
            os.remove(os.path.join(path_outlier, image))
    pickle_write('./temp/select_out_outlier.{}.pkl'.format(time.time()), count_dict)
    for k, v in count_dict.items():
        print(k, 'keep outliers count:', len(v))
    if len(count_dict.keys()) == 0:
        print('every outlier failed manual check')


def count_and_assign(count_for_one=1e4):
    work_assign_dict = ['zhou', 'xie', 'chen', 'deng', 'wang', 'xiao']
    if os.path.exists(path_tobe_assign):
        shutil.rmtree(path_tobe_assign)
    d = {}
    for folder in os.listdir(path_tobe_processed):
        images = os.listdir(os.path.join(path_tobe_processed, folder))
        d[folder] = len(images)
    d = sorted(d.items(), key=lambda i: i[1])
    random.shuffle(d)
    c, ls, assign_index = 0, [], 0
    for k, v in d:
        c += v
        # print(k, v)
        ls.append(k)
        if c > count_for_one:
            print('count:', c, 'for', work_assign_dict[assign_index])
            print()
            c = 0
            for label in ls:
                # get pictures to be assigned for manually check
                if not os.path.exists(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'unchecked_' + work_assign_dict[assign_index], label)):
                    os.makedirs(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'unchecked_' + work_assign_dict[assign_index], label))
                for image in os.listdir(os.path.join(path_tobe_processed, label)):
                    shutil.copy(os.path.join(path_tobe_processed, label, image),
                                os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'unchecked_' + work_assign_dict[assign_index], label, image))
                # get base pictures
                if not os.path.exists(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'base_pictures_' + work_assign_dict[assign_index], label)):
                    os.makedirs(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'base_pictures_' + work_assign_dict[assign_index], label))
                for image in os.listdir(os.path.join(path_to_base_pictures, label)):
                    shutil.copy(os.path.join(path_to_base_pictures, label, image),
                                os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'base_pictures_' + work_assign_dict[assign_index], label, image))
            assign_index += 1
            ls = []
    print('count:', c, 'for', work_assign_dict[assign_index])
    if not os.path.exists(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'unchecked_' + work_assign_dict[assign_index])):
        os.makedirs(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'unchecked_' + work_assign_dict[assign_index]))
    for label in ls:
        # get pictures to be assigned for manually check
        if not os.path.exists(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'unchecked_' + work_assign_dict[assign_index], label)):
            os.makedirs(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'unchecked_' + work_assign_dict[assign_index], label))
        for image in os.listdir(os.path.join(path_tobe_processed, label)):
            shutil.copy(os.path.join(path_tobe_processed, label, image),
                        os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'unchecked_' + work_assign_dict[assign_index], label, image))
        # get base pictures
        if not os.path.exists(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'base_pictures_' + work_assign_dict[assign_index], label)):
            os.makedirs(os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'base_pictures_' + work_assign_dict[assign_index], label))
        for image in os.listdir(os.path.join(path_to_base_pictures, label)):
            shutil.copy(os.path.join(path_to_base_pictures, label, image),
                        os.path.join(path_tobe_assign, work_assign_dict[assign_index], 'base_pictures_' + work_assign_dict[assign_index], label, image))


def investigate_correct_images():
    count_dict = {}
    for label in os.listdir(path_keep_correct_pictures):
        l = len( os.listdir(os.path.join(path_keep_correct_pictures, label)) )
        count_dict[label] = l
    count_dict = sorted(count_dict.items(), key=lambda x: x[1])
    for k, v in count_dict:
        print(k, v)


if __name__ == '__main__':
    # cutoff_processed_pictures_before_manually_check(cutoff_parition=.5)
    # count_and_assign(count_for_one=6500)

    # post_process_for_feedback()
    # check_potential_outlier()

    investigate_correct_images()
    pass