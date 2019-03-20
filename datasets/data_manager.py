import glob
import re
from os import path as osp
import os

"""Dataset classes"""


class Tableware(object):


    def __init__(self, train_root, test_root, test_unseen_root, need_loader=False, **kwargs):
        self.train_dir = train_root
        self.test_dir = test_root
        self.test_unseen_dir = test_unseen_root

        self._check_before_run()
        # if training set label: {1, 12, 3, 4, 67, 8, 102}, it's necessary to relabel!
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=False)
        test, num_test_pids, num_test_imgs = self._process_dir(self.test_dir, relabel=False)
        test_unseen, num_test_unseen_pids, num_test_unseen_imgs = self._process_dir(self.test_unseen_dir, relabel=False)

        print("=> Tableware loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset      | # ids | # images")
        print("  ----------------------------------------")
        print("  train       | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  test        | {:5d} | {:8d}".format(num_test_pids, num_test_imgs))
        print("  test unseen | {:5d} | {:8d}".format(num_test_unseen_pids, num_test_unseen_imgs))
        print("  ----------------------------------------")

        if need_loader:
            self.train = train
            self.test = test

        self.num_train_pids = num_train_pids
        self.num_test_pids = num_test_pids
        self.num_test_unseen_pids = num_test_unseen_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.isdir(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.isdir(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, relabel=False):
        if dir_path == self.train_dir:
            labels = os.listdir(dir_path)
            num_pids = len(labels)
            if num_pids < 200:
                num_imgs = 0
                for label in labels:
                    l = len(os.listdir(os.path.join(dir_path, label)))
                    num_imgs += l
                return None, num_pids, num_imgs

        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        pattern = re.compile(r'([\d]+)_([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            # print(img_path)
            _, pid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            _, pid= map(int, pattern.search(img_path).groups())

            if pid == -1:
                continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

