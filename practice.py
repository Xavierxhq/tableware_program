import numpy as np
import torch
import time, random, os, shutil

from torch import nn
from torch.backends import cudnn
from utils.loss import TripletLoss
from utils.serialization import save_checkpoint
from tester import evaluate_with_models, test_with_classifier
from data_analyzer import DataAnalyzer
from datasets.prepare_dataset import get_training_set_list_practice
from trainers.metric_trainer import train_using_metriclearning, train_using_metriclearning_v2, train_using_metriclearning_with_inception3
from trainers.classifier_trainer import train_with_classifier, load_testing
from trainers.finetune_trainer import train_for_fine_tune
from trainers.train_util import statistic_manager, load_model, log, freeze_model, load_inception3


def _adjust_learning_rate(optimizer, ep, num_gpu=1):
        if ep < 20:
            lr = 1e-4 * (ep + 1) / 2
        elif ep < 80:
            lr = 1e-3 * num_gpu
        elif ep < 180:
            lr = 1e-4 * num_gpu
        elif ep < 300:
            lr = 1e-5 * num_gpu
        elif ep < 320:
            lr = 1e-5 * 0.1 ** ((ep - 320) / 80) * num_gpu
        elif ep < 400:
            lr = 1e-6
        elif ep < 480:
            lr = 1e-4 * num_gpu
        else:
            lr = 1e-5 * num_gpu
        for p in optimizer.param_groups:
            p['lr'] = lr
        return optimizer


class Trainer(object):

    def __init__(self, model_type='resnet', load_model_path=None, num_of_classes=0, update_conv_layers=.05):
        self.model_type = model_type
        self.load_model_path = load_model_path
        self.num_of_classes = num_of_classes
        self.update_conv_layers = update_conv_layers
        if model_type == 'resnet':
            self.model, self.optim_policy = load_model(model_path=load_model_path, num_of_classes=num_of_classes)
            self.w = 128
            self.h = 128
        if model_type == 'inception3':
            self.model, self.optim_policy = load_inception3(model_path=load_model_path)
            self.w = 299
            self.h = 299
        print('model size: {:.5f}M'.format(sum(p.numel() for p in self.model.parameters()) / 1e6))

    def set_super_training_parameters(self, prefix, train_root, test_root, sample_file_dir, batch_size,
                                    test_unseen_root=None,
                                    start_epoch=1,
                                    train_epoches=100):
        self.prefix = prefix
        self.train_root = train_root
        self.test_root = test_root
        self.sample_file_dir = sample_file_dir
        self.batch_size = batch_size
        self.test_unseen_root = test_unseen_root
        self.start_epoch = start_epoch
        self.train_epoches = train_epoches

    def _train_prepare(self):
        if self.update_conv_layers != 0:
            name_ls = [x for x, p in self.model.named_parameters() if ('fc' not in x and 'full' not in x)]
            old_name_ls = [x for x in name_ls if 'added' not in x]
            update_index = 5 * self.update_conv_layers if self.update_conv_layers >= 1 else int(len(old_name_ls) * (1-self.update_conv_layers))
            if 'r-a-ft' in self.prefix:
                update_index = name_ls.index('base.layer4.1.bn3.bias')
            freeze_model(self.model, update_prefix=name_ls[update_index+1:])

        self.num_train_pids, self.num_train_imgs = statistic_manager(self.train_root)
        self.num_test_pids, self.num_test_imgs = statistic_manager(self.test_root)
        self.num_test_unseen_pids, self.num_test_unseen_imgs = statistic_manager(self.test_unseen_root)
        print('\n', 'TRAINING METHOD:', method)
        print('   TRAINING CLASSES:', ('%4d' % self.num_train_pids),       ', COUNT: ', self.num_train_imgs)
        print('       TEST CLASSES:', ('%4d' % self.num_test_pids),        ', COUNT: ', self.num_test_imgs)
        print('UNSEEN TEST CLASSES:', ('%4d' % self.num_test_unseen_pids), ', COUNT: ', self.num_test_unseen_imgs, '\n')

        """
            Hyper-Parameters for model
        """
        SEED, self.margin, self.lr, self.weight_decay, self.num_gpu, self.step_size = 0, 20.0, .1, 5e-4, 1, 40

        self.save_dir = os.path.join('model', self.prefix)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, 'readme.txt'), 'wb+') as f:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            c = 'This folder is generated on %s, created by %s, V10 means 1w pics/epoch,\r\n'\
            'created for saving trained model, usually,\r\n'\
            'the best model should be the one named with best in it.\r\n\r\n' % (time_str, self.prefix)
            f.write(c.encode())

        torch.manual_seed(SEED)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print('Currently using GPU')
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(SEED)
            self.model = nn.DataParallel(self.model).cuda() # use multi GPUs togather
        else:
            print('Currently using CPU')
        self.tri_criterion = TripletLoss(self.margin)
        self.optimizer = torch.optim.Adam(self.optim_policy, lr=self.lr, weight_decay=self.weight_decay)

    def metric_training(self, balance_testset):
        self._train_prepare()

        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epoches):
            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch + 1)
            next_margin = self.margin

            """
                get a brand new training set
            """
            train_pictures = get_training_set_list_practice(self.train_root,
                                                train_limit=100,
                                                random_training_set=False)
            train_using_metriclearning(self.model,
                                    self.optimizer,
                                    self.tri_criterion,
                                    epoch,
                                    self.train_root,
                                    train_pictures=train_pictures,
                                    prefix=self.prefix, WIDTH=self.w, HEIGHT=self.h, batch_size=self.batch_size)
            # true testing on seen classes
            acc = evaluate_with_models([self.model],
                                _range=self.num_train_pids+1,
                                test_dir=self.test_root,
                                sample_file_dir=self.sample_file_dir,
                                seen='seen',
                                temp_prefix=self.prefix,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.4}%(on seen pictures)'.format(self.margin, epoch, acc * 100))
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='metric',
                note='update:%.2f, on seen'%self.update_conv_layers)

            if self.test_unseen_root is not None:
                # true testing on unseen classes
                acc_unseen = evaluate_with_models([self.model],
                                            _range=self.num_train_pids,
                                            test_dir=self.test_unseen_root,
                                            sample_file_dir=self.sample_file_dir,
                                            seen='unseen',
                                            temp_prefix=self.prefix,
                                            balance_testset=balance_testset)
                log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                    epoch=epoch, accuracy=acc_unseen,
                    train_cls_count=self.num_train_pids,
                    test_cls_count=self.num_test_unseen_pids,
                    method='metric',
                    note='update:%.2f, on unseen'%self.update_conv_layers)
                print('Margin: {}, Epoch: {}, Acc: {:.4}%(on unseen pictures)'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            max_acc = max(acc, max_acc)
            if last_acc == .0:
                last_acc = acc
            else:
                if acc < last_acc:
                    drop_count += 1
                else:
                    drop_count = 0
                last_acc = acc
            if max_acc == acc:
                fail_max_count = 0
            else:
                fail_max_count += 1

            if 'inception3' == self.model_type:
                save_model_name = 'inception_v3_metric_conv%.2f.tar' % (self.update_conv_layers)
            else:
                save_model_name = 'resnet_metric_conv%.2f.tar' % (self.update_conv_layers)
            state_dict = self.model.module.state_dict() if self.use_gpu else self.model.state_dict()

            # save model, and check if its the best model. save as the best model if positive
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
            },
                is_best=acc == max_acc,
                save_dir=self.save_dir,
                filename=save_model_name,
                acc=acc,
                method='metric',
                prefix=self.prefix)

            # if the accuracy keep dropping, stop training
            if drop_count == 10 or fail_max_count == 20:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break

            self.margin = next_margin
        return self.save_dir, max_acc


if __name__ == "__main__":

    """
        Set Hyper-Parameters for training
    """
    while True:
        print('Enter the prefix(for personal identity):')
        prefix = input() # for personal identity, the models you train will be store at: './model/prefix'
        if os.path.exists( os.path.join('model', prefix) ):
            print('Folder existed, re-enter.')
        elif prefix is None or prefix == '':
            print('Prefix must be specified!')
        else:
            break
    method = 'metric'
    model_type = 'resnet' # should be in ['resnet', 'inception3']
    train_root = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/o_train/'
    test_root = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_test/'
    sample_file_dir = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_base_sample_5'

    load_model_path = None
    # load_model_path = './model/pretrained/inception_v3_google-1a9a5a14.pth'
    trainer = Trainer(model_type=model_type,
                    load_model_path=load_model_path)
    trainer.set_super_training_parameters(train_root=train_root,
                                        test_root=test_root,
                                        sample_file_dir=sample_file_dir,
                                        prefix=prefix,
                                        batch_size=128)
    save_dir, maxacc = trainer.metric_training(balance_testset=False)

    best_model_path = './model/keep/resnet_%s_%s_conv0.05_%.2f.tar'%(prefix, method, maxacc * 100)
    shutil.copy(os.path.join(save_dir, '%s_%s.pth.tar' % (prefix, method)),
                best_model_path)

    analyzer = DataAnalyzer(sample_file_dir=sample_file_dir,
                            test_dir=test_root,
                            num_of_classes=42,
                            prefix=prefix)
    analyzer.analysis_for_inter_exter_acc(model_path=best_model_path)
