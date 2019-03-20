import torch
import time, os

from torch import nn
from torch.backends import cudnn
from utils.loss import TripletLoss, AugmentedTripletLoss
from utils.serialization import save_checkpoint
from tester import evaluate_with_models, test_with_classifier
from datasets.prepare_dataset import prepare_fine_tune_dataset, get_training_set_list
from trainers.metric_trainer import train_using_metriclearning, train_using_metriclearning_with_inception3
from trainers.classifier_trainer import train_with_classifier, load_testing
from trainers.finetune_trainer import train_for_fine_tune
from trainers.train_util import statistic_manager, load_model, log, freeze_model, load_inception3
from data_analyzer import DataAnalyzer
from utils.file_util import pickle_write


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

    def __init__(self, model_type='resnet',
                load_model_path=None,
                num_of_classes=0,
                update_conv_layers=.05,
                method='metric'):
        self.model_type = model_type
        self.load_model_path = load_model_path
        self.num_of_classes = num_of_classes
        self.update_conv_layers = update_conv_layers
        self.method = method
        if model_type == 'resnet':
            self.model, self.optim_policy = load_model(model_path=load_model_path, num_of_classes=num_of_classes)
        if model_type == 'inception3':
            self.model, self.optim_policy = load_inception3(model_path=load_model_path)
        print('model size: {:.5f}M'.format(sum(p.numel() for p in self.model.parameters()) / 1e6))

    def set_super_training_parameters(self, prefix, train_root, test_root, sample_file_dir, batch_size,
                                    input_w,
                                    input_h,
                                    test_unseen_root=None,
                                    sample_file_dir_unseen=None,
                                    start_epoch=1,
                                    train_epochs=200,
                                    enable_stop_machanism=True
                                    ):
        self.prefix = prefix
        self.train_root = train_root
        self.test_root = test_root
        self.sample_file_dir = sample_file_dir
        self.batch_size = batch_size
        self.test_unseen_root = test_unseen_root
        self.sample_file_dir_unseen = sample_file_dir_unseen
        self.start_epoch = start_epoch
        self.train_epochs = train_epochs
        self.enable_stop_machanism = enable_stop_machanism
        self.w = input_w
        self.h = input_h

    def set_prefix(self, prefix):
        self.prefix = prefix

    def _train_prepare(self):
        if self.update_conv_layers != 0:
            name_ls = [x for x, p in self.model.named_parameters() if ('fc' not in x and 'full' not in x)]
            old_name_ls = [x for x in name_ls if 'added' not in x]
            update_index = 5 * self.update_conv_layers if self.update_conv_layers >= 1 else int(len(old_name_ls) * (1-self.update_conv_layers))
            if 'r-a-ft' in self.prefix and self.model_type == 'inception3':
                update_index = name_ls.index('base.layer4.1.bn3.bias')
            freeze_model(self.model, update_prefix=name_ls[update_index+1:])

        self.num_train_pids, self.num_train_imgs = statistic_manager(self.train_root)
        self.num_test_pids, self.num_test_imgs = statistic_manager(self.test_root)
        self.num_test_unseen_pids, self.num_test_unseen_imgs = statistic_manager(self.test_unseen_root)
        print('\n')
        print('    TRAINING METHOD:', self.method)
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
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            c = 'This folder is generated on {}, marked by {},\r\n'\
            'Here are some settings:\r\n'\
            'train method: {}\r\n'\
            'neural network: {}\r\n'\
            'batch_size: {}\r\n'\
            'training set: {}({} classes, {} images)\r\n'\
            'test set: {}({} classes, {} images)\r\n'\
            'input size: {}x{}\r\n'\
            'and i load a model: {}\r\n\r\n'\
            'And, this folder is created for saving trained model, usually,\r\n'\
            'the best model should be the one named with "{}" in it.\r\n\r\n\r\n' .format (time_str, self.prefix,
                self.method,
                self.model_type,
                self.batch_size,
                self.train_root[21:], self.num_train_pids, self.num_train_imgs,
                self.test_root[21:], self.num_test_pids, self.num_test_imgs,
                self.w, self.h,
                str(self.load_model_path),
                self.prefix)
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
        """
            Training using triditional triplet
        """
        self._train_prepare()

        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            s_time = time.time()

            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch)
            next_margin = self.margin

            """
                get a brand new training set for a new epoch
            """
            if self.num_train_imgs > self.num_train_pids*100:
                train_pictures = get_training_set_list(self.train_root,
                                                    train_limit=100,
                                                    random_training_set=False)
            else:
                train_pictures = None
            train_using_metriclearning(self.model,
                                    self.optimizer,
                                    self.tri_criterion,
                                    epoch,
                                    self.train_root,
                                    train_pictures=train_pictures,
                                    prefix=self.prefix, WIDTH=self.w, HEIGHT=self.h, batch_size=self.batch_size)
            e_time = time.time()

            # true testing on seen classes
            acc = evaluate_with_models([self.model],
                                _range=self.num_train_pids+1,
                                test_dir=self.test_root,
                                sample_file_dir=self.sample_file_dir,
                                seen='seen',
                                temp_prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.3}% (on seen pictures)[Triditional Triplet]'.format(self.margin, epoch, acc * 100))

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
                print('Margin: {}, Epoch: {}, Acc: {:.3}% (on unseen pictures)[Triditional Triplet]'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            # acc = (acc * 0 + acc_unseen * 1 if acc_unseen != -1 else acc)
            max_acc = max(acc, max_acc)
            note = 'update:%.2f, on seen%s' % (self.update_conv_layers, ' - New Top Accuracy' if max_acc==acc else '')
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='metric',
                epoch_time=(e_time-s_time),
                note=note)
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
                prefix=self.prefix.split('/')[0])

            # if the accuracy keep dropping, stop training
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\n[Triditional Triplet]Training finished with: %d epoch, %.2f accuracy' % (epoch, max_acc)
            f.write(c.encode())
        return self.save_dir, max_acc, self.num_test_pids

    def metric_training_v2(self, balance_testset):
        """
            Proposed by keke, sample reweighting
            In this training proceeding, we take Exter-Class-Distance into consideraton,
            means: before starting of an epoch, we evaluate the exter class distance, and strengthen the weight of those
            relatively-closer classes by increasing the account of the images from the classes selected for training.
        """
        self._train_prepare()

        analyzer = DataAnalyzer(sample_file_dir=self.sample_file_dir,
                                test_dir=self.train_root,
                                num_of_classes=self.num_train_pids,
                                prefix=self.prefix)
        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        overlap_dict = {}
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch)
            next_margin = self.margin

            """
                get a brand new training set for a new epoch
            """
            if self.num_train_imgs > self.num_train_pids*100:
                if epoch == 1:
                    true_exter_class_top = None
                elif epoch % 5 == 0 or epoch == 2:
                    true_exter_class_top = exter_class_top
                else:
                    pass
                if true_exter_class_top is not None:
                    print('length of true_exter_class_top:', len(true_exter_class_top), ', and:', true_exter_class_top)
                train_pictures = get_training_set_list(self.train_root,
                                                       train_limit=70,
                                                       random_training_set=False,
                                                       special_classes=true_exter_class_top)
            else:
                train_pictures = None

            train_using_metriclearning(self.model,
                                    self.optimizer,
                                    self.tri_criterion,
                                    epoch,
                                    self.train_root,
                                    train_pictures=train_pictures,
                                    prefix=self.prefix, WIDTH=self.w, HEIGHT=self.h, batch_size=self.batch_size)

            # true testing on seen classes
            exter_class_top, overlap_rate_dict_ls = analyzer.analysis_for_exter_class_overlap(model_path=None,
                                                                                                model=self.model,
                                                                                                WIDTH=self.w,
                                                                                                HEIGHT=self.h)
            if epoch % 5 == 0 or epoch == 2:
                overlap_dict[epoch] = overlap_rate_dict_ls
            acc = evaluate_with_models([self.model],
                                _range=self.num_train_pids+1,
                                test_dir=self.test_root,
                                sample_file_dir=self.sample_file_dir,
                                seen='seen',
                                temp_prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.3}%, Top overlap rate: {:.4} (on seen pictures)[Sample Re-weighting]'.format(self.margin, epoch, acc * 100, overlap_rate_dict_ls[0][1]))

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
                print('Margin: {}, Epoch: {}, Acc: {:.3}% (on unseen pictures)[Sample Re-weighting]'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            max_acc = max(acc, max_acc)
            note = 'update:%.2f, on seen%s' % (self.update_conv_layers, ' - New Top Accuracy' if max_acc==acc else '')
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='metric',
                note=note)

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
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break
            if overlap_rate_dict_ls[0][1] < .1:
                print('Top exter class overlap rate reach a smaller value than threshold, stop in epoch %d\n' % epoch)
                break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\n[Sample Re-weighting]Training finished with: %d epoch, %.2f%% accuracy.' % (epoch, max_acc*100)
            f.write(c.encode())
        pickle_write('./results/temp/%s_v2_overlap_rate_dict.pkl' % self.prefix, overlap_dict)
        return self.save_dir, max_acc, self.num_test_pids

    def metric_training_v3(self, balance_testset):
        """
            Proposed by yaxuan, revise triplet
            use augmentated objective functionin
        """
        self._train_prepare()
        # use augmentated loss
        self.tri_criterion = AugmentedTripletLoss()

        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch)
            next_margin = self.margin

            """
                get a brand new training set for a new epoch
            """
            if self.num_train_imgs > self.num_train_pids*100:
                train_pictures = get_training_set_list(self.train_root,
                                                    train_limit=100,
                                                    random_training_set=False)
            else:
                train_pictures = None
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
                                WIDTH=self.w,
                                HEIGHT=self.h,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.3}% (on seen pictures)[Augmented Triplet Loss]'.format(self.margin, epoch, acc * 100))

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
                print('Margin: {}, Epoch: {}, Acc: {:.3}% (on unseen pictures)[Augmented Triplet Loss]'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            # acc = (acc * 0 + acc_unseen * 1 if acc_unseen != -1 else acc)
            max_acc = max(acc, max_acc)
            note = 'update:%.2f, on seen%s' % (self.update_conv_layers, ' - New Top Accuracy' if max_acc==acc else '')
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='metric',
                note=note)
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
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\n[Augmented Triplet Loss]Training finished with: %d epoch, %.2f%% accuracy' % (epoch, max_acc)
            f.write(c.encode())
        return self.save_dir, max_acc, self.num_test_pids

    def metric_training_v4(self, balance_testset):
        """
            Training using hard sample
        """
        self._train_prepare()

        analyzer = DataAnalyzer(sample_file_dir=self.sample_file_dir,
                                test_dir=self.train_root,
                                num_of_classes=self.num_train_pids,
                                prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h)
        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch)
            next_margin = self.margin

            # get a brand new training set for a new epoch
            if self.num_train_imgs > self.num_train_pids*100:
                train_pictures = get_training_set_list(self.train_root,
                                                       train_limit=100,
                                                       random_training_set=False)
            else:
                train_pictures = None

            # and then go through the training set, to get data needed for hard-sample
            if epoch % 5 == 0 or epoch == 1:
                distance_dict, class_to_nearest_class = analyzer.analysis_for_hard_sample(self.model, test_pictures=train_pictures)

            train_using_metriclearning(self.model,
                                    self.optimizer,
                                    self.tri_criterion,
                                    epoch,
                                    self.train_root,
                                    train_pictures=train_pictures,
                                    prefix=self.prefix,
                                    WIDTH=self.w,
                                    HEIGHT=self.h,
                                    batch_size=self.batch_size,
                                    distance_dict=distance_dict,
                                    class_to_nearest_class=class_to_nearest_class)
            # true testing on seen classes
            acc = evaluate_with_models([self.model],
                                _range=self.num_train_pids+1,
                                test_dir=self.test_root,
                                sample_file_dir=self.sample_file_dir,
                                seen='seen',
                                temp_prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.3}% (on seen pictures)[Hard Sample]'.format(self.margin, epoch, acc * 100))

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
                print('Margin: {}, Epoch: {}, Acc: {:.3}% (on unseen pictures)[Hard Sample]'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            max_acc = max(acc, max_acc)
            note = 'update:%.2f, on seen%s' % (self.update_conv_layers, ' - New Top Accuracy' if max_acc==acc else '')
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='metric',
                note=note)

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
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\n[Hard Sample] Training finished with: %d epoch, %.2f%% accuracy.' % (epoch, max_acc*100)
            f.write(c.encode())
        return self.save_dir, max_acc, self.num_test_pids

    def metric_training_v5(self, balance_testset):
        """
            Training using hard sample + sample re-weighting(proposed by keke)
        """
        self._train_prepare()

        analyzer = DataAnalyzer(sample_file_dir=self.sample_file_dir,
                                test_dir=self.train_root,
                                num_of_classes=self.num_train_pids,
                                prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h)
        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        max_acc_unseen = .0
        overlap_dict = {}
        epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            s_time = time.time()

            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch)
            next_margin = self.margin

            # get a brand new training set for a new epoch
            if self.num_train_imgs > self.num_train_pids*100:
                if epoch == self.start_epoch:
                    true_exter_class_top = None
                elif epoch % 5 == 0 or epoch == (self.start_epoch+1):
                    true_exter_class_top = exter_class_top
                else:
                    pass
                if true_exter_class_top is not None:
                    print('length of true_exter_class_top:', len(true_exter_class_top), ', and:', true_exter_class_top)
                train_pictures = get_training_set_list(self.train_root,
                                                       train_limit=70,
                                                       random_training_set=False,
                                                       special_classes=true_exter_class_top)
            else:
                train_pictures = None

            # and then go through the training set, to get data needed for hard-sample
            if epoch % 5 == 0 or epoch == self.start_epoch:
                distance_dict, class_to_nearest_class = analyzer.analysis_for_hard_sample(self.model, test_pictures=train_pictures)

            train_using_metriclearning(self.model,
                                    self.optimizer,
                                    self.tri_criterion,
                                    epoch,
                                    self.train_root,
                                    train_pictures=train_pictures,
                                    prefix=self.prefix,
                                    WIDTH=self.w,
                                    HEIGHT=self.h,
                                    batch_size=self.batch_size,
                                    distance_dict=distance_dict,
                                    class_to_nearest_class=class_to_nearest_class)
            exter_class_top, overlap_rate_dict_ls = analyzer.analysis_for_exter_class_overlap(model_path=None,
                                                                                                model=self.model,
                                                                                                WIDTH=self.w,
                                                                                                HEIGHT=self.h)
            e_time = time.time()
            if epoch % 5 == 0 or epoch == 2:
                overlap_dict[epoch] = overlap_rate_dict_ls
            # true testing on seen classes
            acc = evaluate_with_models([self.model],
                                _range=self.num_train_pids+1,
                                test_dir=self.test_root,
                                sample_file_dir=self.sample_file_dir,
                                seen='seen',
                                temp_prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.3}%, Top overlap rate: {:.4} (on seen pictures)[Hard Sample + Sample Re-weighting]'.format(self.margin, epoch, acc * 100, overlap_rate_dict_ls[0][1]))

            if self.test_unseen_root is not None:
                # true testing on unseen classes
                acc_unseen = evaluate_with_models([self.model],
                                            _range=self.num_train_pids,
                                            test_dir=self.test_unseen_root,
                                            sample_file_dir=self.sample_file_dir_unseen,
                                            seen='unseen',
                                            temp_prefix=self.prefix,
                                            WIDTH=self.w,
                                            HEIGHT=self.h,
                                            balance_testset=balance_testset)
                max_acc_unseen = max(max_acc_unseen, acc_unseen)
                note = 'update:%.2f, on unseen%s' % (self.update_conv_layers, ' - New Unseen Accuracy' if max_acc_unseen==acc_unseen else '')
                log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                    epoch=epoch, accuracy=acc_unseen,
                    train_cls_count=self.num_train_pids,
                    test_cls_count=self.num_test_unseen_pids,
                    method='metric',
                    note=note)
                print('Margin: {}, Epoch: {}, Acc: {:.3}% (on unseen pictures)[Hard Sample + Sample Re-weighting]'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            max_acc = max(acc, max_acc)
            note = 'update:%.2f, on seen%s' % (self.update_conv_layers, ' - New Seen Accuracy' if max_acc==acc else '')
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='metric',
                epoch_time=(e_time-s_time),
                note=note)

            if epoch == self.start_epoch:
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
                prefix=self.prefix.split('/')[0])

            # if the accuracy keep dropping, stop training
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break
            # if overlap_rate_dict_ls[0][1] < .1:
            #     print('Top exter class overlap rate reach a smaller value than threshold, stop in epoch %d\n' % epoch)
            #     break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\n[Hard Sample + Sample Re-weighting] Training finished with: %d epoch, %.2f%% accuracy.' % (epoch, max_acc*100)
            f.write(c.encode())
        pickle_write('./results/temp/%s_v5_overlap_rate_dict.pkl' % self.prefix, overlap_dict)
        return self.save_dir, max_acc, self.num_test_pids

    def metric_training_v6(self, balance_testset):
        """
            Training using hard sample + sample re-weighting(proposed by keke)
            Hard sample use overlap rate as reference
        """
        self._train_prepare()

        analyzer = DataAnalyzer(sample_file_dir=self.sample_file_dir,
                                test_dir=self.train_root,
                                num_of_classes=self.num_train_pids,
                                prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h)
        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        max_acc_unseen = .0
        overlap_dict = {}
        epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            s_time = time.time()

            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch)
            next_margin = self.margin

            # get a brand new training set for a new epoch
            if self.num_train_imgs > self.num_train_pids*100:
                if epoch == self.start_epoch:
                    true_exter_class_top = None
                elif epoch % 5 == 0 or epoch == (self.start_epoch+1):
                    true_exter_class_top = exter_class_top
                else:
                    pass
                if true_exter_class_top is not None:
                    print('length of true_exter_class_top:', len(true_exter_class_top), ', and:', true_exter_class_top)
                train_pictures = get_training_set_list(self.train_root,
                                                       train_limit=70,
                                                       random_training_set=False,
                                                       special_classes=true_exter_class_top)
            else:
                train_pictures = None

            # and then go through the training set, to get data needed for hard-sample
            if epoch % 5 == 0 or epoch == self.start_epoch:
                distance_dict, class_to_nearest_class = analyzer.analysis_for_hard_sample_v2(self.model, test_pictures=train_pictures)

            train_using_metriclearning(self.model,
                                    self.optimizer,
                                    self.tri_criterion,
                                    epoch,
                                    self.train_root,
                                    train_pictures=train_pictures,
                                    prefix=self.prefix,
                                    WIDTH=self.w,
                                    HEIGHT=self.h,
                                    batch_size=self.batch_size,
                                    distance_dict=distance_dict,
                                    class_to_nearest_class=class_to_nearest_class)
            exter_class_top, overlap_rate_dict_ls = analyzer.analysis_for_exter_class_overlap(model_path=None,
                                                                                                model=self.model,
                                                                                                WIDTH=self.w,
                                                                                                HEIGHT=self.h)
            e_time = time.time()
            if epoch % 5 == 0 or epoch == 2:
                overlap_dict[epoch] = overlap_rate_dict_ls
            # true testing on seen classes
            acc = evaluate_with_models([self.model],
                                _range=self.num_train_pids+1,
                                test_dir=self.test_root,
                                sample_file_dir=self.sample_file_dir,
                                seen='seen',
                                temp_prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.3}%, Top overlap rate: {:.4} (on seen pictures)[Hard Sample + Sample Re-weighting]'.format(self.margin, epoch, acc * 100, overlap_rate_dict_ls[0][1]))

            if self.test_unseen_root is not None:
                # true testing on unseen classes
                acc_unseen = evaluate_with_models([self.model],
                                            _range=self.num_train_pids,
                                            test_dir=self.test_unseen_root,
                                            sample_file_dir=self.sample_file_dir_unseen,
                                            seen='unseen',
                                            temp_prefix=self.prefix,
                                            WIDTH=self.w,
                                            HEIGHT=self.h,
                                            balance_testset=balance_testset)
                max_acc_unseen = max(max_acc_unseen, acc_unseen)
                note = 'update:%.2f, on unseen%s' % (self.update_conv_layers, ' - New Unseen Accuracy' if max_acc_unseen==acc_unseen else '')
                log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                    epoch=epoch, accuracy=acc_unseen,
                    train_cls_count=self.num_train_pids,
                    test_cls_count=self.num_test_unseen_pids,
                    method='metric',
                    note=note)
                print('Margin: {}, Epoch: {}, Acc: {:.3}% (on unseen pictures)[Hard Sample + Sample Re-weighting]'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            max_acc = max(acc, max_acc)
            note = 'update:%.2f, on seen%s' % (self.update_conv_layers, ' - New Seen Accuracy' if max_acc==acc else '')
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='metric',
                epoch_time=(e_time-s_time),
                note=note)

            if epoch == self.start_epoch:
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
                prefix=self.prefix.split('/')[0])

            # if the accuracy keep dropping, stop training
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break
            # if overlap_rate_dict_ls[0][1] < .1:
            #     print('Top exter class overlap rate reach a smaller value than threshold, stop in epoch %d\n' % epoch)
            #     break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\n[Hard Sample + Sample Re-weighting] Training finished with: %d epoch, %.2f%% accuracy.' % (epoch, max_acc*100)
            f.write(c.encode())
        pickle_write('./results/temp/%s_v5_overlap_rate_dict.pkl' % self.prefix, overlap_dict)
        return self.save_dir, max_acc, self.num_test_pids

    def metric_training_v7(self, balance_testset):
        """
            Training using hard sample + sample re-weighting(proposed by keke)
            change every five to every two epoch sample re-weighting
        """
        self._train_prepare()

        analyzer = DataAnalyzer(sample_file_dir=self.sample_file_dir,
                                test_dir=self.train_root,
                                num_of_classes=self.num_train_pids,
                                prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h)
        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        max_acc_unseen = .0
        overlap_dict = {}
        epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            s_time = time.time()

            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch)
            next_margin = self.margin

            # get a brand new training set for a new epoch
            if self.num_train_imgs > self.num_train_pids*100:
                if epoch == self.start_epoch:
                    true_exter_class_top = None
                elif epoch % 2 == 0 or epoch == (self.start_epoch+1):
                    true_exter_class_top = exter_class_top
                else:
                    pass
                if true_exter_class_top is not None:
                    print('length of true_exter_class_top:', len(true_exter_class_top), ', and:', true_exter_class_top)
                train_pictures = get_training_set_list(self.train_root,
                                                       train_limit=70,
                                                       random_training_set=False,
                                                       special_classes=true_exter_class_top)
            else:
                train_pictures = None

            # and then go through the training set, to get data needed for hard-sample
            if epoch % 2 == 0 or epoch == self.start_epoch:
                distance_dict, class_to_nearest_class = analyzer.analysis_for_hard_sample(self.model, test_pictures=train_pictures)

            train_using_metriclearning(self.model,
                                    self.optimizer,
                                    self.tri_criterion,
                                    epoch,
                                    self.train_root,
                                    train_pictures=train_pictures,
                                    prefix=self.prefix,
                                    WIDTH=self.w,
                                    HEIGHT=self.h,
                                    batch_size=self.batch_size,
                                    distance_dict=distance_dict,
                                    class_to_nearest_class=class_to_nearest_class)
            exter_class_top, overlap_rate_dict_ls = analyzer.analysis_for_exter_class_overlap(model_path=None,
                                                                                                model=self.model,
                                                                                                WIDTH=self.w,
                                                                                                HEIGHT=self.h)
            e_time = time.time()
            if epoch % 2 == 0 or epoch == 2:
                overlap_dict[epoch] = overlap_rate_dict_ls
            # true testing on seen classes
            acc = evaluate_with_models([self.model],
                                _range=self.num_train_pids+1,
                                test_dir=self.test_root,
                                sample_file_dir=self.sample_file_dir,
                                seen='seen',
                                temp_prefix=self.prefix,
                                WIDTH=self.w,
                                HEIGHT=self.h,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.3}%, Top overlap rate: {:.4} (on seen pictures)[Hard Sample + Sample Re-weighting]'.format(self.margin, epoch, acc * 100, overlap_rate_dict_ls[0][1]))

            if self.test_unseen_root is not None:
                # true testing on unseen classes
                acc_unseen = evaluate_with_models([self.model],
                                            _range=self.num_train_pids,
                                            test_dir=self.test_unseen_root,
                                            sample_file_dir=self.sample_file_dir_unseen,
                                            seen='unseen',
                                            temp_prefix=self.prefix,
                                            WIDTH=self.w,
                                            HEIGHT=self.h,
                                            balance_testset=balance_testset)
                max_acc_unseen = max(max_acc_unseen, acc_unseen)
                note = 'update:%.2f, on unseen%s' % (self.update_conv_layers, ' - New Unseen Accuracy' if max_acc_unseen==acc_unseen else '')
                log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                    epoch=epoch, accuracy=acc_unseen,
                    train_cls_count=self.num_train_pids,
                    test_cls_count=self.num_test_unseen_pids,
                    method='metric',
                    note=note)
                print('Margin: {}, Epoch: {}, Acc: {:.3}% (on unseen pictures)[Hard Sample + Sample Re-weighting]'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            max_acc = max(acc, max_acc)
            note = 'update:%.2f, on seen%s' % (self.update_conv_layers, ' - New Seen Accuracy' if max_acc==acc else '')
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='metric',
                epoch_time=(e_time-s_time),
                note=note)

            if epoch == self.start_epoch:
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
                prefix=self.prefix.split('/')[0])

            # if the accuracy keep dropping, stop training
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break
            # if overlap_rate_dict_ls[0][1] < .1:
            #     print('Top exter class overlap rate reach a smaller value than threshold, stop in epoch %d\n' % epoch)
            #     break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\n[Hard Sample + Sample Re-weighting] Training finished with: %d epoch, %.2f%% accuracy.' % (epoch, max_acc*100)
            f.write(c.encode())
        pickle_write('./results/temp/%s_v5_overlap_rate_dict.pkl' % self.prefix, overlap_dict)
        return self.save_dir, max_acc, self.num_test_pids

    def classifier_training(self, balance_testset):
        self._train_prepare()

        # prepare_dataloader(train_root, test_root)
        kwargs = {'num_workers': 1, 'pin_memory': True}
        testloader = load_testing(self.test_root, 128, kwargs)

        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            if self.step_size > 0:
                optimizer = _adjust_learning_rate(optimizer, epoch + 1)
            next_margin = self.margin

            train_with_classifier(self.model, epoch=epoch)

            # true testing on seen classes
            acc = test_with_classifier(self.model, testloader)
            print('Margin: {}, Epoch: {}, Acc: {:.3}%(on seen pictures)'.format(self.margin, epoch, acc * 100))
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='classifier',
                note='update:%.2f, on seen'%self.update_conv_layers)

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
                save_model_name = 'inception_v3_classifier_conv%.2f.tar' % (self.update_conv_layers)
            else:
                save_model_name = 'resnet_classifier_conv%.2f.tar' % (self.update_conv_layers)
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
                method='classifier',
                prefix=self.prefix)

            # if the accuracy keep dropping, stop training
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d times, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\nTraining finished with: %d epoch, %.2f accuracy' % (epoch, max_acc)
            f.write(c.encode())
        return self.save_dir, max_acc, self.num_test_pids

    def sample_finetune_training(self, balance_testset):
        self._train_prepare()

        max_acc, last_acc, drop_count, fail_max_count = .0, .0, 0, 0
        for epoch in range(self.start_epoch, self.start_epoch+self.train_epochs):
            if self.step_size > 0:
                self.optimizer = _adjust_learning_rate(self.optimizer, epoch)
            next_margin = self.margin

            if self.load_model_path is None:
                print('It seems U R fine-tuning a model, but no model U provide first, please specify one. Ends now.')
                exit(404)
            fine_tune_root = '/home/ubuntu/Program/Dish_recognition/dataset/fine_tune/'
            prepare_fine_tune_dataset(fine_tune_root=fine_tune_root,
                                    sample_file_dir=self.sample_file_dir)
            train_for_fine_tune(self.model,
                                self.optimizer,
                                self.tri_criterion,
                                epoch,
                                fine_tune_root,
                                None)

            # true testing on seen classes
            acc = evaluate_with_models([self.model],
                                _range=self.num_train_pids+1,
                                test_dir=self.test_root,
                                sample_file_dir=self.sample_file_dir,
                                seen='seen',
                                temp_prefix=self.prefix,
                                balance_testset=balance_testset)
            print('Margin: {}, Epoch: {}, Acc: {:.3}%(on seen pictures)'.format(self.margin, epoch, acc * 100))
            log(log_path=os.path.join(self.save_dir, 'readme.txt'),
                epoch=epoch,
                accuracy=acc,
                train_cls_count=self.num_train_pids,
                test_cls_count=self.num_test_pids,
                method='finetune',
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
                    method='finetune',
                    note='update:%.2f, on unseen'%self.update_conv_layers)
                print('Margin: {}, Epoch: {}, Acc: {:.3}%(on unseen pictures)'.format(self.margin, epoch, acc_unseen * 100))
            else:
                acc_unseen = -1

            # acc = (acc * 0 + acc_unseen * 1 if acc_unseen != -1 else acc)

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
                save_model_name = 'inception_v3_finetune_conv%.2f.tar' % (self.update_conv_layers)
            else:
                save_model_name = 'resnet_finetune_conv%.2f.tar' % (self.update_conv_layers)
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
                method='finetune',
                prefix=self.prefix)

            # if the accuracy keep dropping, stop training
            if (drop_count == 12 or fail_max_count == 24) and self.enable_stop_machanism:
                print('Accuracy dropping for %d times or smaller the max for %d, stop in epoch %d\n' % (
                drop_count, fail_max_count, epoch))
                break

            self.margin = next_margin
        with open(os.path.join(self.save_dir, 'readme.txt'), 'ab+') as f:
            c = '\r\nTraining finished with: %d epoch, %.2f accuracy' % (epoch, max_acc)
            f.write(c.encode())
        return self.save_dir, max_acc, self.num_test_pids
