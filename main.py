import os, time
import torch
# from data_analyzer import DataAnalyzer
from trainer import Trainer
from utils.email_util import Emailer


if __name__ == "__main__":

    """
        Set Hyper-Parameters for training
    """
    while True:
        print('1. Enter the prefix(for personal identity):')
        prefix = input() # for personal identity, the models you train will be store at: './model/prefix'
        date_str = time.strftime("%Y%m%d", time.localtime())
        prefix = date_str + '-' + prefix
        if os.path.exists( os.path.join('model', prefix) ):
            print('Folder existed, re-enter.')
        elif prefix is None or prefix == '':
            print('Prefix must be specified!')
        else:
            break
    # model_type(should be in ["resnet", "inception3"])
    model_type = 'resnet'
    # print('3. Enter the batch_size(try: 32, 64, 128 or others u like):')
    # batch_size = int(input())
    # batch_size = 128
    # print('4. Enter the input size(try: 128, 224, 300, 400 or others u like):')
    # input_size = int(input())
    input_size = 300

    """
        set dataset paths
    """
    train_root = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_train/'
    test_root = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_test'
    sample_file_dir = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_base_sample_5'

    # if need unseen test
    # test_unseen_root = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_test/'
    # sample_file_dir_unseen = '/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_base_sample_5/'

    # if not need unseen test
    test_unseen_root = None
    sample_file_dir_unseen = None

    """
        specify a model loaded while starting training, 
        set None to train from Pretrained(provided by Pytorch)
    """
    p = None
    p = '/home/ubuntu/Program/Dish_recognition/program/model/20190320-xhq-ha+re-au2-O/run_1/20190320-xhq-Hard+Reweight-Augmentation_metric.tar'
    # p = './model/pretrained/inception_v3_google-1a9a5a14.pth'

    print('\nTraining will start. Make sure you have set: train_root, test_root, sample_file_dir!\n')
        
    avg_acc, start_run, avg_epoch = .0, 1, 0
    end_run = 2 if p is None else 4
    batch_size = 128 if p is None else 64
    for i in range(start_run, end_run):
        print()
        print('++++++++++++++++++++')
        print('+ Start new run %d +'%i)
        print('++++++++++++++++++++')
        print()
        temp_prefix = prefix + '/run_' + str(i)
        trainer = None
        trainer = Trainer(model_type=model_type, load_model_path=p, method='metric')
        trainer.set_super_training_parameters(train_root=train_root,
                                            test_root=test_root,
                                            sample_file_dir=sample_file_dir,
                                            test_unseen_root=test_unseen_root,
                                            sample_file_dir_unseen=sample_file_dir_unseen,
                                            start_epoch=1,
                                            enable_stop_machanism=True,
                                            prefix=temp_prefix,
                                            batch_size=batch_size,
                                            input_w=input_size,
                                            input_h=input_size)
        maxacc, epoch = trainer.train(balance_testset=False)

        avg_acc += maxacc
        avg_epoch += epoch -12
        # release GPU memory
        torch.cuda.empty_cache()
    avg_acc /= (end_run - start_run)
    print('The average accuracy is %.2f, average epoch: %d' % (avg_acc * 100, avg_epoch / (end_run - start_run)))

    emailer = Emailer('from@runoob.com', ['xavier.xhq@qq.com'])
    emailer.send(prefix + 'Experiment Finished. And get average accuracy: ' + str(avg_acc*100))
