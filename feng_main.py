import os
import torch
# from data_analyzer import DataAnalyzer
from trainer import Trainer

if __name__ == "__main__":
	''' 20190226
    我说下吧 如果跑实验 Otrain的话用300x300的输入、128的batch size、训练的时候指定是trainer.metric_training_v5。如果用n5train，就改batch size为64，其他不变
    Otrain就Otest，N5/N10 train都用Ntest，注意只有N10train在另一个目录
    保持这些变量统一，然后我们要对比的有：数据随机变化、hard sample选取重叠率为标准、rewweighting设置不同超参数
    '''

	"""
        Set Hyper-Parameters for training
    """
	while True:
		print('1. Enter the prefix(for personal identity):')
		prefix = input()  # for personal identity, the models you train will be store at: './model/prefix'
		if os.path.exists(os.path.join('model', prefix)):
			print('Folder existed, re-enter.')
		elif prefix is None or prefix == '':
			print('Prefix must be specified!')
		else:
			break
	while True:
		print('2. Enter the model_type(should be in ["resnet", "inception3"]):')
		model_type = input()
		if model_type not in ('resnet', 'inception3'):
			print('\nmodel_type should be in ["resnet", "inception3"]!\n')
		else:
			break
	print('3. Enter the batch_size(try: 32, 64, 128 or others u like):')
	batch_size = int(input())
	print('4. Enter the input size(try: 128, 224, 300, 400 or others u like):')
	input_size = int(input())
	method = 'metric'

	"""
        set dataset paths
    """

	train_root = '/home/ubuntu/Program/fsl/54c/n10_train/'
	test_root = '/home/ubuntu/Program/fsl/54c/test/'
	sample_file_dir = '/home/ubuntu/Program/fsl/54c/sample/'

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
	load_model_path = '/home/ubuntu/Program/xhq/TablewareFinetuneProcedure/model.tar'
	# load_model_path = '/home/ubuntu/Program/Dish_recognition/Task/input size(tune twice)/224x224/xhq-Hard+Reweight-O-224x224/run_1/xhq-Hard+Reweight-O-224x224_metric.tar'
	# load_model_path = './model/pretrained/inception_v3_google-1a9a5a14.pth'

	print('\nTraining will start. Make sure you have set: train_root, test_root, sample_file_dir!\n')

	# # save the model to keep file, in here the final models are saved.
	# if 'inception3' == model_type:
	#     best_model_path = './model/keep/inception_v3_%s_%s_conv0.05_%.2f.tar'%(prefix, method, maxacc * 100)
	# else:
	#     best_model_path = './model/keep/resnet_%s_%s_conv0.05_%.2f.tar'%(prefix, method, maxacc * 100)
	# shutil.copy(os.path.join(save_dir, '%s_%s.pth.tar' % (prefix, method)), best_model_path)

	avg_acc, start_run, end_run = .0, 1, 4
	for i in range(start_run, end_run):
		print()
		print('++++++++++++++++++++')
		print('+ Start new run %d +' % i)
		print('++++++++++++++++++++')
		print()
		temp_prefix = prefix + '/run_' + str(i)
		trainer = None
		trainer = Trainer(model_type=model_type, load_model_path=load_model_path, method=method)
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
		if method == 'metric':
			save_dir, maxacc, test_pids = trainer.metric_training_v5(balance_testset=False)
		elif method == 'classifier':
			save_dir, maxacc, test_pids = trainer.classifier_training(balance_testset=False)
		elif method == 'finetune':
			save_dir, maxacc, test_pids = trainer.sample_finetune_training(balance_testset=False)
		else:
			save_dir = ''
			maxacc = 0.0
			print('Wrong method. Specify one in ["metric", "classifier", "finetune"]')
			exit(200)

		avg_acc += maxacc
		# analyzer = DataAnalyzer(sample_file_dir=sample_file_dir,
		#                         test_dir=test_root,
		#                         num_of_classes=test_pids,
		#                         prefix=temp_prefix)
		# analyzer.analysis_for_inter_exter_acc(model_path=os.path.join(save_dir, '%s_%s.tar' % (prefix, method)),
		#                                     WIDTH=trainer.w, HEIGHT=trainer.h)
		# release GPU memory
		torch.cuda.empty_cache()
	avg_acc /= (end_run - start_run)
	print('The average accuracy is %.2f' % (avg_acc * 100))
