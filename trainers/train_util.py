import os, time
import torch
from models import get_baseline_model
from models.inception3 import inception_v3


def load_model(model_path=None, num_of_classes=0):
    model, optim_policy = get_baseline_model(model_path=None, num_of_classes=num_of_classes)
    if model_path is not None:
        model_dict = model.state_dict()
        pretrained_params = torch.load(model_path)
        new_dict = {k: v for k, v in pretrained_params['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(new_dict)
        print('model', model_path.split('/')[-1], 'loaded.')
    model = model.cuda()
    return model, optim_policy


def load_inception3(model_path=None):
    model = inception_v3(pretrained=True)
    optim_policy = [
        {'params': model.parameters()}
    ]
    if model_path is not None:
        # model_dict = model.state_dict()
        pretrained_params = torch.load(model_path)

        model.load_state_dict(pretrained_params, strict=False)
        # new_dict = {k: v for k, v in pretrained_params['state_dict'].items() if k in model_dict.keys()}
        # model_dict.update(new_dict)
        # model.load_state_dict(new_dict)
        print('model', model_path.split('/')[-1], 'loaded.')
    model = model.cuda()
    return model, optim_policy


def freeze_model(model, update_prefix='full_conn'):
    for name, p in model.named_parameters():
        if (type(update_prefix) == str and update_prefix not in name) or name not in update_prefix:
            p.requires_grad = False
            pass
    print('model freezed, update:', update_prefix)


def log(log_path, epoch, accuracy, train_cls_count, test_cls_count, method, note, epoch_time=None, content=None):
    if content is not None:
        with open(log_path, 'ab+') as fp:
            fp.write(content.encode())
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(log_path, 'ab+') as fp:
        c = '%s, Epoch: %3d, Acc: %.2f(%dC-%dC) %s[%s][%s]\r\n' % (time_str[:16],
                                                                    epoch,
                                                                    accuracy * 100,
                                                                    train_cls_count,
                                                                    test_cls_count,
                                                                    ('[time:%.2f]'%epoch_time) if epoch_time is not None else '',
                                                                    method,
                                                                    note)
        if content is not None:
            c = content + c + '\r\n'
        fp.write(c.encode())


def statistic_manager(dataset):
	if dataset is None:
		return 0, 0
	items = os.listdir(dataset)
	class_ls, count = [], 0
	for ind in items:
		if '.png' not in ind and '.jpg' not in ind:
			class_ls.append(ind)
			count += len( os.listdir( os.path.join(dataset, ind) ) )
			continue
		cls_idx = ind.split('_')[-1][:-4]
		if cls_idx not in class_ls:
			class_ls.append(cls_idx)
		count += 1
	return len(class_ls), count
