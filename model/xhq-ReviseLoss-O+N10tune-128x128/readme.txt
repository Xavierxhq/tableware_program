This folder is generated on 2019-01-27 15:41:32, marked by xhq-ReviseLoss-O+N10tune-128x128,
Here are some settings:
train method: metric
neural network: resnet
batch_size: 128
training set: Tableware/DataArgumentation/dataset/o-n10_train/(135 classes, 28187 images)
test set: Tableware/DataArgumentation/dataset/n_test/(42 classes, 1999 images)
input size: 128x128
and i load a model: ./model/xhq_ft/xhq_ft_metric.pth.tar

And, this folder is created for saving trained model, usually,
the best model should be the one named with "xhq-ReviseLoss-O+N10tune-128x128" in it.


2019-01-27 15:49, Epoch:   1, Acc: 82.99, (135C-42C) [metric][update:0.05, on seen - New Top Accuracy]
2019-01-27 15:57, Epoch:   2, Acc: 82.74, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 16:05, Epoch:   3, Acc: 82.49, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 16:13, Epoch:   4, Acc: 82.49, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 16:21, Epoch:   5, Acc: 82.29, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 16:28, Epoch:   6, Acc: 82.19, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 16:36, Epoch:   7, Acc: 82.29, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 16:44, Epoch:   8, Acc: 83.09, (135C-42C) [metric][update:0.05, on seen - New Top Accuracy]
2019-01-27 16:52, Epoch:   9, Acc: 82.69, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 16:59, Epoch:  10, Acc: 83.49, (135C-42C) [metric][update:0.05, on seen - New Top Accuracy]
2019-01-27 17:07, Epoch:  11, Acc: 82.79, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 17:14, Epoch:  12, Acc: 83.34, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 17:22, Epoch:  13, Acc: 82.39, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 17:30, Epoch:  14, Acc: 82.94, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 17:37, Epoch:  15, Acc: 82.69, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 17:45, Epoch:  16, Acc: 82.69, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 17:53, Epoch:  17, Acc: 82.84, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:00, Epoch:  18, Acc: 81.54, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:08, Epoch:  19, Acc: 83.29, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:15, Epoch:  20, Acc: 82.54, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:23, Epoch:  21, Acc: 82.54, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:30, Epoch:  22, Acc: 82.59, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:37, Epoch:  23, Acc: 82.64, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:44, Epoch:  24, Acc: 82.39, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:51, Epoch:  25, Acc: 83.14, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 18:58, Epoch:  26, Acc: 83.34, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 19:05, Epoch:  27, Acc: 82.84, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 19:12, Epoch:  28, Acc: 82.29, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 19:19, Epoch:  29, Acc: 83.14, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 19:26, Epoch:  30, Acc: 82.59, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 19:33, Epoch:  31, Acc: 83.04, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 19:40, Epoch:  32, Acc: 82.79, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 19:47, Epoch:  33, Acc: 82.59, (135C-42C) [metric][update:0.05, on seen]
2019-01-27 19:55, Epoch:  34, Acc: 82.94, (135C-42C) [metric][update:0.05, on seen]

[Augmented Triplet Loss]Training finished with: 34 epoch, 0.83% accuracy