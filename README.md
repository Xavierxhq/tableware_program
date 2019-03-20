# Dish Recognition Porject
  
### Things to Carry in Mind
  
- you DONT'T TOUCH the folder in /home/ubuntu/Program/Dish_recognition/dataset/correct_pictures/, until you fully know and understand what are in this folder and what can be done to them.

- three folders that matter: train, test, test_unseen(all in /home/ubuntu/Program/Dish_recognition/dataset/). they are: training set, test set 1(classes being seen while training), test set 2(classes unseen). you need to prepare them before you start training or test(run trainer.py/tester.py)
  
### Procceeding of Running Projecct
  
1. run prepare_dataset.py in ./datasets(but carefully), in the py file you can see functions for preparing datassets. but if you are going to arrange new datasets for training, talk to others first to ensure the old datasets are taken care of

2. modify the trainer.py according to you exp setting, and then run it

3. if you need to run test seperately, modify the tester.py according to you exp setting, and then run it

4. data_analyzer.py provides some program to do data analyzing, no need to care for this if you dont get to do any analysis under certain conditons
