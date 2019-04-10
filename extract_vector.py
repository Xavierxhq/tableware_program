import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import os
import time
from PIL import Image
from trainer.train_util import load_model
import matplotlib.pyplot as plt
from utils import data_augmentation


loader = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    image_dir = "/home/ubuntu/Program/Tableware/DataArgumentation/dataset/n_base_sample_5"
    out_dir = ""
    out_label = False
    out_strs = []
    imgfiles = os.listdir(image_dir)
    model, _ = load_model('model.tar')
    for imgfile in imgfiles:
        img = loader(Image.open(self.train_path + "/" + imgfile).convert('RGB')).unsqueeze(0).to(device, torch.float)
        out_str = str(torch.squeeze(model(Variable(img))).tolist())
        if out_label:
            out_str = out_str + " " + str(imgfile.split(".")[0].split("_")[-1])
        out_strs.append(out_str)
        print(out_str)
    