import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import skimage.transform as sk
import csv

# Carregar o dataset a partir de um arquivo .txt com tratamento de erros
def load_data_from_txt(file_path):
    try:
        data = pd.read_csv(file_path, delimiter=' ', header=None, error_bad_lines=False)
    except pd.errors.ParserError as e:
        print(f"Erro ao processar o arquivo: {e}")
        return None, None
    img_data = data.iloc[:, 1:-1].values
    lab_data = data.iloc[:, -1].values
    return img_data, lab_data

# Carregar os dados do arquivo .txt
img_data, lab_data = load_data_from_txt('C:/Users/danie/Downloads/Projeto/train.txt')

if img_data is None or lab_data is None:
    print("Falha ao carregar os dados. Verifique o arquivo de entrada.")
else:
    print(f"Dados carregados com sucesso. {img_data.shape[0]} amostras encontradas.")

img_data, lab_data = load_data_from_txt('C:/Users/danie/Downloads/Projeto/train.txt')
alpha = np.load('C:/Users/danie/Downloads/Projeto/alpha.npy')
labels = np.load('C:/Users/danie/Downloads/Projeto/labels_PBSC.npy')
print(labels.shape)

img_data_train = img_data[:75000]
labels_train = labels[:75000, :, :]
img_data_val = img_data[75000:]
labels_val = labels[75000:, :, :]

del img_data, labels

class modele(nn.Module):
    def __init__(self):
        super(modele, self).__init__()

        self.conv01 = nn.Conv2d(1, 64, 5)
        self.padd01 = nn.ConstantPad2d((0, 4, 0, 4), 0)
        self.relu01 = nn.ReLU()
        self.bach01 = nn.BatchNorm2d(64)

        self.conv02 = nn.Conv2d(64, 64, 5)
        self.padd02 = nn.ConstantPad2d((0, 4, 0, 4), 0)
        self.relu02 = nn.ReLU()
        self.bach02 = nn.BatchNorm2d(64)

        self.max0 = nn.MaxPool2d(2)

        ################################

        self.conv11 = nn.Conv2d(64, 96, 5)
        self.padd11 = nn.ConstantPad2d((0, 4, 0, 4), 0)
        self.relu11 = nn.ReLU()
        self.bach11 = nn.BatchNorm2d(96)

        self.conv12 = nn.Conv2d(96, 96, 5)
        self.padd12 = nn.ConstantPad2d((0, 4, 0, 4), 0)
        self.relu12 = nn.ReLU()
        self.bach12 = nn.BatchNorm2d(96)

        self.max1 = nn.MaxPool2d(2)

        ################################

        self.conv21 = nn.Conv2d(96, 128, 5)
        self.padd21 = nn.ConstantPad2d((0, 4, 0, 4), 0)
        self.relu21 = nn.ReLU()
        self.bach21 = nn.BatchNorm2d(128)

        self.conv22 = nn.Conv2d(128, 128, 5)
        self.padd22 = nn.ConstantPad2d((0, 4, 0, 4), 0)
        self.relu22 = nn.ReLU()
        self.bach22 = nn.BatchNorm2d(128)

        self.max2 = nn.MaxPool2d(2)

        self.conv31 = nn.Conv2d(128, 128, 5)
        self.relu31 = nn.ReLU()
        self.bach31 = nn.BatchNorm2d(128)

        self.lin1 = nn.Linear(2176, 2000)
        self.relu1 = nn.ReLU()
        self.bach1 = nn.BatchNorm1d(2000)

        self.lin2 = nn.Linear(2000, 1908)

        self.soft = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.conv01(x)
        x = self.padd01(x)
        x = self.relu01(x)
        x = self.bach01(x)

        x = self.conv02(x)
        x = self.padd02(x)
        x = self.relu02(x)
        x = self.bach02(x)

        x = self.max0(x)

        x = self.conv11(x)
        x = self.padd11(x)
        x = self.relu11(x)
        x = self.bach11(x)

        x = self.conv12(x)
        x = self.padd12(x)
        x = self.relu12(x)
        x = self.bach12(x)

        x = self.max1(x)

        x = self.conv21(x)
        x = self.padd21(x)
        x = self.relu21(x)
        x = self.bach21(x)

        x = self.conv22(x)
        x = self.padd22(x)
        x = self.relu22(x)
        x = self.bach22(x)

        x = self.max2(x)

        x = self.conv31(x)
        x = self.relu31(x)
        x = self.bach31(x)

        x = x.view(x.shape[0], -1)

        x = self.lin1(x)
        x = self.bach1(x)
        x = self.relu1(x)

        x = self.lin2(x)
        x = x.view(x.shape[0], 36, 53)

        return self.soft(x)

model = modele().cuda()
model.load_state_dict(torch.load('C:/Users/danie/Downloads/Projeto/model_cnn_dict_PBSC.pth'))

threshold_valid = 100
lr_par = 1e-3

losses = [['loss_train', 'loss_validation']]

for ep in range(1000):
    optimizer = optim.Adam(model.parameters(), lr=lr_par, betas=(0.9, 0.999), weight_decay=1e-5)

    ## Treinamento
    model.train()
    s_train = 0
    nb_train = 0
    Xdata = np.zeros((img_data_train.shape[0], 1, 40, 170), dtype='f')
    i = 0
    for x in img_data_train:
        x = sk.rotate(x, random.randint(-15, 15), preserve_range=True)
        x = np.clip(x, 0, 255)
        mn = 13.330751
        std = 39.222755
        x = (x - mn) / std
        Xdata[i, 0, :, :] = x
        i += 1

    Ydata = labels_train.copy()
    shuf = np.arange(Ydata.shape[0])
    np.random.shuffle(shuf)
    Y = Ydata[shuf]
    Xdata = Xdata[shuf]

    for i in range(int(Xdata.shape[0] / 32)):
        input = Variable(torch.from_numpy(Xdata[i * 32:(i + 1) * 32, :, :, :]).cuda())
        target = Variable(torch.from_numpy(Y[i * 32:(i + 1) * 32, :, :]).cuda())
        output = model(input)
        loss = F.binary_cross_entropy(output, target)
        s_train += loss.item()
        nb_train += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## Validação
    model.eval()
    s_val = 0
    nb_val = 0
    Xdata = np.zeros((img_data_val.shape[0], 1, 40, 170), dtype='f')
    i = 0
    for x in img_data_val:
        mn = 13.330751
        std = 39.222755
        x = (x - mn) / std
        Xdata[i, 0, :, :] = x
        i += 1
    Y = labels_val.copy()

    for i in range(int(Xdata.shape[0] / 32)):
        input = Variable(torch.from_numpy(Xdata[i * 32:(i + 1) * 32, :, :, :]).cuda())
        target = Variable(torch.from_numpy(Y[i * 32:(i + 1) * 32, :, :]).cuda())
        output = model(input)
        loss = F.binary_cross_entropy(output, target)
        s_val += loss.item()
        nb_val += 1

    if s_val / nb_val < threshold_valid:
        threshold_valid = s_val / nb_val
        losses.append([s_train / nb_train, s_val / nb_val])
        print('error loss:', s_train / nb_train, 'val loss:', threshold_valid, lr_par)
        torch.save(model.state_dict(), 'C:/Users/danie/Downloads/Projeto/model_cnn_dict_PBSC.pth')
        torch.save(model, 'C:/Users/danie/Downloads/Projeto/model_cnn_PBSC.pth')
    else:
        lr_par = lr_par / 1.05
        losses.append([s_train / nb_train, s_val / nb_val])

    with open("C:/Users/danie/Downloads/Projeto/losses_cnn_PBSC.csv", 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(losses)
