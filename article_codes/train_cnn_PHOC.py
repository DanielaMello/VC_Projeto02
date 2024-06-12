import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import skimage.transform as sk
from matplotlib import pyplot as plt
import csv
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import representations as rep

def load_data_from_txt(file_path):
    img_ids = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            img_ids.append(parts[0])
            labels.append(parts[1])
    return np.array(img_ids), np.array(labels)

def load_images_and_labels(img_ids, label_data, img_dir):
    img_data = []
    labels = []
    for img_id, label in zip(img_ids, label_data):
        parts = img_id.split('-')
        if len(parts) >= 2:
            img_path = os.path.join(img_dir, parts[0], f"{parts[0]}-{parts[1]}", f"{img_id}.png")
            if os.path.exists(img_path) and img_path.lower().endswith('.png'):
                try:
                    img = Image.open(img_path)
                    img = img.convert('L')
                    img = img.resize((170, 40))
                    img = np.array(img, dtype='float32')
                    img = img / 255.0
                    img_data.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
            else:
                print(f"Imagem {img_path} não encontrada ou não é um arquivo PNG.")
        else:
            print(f"Formato do img_id {img_id} não é válido.")
    return img_data, labels

words_txt_path = 'C:/Users/danie/Downloads/Projeto/train.txt'
img_ids, label_data = load_data_from_txt(words_txt_path)
alpha = np.load('C:/Users/danie/Downloads/Projeto/alpha.npy')

# Ydata = np.zeros((label_data.shape[0], 520), dtype='f')
# for i, x in enumerate(label_data):
#     tmp = rep.rep_PHOC(x, alpha)
#     Ydata[i, :] = tmp
#     print(x, i)
# np.save('C:/Users/danie/Downloads/Projeto/labels_PHOC.npy', Ydata)

train_txt_path = 'C:/Users/danie/Downloads/Projeto/train.txt'
alpha_path = 'C:/Users/danie/Downloads/Projeto/alpha.npy'
labels_path = 'C:/Users/danie/Downloads/Projeto/labels_PHOC.npy'
img_dir = 'C:/Users/danie/Downloads/Projeto/words'

if not os.path.exists(train_txt_path):
    raise FileNotFoundError(f"O arquivo {train_txt_path} não foi encontrado.")

img_ids, lab_data = load_data_from_txt(train_txt_path)
alpha = np.load(alpha_path)
labels = np.load(labels_path)
img_data, label_data = load_images_and_labels(img_ids, labels, img_dir)

if len(img_data) == 0:
    raise ValueError("Nenhuma imagem válida foi encontrada.")
else:
    img_shape = img_data[0].shape
    for img in img_data:
        if img.shape != img_shape:
            raise ValueError("As imagens carregadas não têm todas o mesmo formato.")

img_data = np.array(img_data)
label_data = np.array(label_data)

print(labels.shape)

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

        self.conv11 = nn.Conv2d(64, 96, 5)
        self.padd11 = nn.ConstantPad2d((0, 4, 0, 4), 0)
        self.relu11 = nn.ReLU()
        self.bach11 = nn.BatchNorm2d(96)

        self.conv12 = nn.Conv2d(96, 96, 5)
        self.padd12 = nn.ConstantPad2d((0, 4, 0, 4), 0)
        self.relu12 = nn.ReLU()
        self.bach12 = nn.BatchNorm2d(96)

        self.max1 = nn.MaxPool2d(2)

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

        self.lin2 = nn.Linear(2000, 520)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv01(x)
        x = self.padd01(x)
        x = self.relu01(x)
        x = self.bach01(x)
        x = F.dropout2d(x, p=0.25)

        x = self.conv02(x)
        x = self.padd02(x)
        x = self.relu02(x)
        x = self.bach02(x)
        x = F.dropout2d(x, p=0.25)

        x = self.max0(x)

        x = self.conv11(x)
        x = self.padd11(x)
        x = self.relu11(x)
        x = self.bach11(x)
        x = F.dropout2d(x, p=0.25)

        x = self.conv12(x)
        x = self.padd12(x)
        x = self.relu12(x)
        x = self.bach12(x)
        x = F.dropout2d(x, p=0.25)

        x = self.max1(x)

        x = self.conv21(x)
        x = self.padd21(x)
        x = self.relu21(x)
        x = self.bach21(x)
        x = F.dropout2d(x, p=0.25)

        x = self.conv22(x)
        x = self.padd22(x)
        x = self.relu22(x)
        x = self.bach22(x)
        x = F.dropout2d(x, p=0.25)

        x = self.max2(x)

        x = self.conv31(x)
        x = self.relu31(x)
        x = self.bach31(x)
        x = F.dropout2d(x, p=0.25)

        x = x.view(x.shape[0], -1)

        x = self.lin1(x)
        x = self.bach1(x)
        x = self.relu1(x)

        x = self.lin2(x)
        x = self.sigmoid(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = modele().to(device)
model_path = 'C:/Users/danie/Downloads/Projeto/model_cnn_dict_PHOC.pth'

# # Verificar se o modelo já existe
if os.path.exists(model_path):
     print(f"Carregando modelo existente de {model_path}")
     model.load_state_dict(torch.load(model_path))
else:
# Salvar o novo modelo inicial (isso sobrescreverá o modelo existente)
    torch.save(model.state_dict(), model_path)
    print(f"Arquivo de modelo não encontrado em {model_path}. Treinando um novo modelo.")

img_data_train = img_data[:75000]
labels_train = labels[:75000, :]
img_data_val = img_data[75000:]
labels_val = labels[75000:, :]

del img_data, labels

threshold_valid = 100
lr_par = 1e-3

losses = [['loss_train', 'loss_validation']]

def prepare_data(img_data, labels, batch_size=32):
    Xdata = np.zeros((img_data.shape[0], 1, 40, 170), dtype='float32')
    for i, x in enumerate(img_data):
        x = sk.rotate(x, random.randint(-15, 15), preserve_range=True)
        x = np.clip(x, 0, 255)
        mn = 13.330751
        std = 39.222755
        x = (x - mn) / std
        Xdata[i, 0, :, :] = x

    Ydata = labels.copy()
    shuf = np.arange(Ydata.shape[0])
    np.random.shuffle(shuf)
    Xdata = Xdata[shuf]
    Ydata = Ydata[shuf]

    return Xdata, Ydata

img_data_train, labels_train = prepare_data(img_data_train, labels_train)
img_data_val, labels_val = prepare_data(img_data_val, labels_val)

labels_train = np.clip(labels_train, 0, 1)
labels_val = np.clip(labels_val, 0, 1)

for ep in range(1000):
    optimizer = optim.Adam(model.parameters(), lr=lr_par, betas=(0.9, 0.999), weight_decay=1e-7)

    model.train()
    train_losses = []

    for i in range(int(img_data_train.shape[0] / 48)):
        input = torch.from_numpy(img_data_train[i * 48:(i + 1) * 48, :, :, :]).to(device)
        target = torch.from_numpy(labels_train[i * 48:(i + 1) * 48, :]).to(device)
        output = model(input)
        loss = F.binary_cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    all_preds = []
    all_targets = []

    for i in range(int(img_data_val.shape[0] / 48)):
        input = torch.from_numpy(img_data_val[i * 48:(i + 1) * 48, :, :, :]).to(device)
        target = torch.from_numpy(labels_val[i * 48:(i + 1) * 48, :]).to(device)

        output = model(input)
        loss = F.binary_cross_entropy(output, target)

        val_losses.append(loss.item())

        preds = (output > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    # all_preds = np.concatenate(all_preds, axis=0).reshape(-1)
    # all_targets = np.concatenate(all_targets, axis=0).reshape(-1)
    #
    # precision = precision_score(all_targets, all_preds)
    # recall = recall_score(all_targets, all_preds)
    # f1 = f1_score(all_targets, all_preds)
    # accuracy = accuracy_score(all_targets, all_preds)

    if val_loss < threshold_valid:
        threshold_valid = val_loss
        losses.append([train_loss, val_loss])
        print(f'Epoch: {ep + 1}/1000')
        print(f'Training loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        # print(f'Accuracy: {accuracy:.4f}')
        print(f'New best validation loss: {val_loss:.4f}. Model saved.')
        torch.save(model.state_dict(), 'C:/Users/danie/Downloads/Projeto/model_cnn_dict_PHOC.pth')
        torch.save(model, 'C:/Users/danie/Downloads/Projeto/model_cnn_PHOC.pth')
    else:
        lr_par /= 1.05
        losses.append([train_loss, val_loss])
        print(f'Epoch: {ep + 1}/1000')
        print(f'Training loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        # print(f'Accuracy: {accuracy:.4f}')
        print(f'Validation loss did not improve. Reducing learning rate to {lr_par:.6f}.')

with open("C:/Users/danie/Downloads/Projeto/losses_cnn_PHOC.csv", 'w') as f:
       writer = csv.writer(f, delimiter=',')
       writer.writerows(losses)
