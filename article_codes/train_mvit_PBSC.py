import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import skimage.transform as sk
import csv
from PIL import Image
from article_codes.mvit import MobileViT

# Função para carregar dados de um arquivo words.txt
def load_data_from_txt(file_path):
    img_ids = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            img_ids.append(parts[0])  # ID da palavra
            labels.append(parts[1])  # Transcrição da palavra
    return np.array(img_ids), np.array(labels)

# Função para carregar imagens e converter rótulos
def load_images_and_labels(img_ids, label_data, img_dir):
    img_data = []
    labels = []
    for img_id, label in zip(img_ids, label_data):
        # Construir o caminho da imagem
        parts = img_id.split('-')
        if len(parts) >= 2:  # Verificar se o formato do img_id é esperado
            img_path = os.path.join(img_dir, parts[0], f"{parts[0]}-{parts[1]}", f"{img_id}.png")
            if os.path.exists(img_path) and img_path.lower().endswith('.png'):
                try:
                    img = Image.open(img_path)
                    img = img.convert('L')  # Converter para escala de cinza
                    img = img.resize((170, 40))  # Redimensionar todas as imagens para o mesmo tamanho
                    img = np.array(img, dtype='float32')
                    img = img / 255.0  # Normalizar os valores dos pixels
                    img_data.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
            else:
                print(f"Imagem {img_path} não encontrada ou não é um arquivo PNG.")
        else:
            print(f"Formato do img_id {img_id} não é válido.")
    return img_data, labels

# Carrega os dados do arquivo train.txt
words_txt_path = 'C:/Users/danie/Downloads/Projeto/train.txt'
img_ids, label_data = load_data_from_txt(words_txt_path)

# Carrega o arquivo alpha.npy
alpha = np.load('C:/Users/danie/Downloads/Projeto/alpha.npy')

# Geração de rótulos PBSC (apenas uma vez)
#Ydata = np.zeros((label_data.shape[0], 36, 53), dtype='f')
# for i, x in enumerate(label_data):
#     tmp = np.reshape(rep.rep_PBSC(x, alpha, 3), (36, 53))
#     Ydata[i, :, :] = tmp
#     print(x, i)

# Salva os rótulos PBSC
# np.save('C:/Users/danie/Downloads/Projeto/labels_mvit_PBSC.npy', Ydata)

# Caminhos dos arquivos
train_txt_path = 'C:/Users/danie/Downloads/Projeto/train.txt'
alpha_path = 'C:/Users/danie/Downloads/Projeto/alpha.npy'
labels_path = 'C:/Users/danie/Downloads/Projeto/labels_mvit_PBSC.npy'
img_dir = 'C:/Users/danie/Downloads/Projeto/words'

# Verificar se o arquivo txt existe
if not os.path.exists(train_txt_path):
    raise FileNotFoundError(f"O arquivo {train_txt_path} não foi encontrado.")

# Carregar os dados do arquivo train.txt
img_ids, lab_data = load_data_from_txt(train_txt_path)

# Carregar os arquivos .npy
alpha = np.load(alpha_path)
labels = np.load(labels_path)

# Carregar e preprocessar as imagens
img_data, label_data = load_images_and_labels(img_ids, labels, img_dir)

# Verifique se img_data não está vazio e tenha imagens com o mesmo formato
if len(img_data) == 0:
    raise ValueError("Nenhuma imagem válida foi encontrada.")
else:
    # Verifique se todas as imagens têm o mesmo formato
    img_shape = img_data[0].shape
    for img in img_data:
        if img.shape != img_shape:
            raise ValueError("As imagens carregadas não têm todas o mesmo formato.")

# Converter img_data e label_data para arrays numpy
img_data = np.array(img_data)
label_data = np.array(label_data)

# Verificar a forma dos rótulos
print(labels.shape)

model = MobileViT(image_size=(40, 170), dims=[96, 96, 96],
                  channels=[32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 114, 114, 128, 24], num_classes=1908).cuda()
print(model)

# Verifique se a GPU está disponível
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Usando GPU:', torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Usando CPU")

model = model.to(device)
model_path = 'C:/Users/danie/Downloads/Projeto/model_mvit_dict_PBSC.pth'

# # Verificar se o modelo já existe
if os.path.exists(model_path):
     print(f"Carregando modelo existente de {model_path}")
     model.load_state_dict(torch.load(model_path))
else:
# Salvar o novo modelo inicial (isso sobrescreverá o modelo existente)
    torch.save(model.state_dict(), model_path)
    print(f"Arquivo de modelo não encontrado em {model_path}. Treinando um novo modelo.")

img_data_train = img_data[:75000]
labels_train = labels[:75000, :, :]
img_data_val = img_data[75000:]
labels_val = labels[75000:, :, :]

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

# Preparação dos dados de treinamento e validação
img_data_train, labels_train = prepare_data(img_data_train, labels_train)
img_data_val, labels_val = prepare_data(img_data_val, labels_val)

# Processo de treinamento e validação
for ep in range(1000):
    optimizer = optim.Adam(model.parameters(), lr=lr_par, betas=(0.9, 0.999), weight_decay=0)

    ## Train
    model.train()
    train_losses = []

    for i in range(int(img_data_train.shape[0] / 32)):
        input = torch.from_numpy(img_data_train[i * 32:(i + 1) * 32, :, :, :]).to(device)
        target = torch.from_numpy(labels_train[i * 32:(i + 1) * 32, :, :]).to(device)

        output = model(input)
        loss = F.binary_cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    ## Validation
    model.eval()
    val_losses = []

    for i in range(int(img_data_val.shape[0] / 32)):
        input = torch.from_numpy(img_data_val[i * 32:(i + 1) * 32, :, :, :]).to(device)
        target = torch.from_numpy(labels_val[i * 32:(i + 1) * 32, :, :]).to(device)

        output = model(input)
        loss = F.binary_cross_entropy(output, target)

        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    if val_loss < threshold_valid:
        threshold_valid = val_loss
        losses.append([train_loss, val_loss])
        print(f'Epoch: {ep + 1}/1000')
        print(f'Training loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        print(f'New best validation loss: {val_loss:.4f}. Model saved.')
        torch.save(model.state_dict(), 'C:/Users/danie/Downloads/Projeto/model_mvit_dict_PBSC.pth')
        torch.save(model, 'C:/Users/danie/Downloads/Projeto/model_mvit_PBSC.pth')
    else:
        lr_par /= 1.05
        losses.append([train_loss, val_loss])
        print(f'Epoch: {ep + 1}/1000')
        print(f'Training loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        print(f'Validation loss did not improve. Reducing learning rate to {lr_par:.6f}.')

    with open("C:/Users/danie/Downloads/Projeto/losses_mvit_PBSC.csv", 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(losses)
