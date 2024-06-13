import os
import numpy as np
import torch
import random
from PIL import Image
import torch.nn as nn
import skimage.transform as sk
from sklearn.metrics import average_precision_score

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

# Caminhos dos arquivos
train_txt_path = 'C:/Users/danie/Downloads/Projeto/train.txt'
alpha_path = 'C:/Users/danie/Downloads/Projeto/alpha.npy'
labels_path = 'C:/Users/danie/Downloads/Projeto/labels_PBSC.npy'
img_dir = 'C:/Users/danie/Downloads/Projeto/words'

# Verificar se o arquivo txt existe
if not os.path.exists(train_txt_path):
    raise FileNotFoundError(f"O arquivo {train_txt_path} não foi encontrado.")

# Carregar os dados do arquivo test.txt
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

        # Ajuste do tamanho da camada linear
        self.lin1 = nn.Linear(2176, 2000)
        self.relu1 = nn.ReLU()
        self.bach1 = nn.BatchNorm1d(2000)

        # Ajuste do tamanho da saída final
        self.lin2 = nn.Linear(2000, 48 * 53)

        self.soft = nn.Softmax(2)

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
        x = x.view(x.shape[0], 48, 53)

        return self.soft(x)

# Verifique se a GPU está disponível
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Usando GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Usando CPU")

# Carregar o modelo treinado
model_path = 'C:/Users/danie/Downloads/Projeto/model_cnn_dict_PBSC.pth'
model = modele().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Dividir dados em treinamento e validação
img_data_train = img_data[:75000]
labels_train = labels[:75000]
img_data_val = img_data[75000:]
labels_val = labels[75000:]
del img_data, labels

# Função para preparar os dados
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

# Preparar dados de treinamento e validação
img_data_train, labels_train = prepare_data(img_data_train, labels_train)
img_data_val, labels_val = prepare_data(img_data_val, labels_val)

# Função para extrair características

def extract_features(model, img_data, device):
    model.eval()
    features = []
    with torch.no_grad():
        for img in img_data:
            input = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(input).cpu().numpy()
            features.append(output.flatten())
    return np.array(features)


# Extrair características de consulta e referência
query_features = extract_features(model, img_data_val, device)
ref_features = extract_features(model, img_data_train, device)


def calculate_map(query_features, query_labels, ref_features, ref_labels):
    map_score = 0
    for query_feature, query_label in zip(query_features, query_labels):
        distances = np.linalg.norm(ref_features - query_feature, axis=1)
        sorted_indices = np.argsort(distances)
        sorted_labels = ref_labels[sorted_indices]
        relevant = (sorted_labels == query_label).astype(int)
        if np.sum(relevant) > 0:
            scores = 1 - (distances[sorted_indices] / np.max(distances))  # Convertendo distâncias em pontuações
            ap = average_precision_score(relevant, scores)
            map_score += ap
    map_score /= len(query_features)
    return map_score

# Calcular mAP
map_score = calculate_map(query_features, labels_val, ref_features, labels_train)
print(f"mAP: {map_score}")
