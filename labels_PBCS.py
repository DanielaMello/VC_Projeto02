import numpy as np
import representations as rep


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


# Carrega os dados do arquivo test.txt
words_txt_path = 'C:/Users/danie/Downloads/Projeto/train.txt'
img_ids, label_data = load_data_from_txt(words_txt_path)

# Carrega o arquivo alpha.npy
alpha = np.load('C:/Users/danie/Downloads/Projeto/alpha.npy')

# Gera os rótulos PBSC ...apenas uma vez
Ydata = np.zeros((label_data.shape[0], 48, 53), dtype='f')
for i, x in enumerate(label_data):
    tmp = np.reshape(rep.rep_PBSC(x, alpha, 4), (48, 53))
    Ydata[i, :, :] = tmp
    print(x, i)

# Salva os rótulos PBSC
np.save('C:/Users/danie/Downloads/Projeto/labels_PBSC.npy', Ydata)
