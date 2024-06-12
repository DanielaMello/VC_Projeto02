import random

def split_train_val(words, train_size):
    # Verificando se há palavras suficientes para o grupo de treinamento
    if len(words) < train_size:
        raise ValueError("Não há palavras suficientes para o grupo de treinamento.")

    # Selecionando palavras aleatórias para o grupo de treinamento
    train_indices = random.sample(range(len(words)), train_size)

    # Obtendo os índices das palavras que não estão no grupo de treinamento (grupo de teste)
    test_indices = list(set(range(len(words))) - set(train_indices))

    return train_indices, test_indices

def load_data_from_txt(file_path):
    img_ids = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            img_ids.append(parts[0])  # ID da palavra
            labels.append(parts[-1])  # Transcrição da palavra
    return img_ids, labels

def save_to_txt(file_path, indices, img_ids, labels):
    with open(file_path, 'w') as f:
        for idx in indices:
            f.write(f"{img_ids[idx]} {labels[idx]}\n")

words_txt_path = 'C:/Users/danie/Downloads/Projeto/words_filtered.txt'
img_ids, label_data = load_data_from_txt(words_txt_path)

total_words = len(label_data)
train_size = 82703  # Número de palavras para o grupo de treinamento

train_indices, test_indices = split_train_val(label_data, train_size)

print("Número de palavras para treinamento:", len(train_indices))
print("Número de palavras para teste:", len(test_indices))

save_to_txt('C:/Users/danie/Downloads/Projeto/train.txt', train_indices, img_ids, label_data)
save_to_txt('C:/Users/danie/Downloads/Projeto/test.txt', test_indices, img_ids, label_data)