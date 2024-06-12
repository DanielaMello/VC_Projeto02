import numpy as np

# Definindo o alfabeto conforme as informações fornecidas
alpha = list('!"#&’()*+,-./0123456789:;?abcdefghijklmnopqrstuvwxyz')

# Salvando o alfabeto em um arquivo .npy
np.save('/content/drive/MyDrive/Projeto/IAM/alpha.npy', alpha)

# Verificando o conteúdo do arquivo alpha.npy
loaded_alpha = np.load('C:/Users/danie/Downloads/Projeto/IAM/alpha.npy', allow_pickle=True)
print("Conteúdo do alpha.npy:")
print(loaded_alpha)

# Contando a quantidade de caracteres em alpha
K = len(alpha)
print("Quantidade de caracteres em alpha:", K)