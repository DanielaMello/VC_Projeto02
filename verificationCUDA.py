
import torch

if torch.cuda.is_available():
    print("CUDA está disponível!")
else:
    print("CUDA não está disponível.")

print(torch.cuda.is_available())  # Deve imprimir: True
print(torch.cuda.current_device())  # Deve imprimir o índice da GPU atual, geralmente 0
print(torch.cuda.get_device_name(0))  # Deve imprimir o nome da sua GPU, como "NVIDIA GeForce RTX 3050"
