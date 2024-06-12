import torch

model.load_state_dict(torch.load(model_path))
model.eval()
print("Modelo revertido para o estado salvo.")