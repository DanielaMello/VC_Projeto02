import os
import numpy as np
from PIL import Image
import cv2
import pytesseract
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Definindo o caminho para o executável do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Função para carregar dados de um arquivo .txt
def load_data_from_txt(file_path):
    img_ids = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            # ID da palavra
            img_ids.append(parts[0])
            # Transcrição da palavra
            labels.append(parts[1])
    return np.array(img_ids), np.array(labels)


# Função para carregar imagens e converter rótulos
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
                    # Convertendo para escala de cinza
                    img = img.convert('L')
                    # Redimensionando as imagens para o mesmo tamanho
                    img = img.resize((170, 40))

                    # Aplicando filtro Gaussiano para suavização
                    img = cv2.GaussianBlur(np.array(img), (5, 5), 0)

                    # Aplicando binarização adaptativa
                    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # Normalizando os valores dos pixels
                    img = img.astype(np.float32) / 255.0
                    img_data.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
            else:
                print(f"Imagem {img_path} não encontrada ou não é um arquivo PNG.")
        else:
            print(f"Formato do img_id {img_id} não é válido.")
    return img_data, labels


# Função para executar OCR na imagem
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image, lang='eng')

    return text.strip()


# Caminhos dos arquivos
train_txt_path = 'C:/Users/danie/Downloads/Projeto/train.txt'
img_dir = 'C:/Users/danie/Downloads/Projeto/words'
output_metrics_path = 'C:/Users/danie/Downloads/Projeto/OCR_PDI_results.txt'

# Carregando IDs e rótulos
img_ids, labels = load_data_from_txt(train_txt_path)

# Carregando imagens e rótulos correspondentes
img_data, label_data = load_images_and_labels(img_ids, labels, img_dir)

# Executando OCR nas imagens carregadas
ocr_results = []
for i, img_array in enumerate(img_data, 1):
    img_cv = (img_array * 255).astype(np.uint8)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

    # Extraindo texto da imagem
    extracted_text = extract_text_from_image(img_cv)
    ocr_results.append(extracted_text)

    # Exibindo progresso
    print(f"Processando imagem {i}/{len(img_data)}")

# Exibindo os resultados do OCR
for original_label, ocr_result in zip(label_data, ocr_results):
    print(f"Rótulo Original: {original_label} | OCR Resultado: {ocr_result}")

# Função para calcular métricas de avaliação
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    return accuracy, precision, recall, f1


# Calculando métricas de avaliação
accuracy, precision, recall, f1 = calculate_metrics(label_data, ocr_results)

print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Salvando as métricas em um arquivo de texto
with open(output_metrics_path, 'w') as f:
    f.write(f"Acurácia: {accuracy:.4f}\n")
    f.write(f"Precisão: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"Métricas salvas em {output_metrics_path}")
