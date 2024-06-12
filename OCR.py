import os
import numpy as np
from PIL import Image
import cv2
import pytesseract
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Definir o caminho para o executável do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'c:\users\danie\appdata\local\programs\python\python312\lib\site-packages'

# Função para carregar IDs e rótulos de um arquivo de texto
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


# Função para pré-processar a imagem para OCR
def preprocess_image_for_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary_image


# Função para executar OCR na imagem pré-processada
def extract_text_from_image(image):
    # Pré-processar a imagem
    processed_image = preprocess_image_for_ocr(image)

    # Executar OCR na imagem processada
    text = pytesseract.image_to_string(processed_image, lang='eng')

    return text.strip()


# Caminhos dos arquivos
train_txt_path = 'C:/Users/danie/Downloads/Projeto/train.txt'
img_dir = 'C:/Users/danie/Downloads/Projeto/words'

# Carregar IDs e rótulos
img_ids, labels = load_data_from_txt(train_txt_path)

# Carregar imagens e rótulos correspondentes
img_data, label_data = load_images_and_labels(img_ids, labels, img_dir)

# Executar OCR nas imagens carregadas
ocr_results = []
for img_array in img_data:
    # Convert numpy array to OpenCV image format
    img_cv = (img_array * 255).astype(np.uint8)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

    # Extrair texto da imagem
    extracted_text = extract_text_from_image(img_cv)
    ocr_results.append(extracted_text)

# Exibir os resultados do OCR
for original_label, ocr_result in zip(label_data, ocr_results):
    print(f"Rótulo Original: {original_label} | OCR Resultado: {ocr_result}")


# Função para calcular métricas de avaliação
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    return accuracy, precision, recall, f1


# Calcular métricas de avaliação
accuracy, precision, recall, f1 = calculate_metrics(label_data, ocr_results)

print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

