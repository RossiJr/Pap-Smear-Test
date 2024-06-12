import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random
from operations import image_manipulation as im

# Arquivos e diretórios de entrada e saída
CLASSIFICATION_FILE = '../static/classifications.csv'
DATASET_DIR = '../static/dataset'
CROPPED_DATASET_DIR = '../static/cropped_dataset'

# Variáveis globais para contagem
NUCLEUS_COUNTER = 0
NUCLEUS_TOTAL = 0
counter = 0

def generate_haralick_features():
    """
    Função para gerar características de Haralick de todas as imagens no diretório CROPPED_DATASET_DIR.
    """
    dataset = []
    labels = []

    # Conta o total de imagens no diretório cropped_dataset
    total_imgs = sum(len(files) for _, _, files in os.walk(CROPPED_DATASET_DIR))

    img_counter = 0
    for folder in os.listdir(CROPPED_DATASET_DIR):
        folder_path = os.path.join(CROPPED_DATASET_DIR, folder)
        for filename in os.listdir(folder_path):
            img_counter += 1
            img = cv2.imread(os.path.join(folder_path, filename))
            # Gera as características de Haralick para a imagem
            haralick = im.haralick_gray_scale(img, distances=[1, 2, 4, 8], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
            dataset.append(haralick)
            labels.append(folder)
            print(f"Haralick - {img_counter}/{total_imgs}")

    return dataset, labels

def random_remove_images(folder_path, percentage):
    """
    Remove uma porcentagem aleatória de imagens de um diretório.
    """
    files = os.listdir(folder_path)
    number_of_files_to_remove = int(len(files) * percentage)
    files_to_remove = random.sample(files, number_of_files_to_remove)

    for i, file in enumerate(files_to_remove, 1):
        os.remove(os.path.join(folder_path, file))
        print(f"{i}/{number_of_files_to_remove} files removed")

def rotate_image(image_path, angles):
    """
    Rotaciona a imagem nos ângulos especificados.
    """
    image = Image.open(image_path)
    return [im.rotate_image(image, angle) for angle in angles]

def save_images(images, cell_id, cell_class):
    """
    Salva várias imagens geradas a partir de uma imagem original.
    """
    for i, image in enumerate(images):
        save_image(image, f"{cell_id}_{i}", cell_class)

def save_image(image, cell_id, cell_class):
    """
    Salva uma única imagem no diretório especificado.
    """
    global counter
    folder_path = os.path.join(CROPPED_DATASET_DIR, cell_class)
    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(os.path.join(folder_path, f"{cell_id}.png"), np.array(image) if isinstance(image, Image.Image) else image)
    counter += 1

def get_png_jpg_filenames(directory):
    """
    Retorna uma lista de arquivos que terminam com .jpg ou .png em um diretório.
    """
    return [file for file in os.listdir(directory) if file.endswith('.jpg') or file.endswith('.png')]

def get_image(file_name):
    """
    Lê uma imagem a partir de seu nome de arquivo.
    """
    return cv2.imread(os.path.join(DATASET_DIR, file_name))

def oversampling():
    """
    Realiza oversampling nas imagens, rotacionando-as e aplicando filtros.
    """
    rotating_angles = [90, 180]
    filters = []

    # Conta o total de imagens a serem processadas para oversampling
    total_imgs = sum(len(files) for folder in os.listdir(CROPPED_DATASET_DIR) if folder != 'Negative for intraepithelial lesion' for _, _, files in os.walk(os.path.join(CROPPED_DATASET_DIR, folder)))

    img_counter = 0
    for folder in os.listdir(CROPPED_DATASET_DIR):
        if folder == 'Negative for intraepithelial lesion':
            continue
        folder_path = os.path.join(CROPPED_DATASET_DIR, folder)
        for filename in os.listdir(folder_path):
            img_counter += 1

            # Rotaciona as imagens nos ângulos especificados
            rotated_images = rotate_image(os.path.join(folder_path, filename), rotating_angles)
            save_images(rotated_images, filename.split(".png")[0], folder)

            # Aplica filtros (se houver) e salva as imagens resultantes
            for filter in filters:
                if filter == 'gaussian':
                    img = cv2.imread(os.path.join(folder_path, filename))
                    noisy_image = im.add_gaussian_noise(img)
                    save_image(noisy_image, f"{filename.split('.png')[0]}_gaussian", folder)

            print(f"Oversampled - {img_counter}/{total_imgs}")

def get_cell_nucleus(row):
    """
    Extrai a região do núcleo de uma célula em uma imagem.
    """
    global NUCLEUS_COUNTER

    img = get_image(row['image_filename'])
    nucleus_x, nucleus_y = row['nucleus_x'], row['nucleus_y']
    SIZE_SQUARE = 100

    # Define os limites da região a ser extraída
    left = max(0, nucleus_x - SIZE_SQUARE // 2)
    top = max(0, nucleus_y - SIZE_SQUARE // 2)
    right = min(img.shape[1], nucleus_x + SIZE_SQUARE // 2)
    bottom = min(img.shape[0], nucleus_y + SIZE_SQUARE // 2)

    img = img[top:bottom, left:right]

    # Cria um canvas preto de tamanho 100x100
    canvas = np.zeros((SIZE_SQUARE, SIZE_SQUARE, 3), dtype=np.uint8)
    x_offset = max(0, SIZE_SQUARE // 2 - nucleus_x + left)
    y_offset = max(0, SIZE_SQUARE // 2 - nucleus_y + top)

    canvas[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    NUCLEUS_COUNTER += 1
    print(f'Loaded - {NUCLEUS_COUNTER}/{NUCLEUS_TOTAL}')

    return canvas

def crop_images():
    """
    Corta as imagens para obter apenas a região do núcleo da célula.
    """
    global NUCLEUS_TOTAL

    df = pd.read_csv(CLASSIFICATION_FILE)
    df = df[df['image_filename'].isin(get_png_jpg_filenames(DATASET_DIR))]
    df = df[~df['cell_id'].isin([527, 530])]

    NUCLEUS_TOTAL = len(df)
    for _, row in df.iterrows():
        save_image(get_cell_nucleus(row), row["cell_id"], row["bethesda_system"])

def exclude_black_images(percentage=0.5, classes=None, excluded_classes=None):
    """
    Exclui imagens que possuem uma porcentagem de pixels pretos acima de um determinado limiar.
    """
    for folder in os.listdir(CROPPED_DATASET_DIR):
        if excluded_classes and folder in excluded_classes:
            continue
        if classes is None or folder in classes:
            folder_path = os.path.join(CROPPED_DATASET_DIR, folder)
            for filename in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, filename))
                img_gray = im.convert_image_to_gray_scale_cv2(img)
                histogram = im.generate_image_histogram_gray_scale(img_gray)
                if histogram[0] > percentage * img_gray.size:
                    os.remove(os.path.join(folder_path, filename))

def count_images():
    """
    Conta e exibe a quantidade de imagens em cada classe.
    """
    total_images = 0
    binary_count = [0, 0]

    for folder in os.listdir(CROPPED_DATASET_DIR):
        folder_path = os.path.join(CROPPED_DATASET_DIR, folder)
        class_image_count = len(os.listdir(folder_path))
        total_images += class_image_count
        if folder == 'Negative for intraepithelial lesion':
            binary_count[0] = class_image_count
        else:
            binary_count[1] += class_image_count
        print(f"Amount of images in class {folder}: {class_image_count}")

    print(f"--x-- Total images: {total_images}")
    print(f"--x-- Binary count: 0: {binary_count[0]} | 1: {binary_count[1]}")

def load_images():
    """
    Carrega, processa e equilibra as imagens.
    """
    crop_images()
    count_images()

    exclude_black_images(percentage=0.3, classes=['Negative for intraepithelial lesion'])
    exclude_black_images(percentage=0.5, excluded_classes=['Negative for intraepithelial lesion'])

    print("\nExcluded images with more than 30% black pixels in the negative class and 70% in the rest of the classes")

    oversampling()

    random_remove_images(os.path.join(CROPPED_DATASET_DIR, 'Negative for intraepithelial lesion'), 0.1)

    print("\n\n")
    count_images()

if __name__ == '__main__':
    count_images()

    hara_dataset, labels = generate_haralick_features()
    np.save('../static/hara_dataset.npy', hara_dataset)
    np.save('../static/labels.npy', labels)
