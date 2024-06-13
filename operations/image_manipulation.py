from PIL import Image
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os

def add_gaussian_noise(image: Image, mean=0, std=25):
    # Converte a imagem para um array numpy de float32
    np_image = np.array(image, dtype=np.float32)

    # Gera ruído Gaussiano com média e desvio padrão fornecidos
    gauss = np.random.normal(mean, std, np_image.shape).astype(np.float32)

    # Adiciona o ruído à imagem
    noisy_image = np_image + gauss

    # Limita os valores dos pixels ao intervalo [0, 255] e converte de volta para uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def rotate_image(image: Image, angle: int):
    # Rotaciona a imagem pelo ângulo especificado sem expandir o tamanho da imagem
    return image.rotate(angle, expand=False)

def pre_process_image(image: np.ndarray, resize=False, resize_width=100, resize_height=100):
    if image is None:
        return None
    # Converte a imagem para escala de cinza usando a função do OpenCV
    pre_processed_img = convert_image_to_gray_scale_cv2(image)
    # Redimensiona a imagem se necessário
    return resize_image(pre_processed_img, resize_width, resize_height) if resize else pre_processed_img

def resize_image(image: np.ndarray, width: int, height: int):
    # Redimensiona a imagem para a largura e altura especificadas
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def convert_image_to_gray_scale(image: Image):
    # Converte a imagem PIL para escala de cinza
    return image.convert("L")

def convert_image_to_gray_scale_cv2(image: np.ndarray):
    # Converte a imagem OpenCV para escala de cinza
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def generate_image_histogram_gray_scale(image: cv2):
    # Gera o histograma para uma imagem em escala de cinza usando OpenCV
    return cv2.calcHist([image], [0], None, [256], [0, 256])

def generate_image_histogram(image: Image):
    # Verifica se a imagem está em escala de cinza e gera o histograma
    if image.mode == 'L':
        return image.histogram()
    else:
        # Converte a imagem para HSV e gera os histogramas para cada canal
        image_hsv = image.convert('HSV')
        h, s, v = image_hsv.split()
        return h.histogram(), s.histogram(), v.histogram()

def histogram_test():
    image = Image.open("D:\\PAI\\Test\\static\\images\\current_altered.png")
    # Gera e imprime o histograma da imagem carregada
    histogram = generate_image_histogram(image)
    print(histogram)
    print(len(histogram))

def haralick_gray_scale(image: np.ndarray, distances=None, angles=None, api_call=False):
    """
    Esta função calcula as características de textura de Haralick para uma imagem em escala de cinza.
    : distances: A distância entre os pixels
    : angles: Os ângulos para o cálculo do GLCM
    : image: Um objeto de imagem RGB
    :return: Um dicionário contendo as características de textura de Haralick e a imagem original
    """
    # Distâncias padrão usadas para calcular o GLCM
    if distances is None:
        distances = [1]

    # Ângulos padrão usados para calcular o GLCM
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2]

    # Se a imagem for colorida, converte para escala de cinza
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Computa o GLCM (Matriz de Co-ocorrência de Níveis de Cinza)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Computa as características de textura de Haralick
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']  # ASM é energia angular do segundo momento
    features_api = []

    if api_call:
        for prop in properties:
            feature_temp = graycoprops(glcm, prop)
            feature = {'property': prop, 'values': []}
            for i in range(len(distances)):
                feature['values'].append({'distance': distances[i], 'values': []})
                for j in range(len(angles)):
                    feature['values'][i]['values'].append({'angle': angles[j], 'value': feature_temp[i, j]})
            features_api.append(feature)
        return features_api
    else:
        for prop in properties:
            feature_temp = graycoprops(glcm, prop)
            for i in range(len(distances)):
                for j in range(len(angles)):
                    features.append(feature_temp[i, j])
        return features

def __entropy(glcm):
    # Normaliza o GLCM
    glcm_normalized = glcm / np.sum(glcm)

    # Computa a entropia
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Adiciona um pequeno epsilon para evitar log(0)

    return entropy

def calculate_hu_moments(image_url: str, type: str):
    # Se a imagem for colorida, chama __calculate_hu_moments_color
    if type == 'color':
        return __calculate_hu_moments_color(image_url)
    # Se a imagem for em escala de cinza, chama __calculate_hu_moments_gray
    elif type == 'gray':
        return __calculate_hu_moments_gray(image_url)

def __calculate_hu_moments_color(image_path: str):
    image = cv2.imread(image_path)

    # Separa a imagem em três canais (B, G, R)
    b, g, r = cv2.split(image)

    # Calcula os momentos de Hu para cada canal
    hu_moments_b = cv2.HuMoments(cv2.moments(b)).flatten()
    hu_moments_g = cv2.HuMoments(cv2.moments(g)).flatten()
    hu_moments_r = cv2.HuMoments(cv2.moments(r)).flatten()

    return hu_moments_b, hu_moments_g, hu_moments_r

def __calculate_hu_moments_gray(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Converte a imagem para binário usando limiarização adaptativa
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Calcula os momentos centrais
    central_moments = cv2.moments(binary)

    # Calcula os momentos de Hu
    hu_moments = cv2.HuMoments(central_moments)

    return binary, hu_moments

def co_ocurrence_matriz(image_path: str):
    image_path = 'D:\\PAI\\Test\\static\\images\\current_altered.png'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Computa o GLCM (Matriz de Co-ocorrência de Níveis de Cinza)
    distances = [32]  # distância entre os pixels
    angles = [0]

    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    print(glcm)
