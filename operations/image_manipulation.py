from PIL import Image
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os


def add_gaussian_noise(image: Image, mean=0, std=25):
    # Convert image to a numpy array
    np_image = np.array(image, dtype=np.float32)

    # Generate Gaussian noise
    gauss = np.random.normal(mean, std, np_image.shape).astype(np.float32)

    # Add the noise to the image
    noisy_image = np_image + gauss

    # Clip the pixel values to the range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


def rotate_image(image: Image, angle: int):
    return image.rotate(angle, expand=False)


def pre_process_image(image: np.ndarray, resize=False, resize_width=100, resize_height=100):
    if image is None:
        return None
    pre_processed_img = convert_image_to_gray_scale_cv2(image)
    return resize_image(pre_processed_img, resize_width, resize_height) if resize else pre_processed_img


def resize_image(image: np.ndarray, width: int, height: int):
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def convert_image_to_gray_scale(image: Image):
    return image.convert("L")


def convert_image_to_gray_scale_cv2(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def generate_image_histogram_gray_scale(image: cv2):
    return cv2.calcHist([image], [0], None, [256], [0, 256])


def generate_image_histogram(image: Image):
    # Checks if the image is in gray scale, else converts it to HSV and generates the histogram
    if image.mode == 'L':
        return image.histogram()
    else:
        image_hsv = image.convert('HSV')
        h, s, v = image_hsv.split()
        return h.histogram(), s.histogram(), v.histogram()


def histogram_test():
    image = Image.open("D:\\PAI\\Test\\static\\images\\current_altered.png")
    histogram = generate_image_histogram(image)
    print(histogram)
    print(len(histogram))


def haralick_gray_scale(image: np.ndarray, distances=None, angles=None):
    """
    This function computes Haralick texture features for a grayscale image.
    :param distances: The distance between pixels
    :param angles: The angles for GLCM calculation
    :param image: An RGB Image object
    :return: A dictionary containing the computed Haralick texture features and the original image
    """
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2]
    if len(image.shape) == 3:
        # Step 2: Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Step 3: Compute Haralick texture features
    # Define parameters for GLCM calculation
    # distances = [1]  # distance between pixels
    # angles = [0, np.pi / 4, np.pi / 2]  # angles for GLCM calculation

    # Compute GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Compute Haralick texture features
    # entropy = __entropy(glcm)
    # contrast = graycoprops(glcm, 'contrast')
    # homogeneity = graycoprops(glcm, 'homogeneity')

    # Calcular os descritores de Haralick
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation',
                  'ASM']  # ASM é energia angular do segundo momento

    for prop in properties:
        for angle in range(len(angles)):
            feature = graycoprops(glcm, prop)[0, angle]
            features.append(feature)

    return features


def __entropy(glcm):
    # Normalize GLCM
    glcm_normalized = glcm / np.sum(glcm)

    # Compute entropy
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Add a small epsilon to avoid log(0)

    return entropy


def calculate_hu_moments(image_url: str, type: str):
    # If the image is colored, call __calculate_hu_moments_color
    if type == 'color':
        return __calculate_hu_moments_color(image_url)
    # If the image is grayscale, call __calculate_hu_moments_gray
    elif type == 'gray':
        return __calculate_hu_moments_gray(image_url)


def __calculate_hu_moments_color(image_path: str):
    image = cv2.imread(image_path)

    b, g, r = cv2.split(image)

    # Calculate Hu moments for each channel
    hu_moments_b = cv2.HuMoments(cv2.moments(b)).flatten()
    hu_moments_g = cv2.HuMoments(cv2.moments(g)).flatten()
    hu_moments_r = cv2.HuMoments(cv2.moments(r)).flatten()

    return hu_moments_b, hu_moments_g, hu_moments_r


def __calculate_hu_moments_gray(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to binary using adaptive thresholding
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Calculate the central moments
    central_moments = cv2.moments(binary)

    # Calculate the Hu moments
    hu_moments = cv2.HuMoments(central_moments)

    return binary, hu_moments


def co_ocurrence_matriz(image_path: str):
    image_path = 'D:\\PAI\\Test\\static\\images\\current_altered.png'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Compute GLCM (Gray-Level Co-occurrence Matrix)
    distances = [32]  # distance between pixels
    angles = [0]

    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    print(glcm)
