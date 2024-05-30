from PIL import Image
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os


def convert_image_to_gray_scale(image: Image):
    return image.convert("L")


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


def haralick_gray_scale(image: Image):
    """
    This function computes Haralick texture features for a grayscale image.
    :param image: An RGB Image object
    :return: A dictionary containing the computed Haralick texture features and the original image
    """
    if image.mode != 'L':
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Step 2: Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = np.array(image)

    # Step 3: Compute Haralick texture features
    # Define parameters for GLCM calculation
    distances = [1, 2, 4, 8, 16, 32]  # distance between pixels
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # angles for GLCM calculation

    # Compute GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Compute Haralick texture features
    entropy = __entropy(glcm)
    contrast = graycoprops(glcm, 'contrast')
    homogeneity = graycoprops(glcm, 'homogeneity')

    return {'contrast': contrast[0, 0], 'entropy': entropy, 'homogeneity': homogeneity[0, 0],
            'image': gray_image}


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
