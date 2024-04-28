from PIL import Image
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


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
    distances = [1]  # distance between pixels
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # angles for GLCM calculation

    # Compute GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Compute Haralick texture features
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    return {'contrast': contrast[0, 0], 'dissimilarity': dissimilarity[0, 0], 'homogeneity': homogeneity[0, 0],
            'energy': energy[0, 0], 'correlation': correlation[0, 0], 'image': gray_image}
