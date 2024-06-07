from operations import image_manipulation as im
from PIL import Image
import pandas as pd
import numpy as np
import random
import cv2
import os

CLASSIFICATION_FILE = '../static/classifications.csv'
DATASET_DIR = '../static/dataset'
CROPPED_DATASET_DIR = '../static/cropped_dataset'
NUCLEUS_COUNTER = 0
NUCLEUS_TOTAL = 0

counter = 0

def generate_haralick_features():
    dataset = []
    labels = []

    total_imgs = 0
    for folder in os.listdir(CROPPED_DATASET_DIR):
        total_imgs += len(os.listdir(os.path.join(CROPPED_DATASET_DIR, folder)))

    img_counter = 0
    for folder_hara in os.listdir(CROPPED_DATASET_DIR):
        for filename_hara in os.listdir(os.path.join(CROPPED_DATASET_DIR, folder_hara)):
            img_counter += 1
            img = cv2.imread(os.path.join(CROPPED_DATASET_DIR, folder_hara, filename_hara))
            haralick = im.haralick_gray_scale(img, distances=[1, 2, 4, 8], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
            dataset.append(haralick)
            labels.append(folder_hara)
            print(f"Haralick - {img_counter}/{total_imgs}")

    return dataset, labels

def random_remove_images(folder_path, percentage):
    # All files in the folder
    files = os.listdir(folder_path)

    # Number of files to remove
    number_of_files_to_remove = int(len(files) * percentage)

    files_to_remove = random.sample(files, number_of_files_to_remove)
    f_counter = 0
    for file in files_to_remove:
        f_counter += 1
        os.remove(os.path.join(folder_path, file))
        print(f"{f_counter}/{number_of_files_to_remove} files removed")


def rotate_image(image_path, angles):
    images = []
    for angle in angles:
        image = Image.open(image_path)
        images.append(im.rotate_image(image, angle))
    return images


def __save_images(images, cell_id, cell_class):
    save_counter = 0
    for image in images:
        __save_image(image, f"{cell_id}_{save_counter}", cell_class)
        save_counter += 1


def __save_image(image, cell_id, cell_class):
    # global counter
    # Check if exists a directory with the cell_class name
    global counter
    if not os.path.exists(f'{CROPPED_DATASET_DIR}/{cell_class}'):
        os.makedirs(f'{CROPPED_DATASET_DIR}/{cell_class}')
    cv2.imwrite(f'{CROPPED_DATASET_DIR}/{cell_class}/{cell_id}.png', image if not isinstance(image, Image.Image) else np.array(image))
    counter += 1


def __get_png_jpg_filenames(directory):
    filenames = []
    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):
            filenames.append(file)

    return filenames


def __get_image(file_name):
    return cv2.imread(f'{DATASET_DIR}/{file_name}')


def oversampling():
    # Angles to rotate the image
    rotating_angles = [90, 180]
    filters = []

    total_imgs = 0

    for folder_over in os.listdir(CROPPED_DATASET_DIR):
        if folder_over != 'Negative for intraepithelial lesion':
            total_imgs += len(os.listdir(os.path.join(CROPPED_DATASET_DIR, folder_over)))

    image_counter = 0
    for folder_over in os.listdir(CROPPED_DATASET_DIR):
        if folder_over != 'Negative for intraepithelial lesion':
            for filename_over in os.listdir(os.path.join(CROPPED_DATASET_DIR, folder_over)):
                image_counter += 1

                # Rotate the image based on the angles defined in the rotating_angles list
                rotated_images = rotate_image(os.path.join(CROPPED_DATASET_DIR, folder_over, filename_over),
                                              rotating_angles)
                __save_images(rotated_images, filename_over.split(".png")[0], folder_over)

                # Apply filters in the image based on the filters list
                for filter in filters:
                    if filter == 'gaussian':
                        img = cv2.imread(os.path.join(CROPPED_DATASET_DIR, folder_over, filename_over))
                        noisy_image = im.add_gaussian_noise(img)
                        __save_image(noisy_image, f"{filename_over.split('.png')[0]}_gaussian", folder_over)
                print(f"Oversampled - {image_counter}/{total_imgs}")



def get_cell_nucleus(row):
    global NUCLEUS_COUNTER
    # Read an image from the __get_image_path function
    img = __get_image(row['image_filename'])

    # Get the nucleus_x and nucleus_y values from the dataset
    nucleus_x = row['nucleus_x']
    nucleus_y = row['nucleus_y']

    # Define the size of the square to extract
    SIZE_SQUARE = 100

    # Calculate the boundaries of the region to extract
    left = max(0, nucleus_x - int(SIZE_SQUARE / 2))
    top = max(0, nucleus_y - int(SIZE_SQUARE / 2))
    right = min(img.shape[1], nucleus_x + int(SIZE_SQUARE / 2))
    bottom = min(img.shape[0], nucleus_y + int(SIZE_SQUARE / 2))

    # Extract the region from the image
    img = img[top:bottom, left:right]

    # Create a black canvas of size 100x100
    canvas = np.zeros((SIZE_SQUARE, SIZE_SQUARE, 3), dtype=np.uint8)

    # Calculate the position to paste the extracted image onto the canvas
    x_offset = max(0, int(SIZE_SQUARE / 2) - nucleus_x + left)
    y_offset = max(0, int(SIZE_SQUARE / 2) - nucleus_y + top)

    # Paste the extracted image onto the canvas
    canvas[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    NUCLEUS_COUNTER += 1

    print(f'Loaded - {NUCLEUS_COUNTER}/{NUCLEUS_TOTAL}')

    return canvas


def crop_images():
    global NUCLEUS_TOTAL

    # Read the dataset .csv file
    df = pd.read_csv(CLASSIFICATION_FILE, sep=',')

    # Filter the dataset to only include images that the image_filename is in the list of filenames
    df = df[df['image_filename'].isin(__get_png_jpg_filenames(DATASET_DIR))]
    # Don't crop the ones with cell id equals to 530 and 527
    df = df[df['cell_id'] != 527]
    df = df[df['cell_id'] != 530]

    # Crop the dataset based on the nucleus_x and nucleus_y columns found in the dataset
    NUCLEUS_TOTAL = len(df)
    [__save_image(get_cell_nucleus(row), row["cell_id"], row["bethesda_system"]) for index, row in
     df.iterrows()]


def exclude_black_images(percentage=0.5, classes=None, excluded_classes=None):
    for folder_exclude in os.listdir(CROPPED_DATASET_DIR):
        if excluded_classes is not None and folder_exclude in excluded_classes:
            continue
        if classes is None or folder_exclude in classes:
            for filename_exclude in os.listdir(os.path.join(CROPPED_DATASET_DIR, folder_exclude)):
                img = cv2.imread(os.path.join(CROPPED_DATASET_DIR, folder_exclude, filename_exclude))
                img_gray = im.convert_image_to_gray_scale_cv2(img)
                histogram = im.generate_image_histogram_gray_scale(img_gray)

                # If the image has more than the previous defined amount of black pixels, remove it
                if histogram[0] > percentage * img_gray.size:
                    os.remove(os.path.join(CROPPED_DATASET_DIR, folder_exclude, filename_exclude))


def count_images():
    amount_of_images = 0
    binary_counting = [0, 0]

    for folder in os.listdir(CROPPED_DATASET_DIR):
        amount_img_this_class = 0
        if folder == 'Negative for intraepithelial lesion':
            binary_counting[0] = len(os.listdir(os.path.join(CROPPED_DATASET_DIR, folder)))
        else:
            binary_counting[1] += len(os.listdir(os.path.join(CROPPED_DATASET_DIR, folder)))
        amount_img_this_class += len(os.listdir(os.path.join(CROPPED_DATASET_DIR, folder)))
        amount_of_images += len(os.listdir(os.path.join(CROPPED_DATASET_DIR, folder)))
        print(f"Amount of images in class {folder}: {amount_img_this_class}")

    print(f"--x-- Amount of images left: {amount_of_images}")
    print(f"--x-- Binary count: .0: {binary_counting[0]} | .1: {binary_counting[1]}")


def load_images():
    # 1. Step: Crop the images
    crop_images()

    count_images()

    # 2. Step: Exclude negative classes images with 30% or more of black pixels, and the normal ones with 70% or more
    exclude_black_images(percentage=0.3, classes=['Negative for intraepithelial lesion'])
    exclude_black_images(percentage=0.5, excluded_classes=['Negative for intraepithelial lesion'])

    print("\nExcluded images with more than 30% of black pixels in the negative class and 70% in the rest of the classes")

    # 3. Step: Oversample the images
    oversampling()

    # 4. Step: Balance the classes by removing images from the negative class, which has more images
    random_remove_images(os.path.join(CROPPED_DATASET_DIR, 'Negative for intraepithelial lesion'), 0.1)

    print("\n\n")
    count_images()


if __name__ == '__main__':
    # load_images()

    count_images()

    hara_dataset, labels = generate_haralick_features()
    np.save('../static/hara_dataset.npy', hara_dataset)
    np.save('../static/labels.npy', labels)

