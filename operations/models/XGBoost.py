import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from operations import image_manipulation as im
import os

CROPPED_DATASET_DIR = '../../static/cropped_dataset'


def decode_label(label: int, label_encoder: LabelEncoder):
    return label_encoder.inverse_transform(label)


def get_model(mode: str, path: str, label_encoder_path=None):
    if mode == 'binary':
        b_model = xgb.XGBClassifier()
        b_model.load_model(path)
        return b_model
    elif mode == 'multiclass':
        m_model = xgb.XGBClassifier()
        m_model.load_model(path)
        label_encode_multi = LabelEncoder()
        label_encode_multi.classes_ = np.load(label_encoder_path)
        return m_model, label_encode_multi


def train_model(mode: str, dataset: np.array, labels: np.array):
    if mode == 'binary':
        negative_value = 'Negative for intraepithelial lesion'

        # Encode the labels as binary considering 0 for negative_values and 1 for the rest
        binarized_labels = np.where(labels == negative_value, 0, 1)

        # Split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(dataset, binarized_labels, test_size=0.2, random_state=42,
                                                            stratify=binarized_labels)

        # Create the model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation metrics
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
        print('Classification Report: ', classification_report(y_test, y_pred))

        return model
    elif mode == 'multiclass':
        X_train, X_test, y_train, y_test = [], [], [], []

        # Encode the labels as integers
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Save this label encoder for future use
        np.save('../../static/label_encoder.npy', label_encoder.classes_)

        # Split the dataset into training and testing
        current_pos = 0
        for folder in os.listdir(CROPPED_DATASET_DIR):
            qtd_imgs = len(os.listdir(os.path.join(CROPPED_DATASET_DIR, folder)))
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                dataset[current_pos:current_pos + qtd_imgs], encoded_labels[current_pos:current_pos + qtd_imgs],
                test_size=0.2, random_state=42, stratify=encoded_labels[current_pos:current_pos + qtd_imgs])
            X_train.extend(X_train_temp)
            X_test.extend(X_test_temp)
            y_train.extend(y_train_temp)
            y_test.extend(y_test_temp)
            current_pos += qtd_imgs

        model = xgb.XGBClassifier(eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation metrics
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
        print('Classification Report: ', classification_report(y_test, y_pred))

        return model


def predict(model: xgb.XGBClassifier, image: np.array):
    haralick = [im.haralick_gray_scale(image, distances=[1, 2, 4, 8], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])]

    return model.predict(haralick)


if __name__ == '__main__':
    # dataset = np.load('../../static/hara_dataset.npy')
    # labels = np.load('../../static/labels.npy')

    # multiclass_model = train_model('multiclass', dataset, labels)

    # Load the model
    multiclass_model = xgb.XGBClassifier()
    multiclass_model.load_model('../../static/multiclassModel.json')

    # Load the label encoder for future use
    label_encoder_ = LabelEncoder()
    label_encoder_.classes_ = np.load('../../static/label_encoder.npy')

    # Save the model
    multiclass_model.save_model('../../static/multiclassModel.json')

    # Predict
    prediction = predict(multiclass_model, cv2.imread('../../static/530.png'))
    print(label_encoder_.inverse_transform(prediction))

    # binary_model = train_model('binary', dataset, labels)
    # #
    # # Save the model
    # binary_model.save_model('../../static/binaryModel.json')
    #
    # # Load the model
    # # binary_model = xgb.XGBClassifier()
    # # binary_model.load_model('../../static/binaryModel.json')
    #
    # # Predict
    # print(predict(binary_model, cv2.imread('../../static/527.png')))
    # print(predict(binary_model, cv2.imread('../../static/530.png')))
