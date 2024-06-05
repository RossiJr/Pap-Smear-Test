import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from operations import image_manipulation as im

def get_model(mode: str, path: str):
    if mode == 'binary':
        b_model = xgb.XGBClassifier()
        b_model.load_model(path)
        return b_model


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


def predict(model: xgb.XGBClassifier, image: np.array):
    haralick = [im.haralick_gray_scale(image, distances=[1, 2, 4, 8], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])]

    return model.predict(haralick)


if __name__ == '__main__':
    # dataset = np.load('../../static/hara_dataset.npy')
    # labels = np.load('../../static/labels.npy')
    # binary_model = train_model('binary', dataset, labels)
    #
    # # Save the model
    # binary_model.save_model('../../static/binaryModel.json')

    # Load the model
    binary_model = xgb.XGBClassifier()
    binary_model.load_model('../../static/binaryModel.json')

    # Predict
    print(predict(binary_model, cv2.imread('../../static/527.png')))
    print(predict(binary_model, cv2.imread('../../static/530.png')))
