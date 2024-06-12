import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from operations import image_manipulation as im
import os

# Diretório onde está o dataset recortado
CROPPED_DATASET_DIR = '../../static/cropped_dataset'

# Função para decodificar um label
def decode_label(label: int, label_encoder: LabelEncoder):
    return label_encoder.inverse_transform(label)

# Função para obter um modelo salvo
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

# Função para treinar um modelo
def train_model(mode: str, dataset: np.array, labels: np.array):
    if mode == 'binary':
        negative_value = 'Negative for intraepithelial lesion'

        # Codifica os labels como binários (0 para valores negativos e 1 para o resto)
        binarized_labels = np.where(labels == negative_value, 0, 1)

        # Divide o dataset em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(dataset, binarized_labels, test_size=0.2, random_state=42, stratify=binarized_labels)

        # Cria o modelo
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        # Treina o modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métricas de avaliação
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
        print('Classification Report: ', classification_report(y_test, y_pred))

        return model
    elif mode == 'multiclass':
        X_train, X_test, y_train, y_test = [], [], [], []

        # Codifica os labels como inteiros
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Salva o label encoder para uso futuro
        np.save('../../static/label_encoder.npy', label_encoder.classes_)

        # Divide o dataset em treino e teste
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

        # Métricas de avaliação
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
        print('Classification Report: ', classification_report(y_test, y_pred))

        return model

# Função para fazer predições
def predict(model: xgb.XGBClassifier, image: np.array):
    haralick = [im.haralick_gray_scale(image, distances=[1, 2, 4, 8], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])]
    return model.predict(haralick)

# Execução principal
if __name__ == '__main__':
        # Carrega o dataset e labels
    # dataset = np.load('../../static/hara_dataset.npy')
    # labels = np.load('../../static/labels.npy')

    # Treina o modelo multiclasse
    # multiclass_model = train_model('multiclass', dataset, labels)
    
    # Carrega o modelo multiclasse salvo
    multiclass_model = xgb.XGBClassifier()
    multiclass_model.load_model('../../static/multiclassModel.json')

    # Carrega o label encoder salvo para uso futuro
    label_encoder_ = LabelEncoder()
    label_encoder_.classes_ = np.load('../../static/label_encoder.npy')

    # Salva o modelo multiclasse
    multiclass_model.save_model('../../static/multiclassModel.json')

    # Faz uma predição com o modelo multiclasse em uma imagem especificada
    prediction = predict(multiclass_model, cv2.imread('../../static/530.png'))
    print(label_encoder_.inverse_transform(prediction))

    # Treina o modelo binário
    # binary_model = train_model('binary', dataset, labels)
    
    # Salva o modelo
    # binary_model.save_model('../../static/binaryModel.json')
    
    # Carrega o modelo
    # binary_model = xgb.XGBClassifier()
    # binary_model.load_model('../../static/binaryModel.json')
    
    # Faz predição
    # print(predict(binary_model, cv2.imread('../../static/527.png')))
    # print(predict(binary_model, cv2.imread('../../static/530.png')))