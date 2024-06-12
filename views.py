import cv2
from PIL import Image
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.shortcuts import render
import os
import json
import numpy as np

from operations.models import XGBoost as xgb
from operations.models import EfficientNetMulticlass as enetmulti, EfficientNetBinary as enetbinary

from Test import settings
import operations.image_manipulation as im

# Função para renderizar a página inicial
def myview(request):
    return render(request, 'index.html')

# Função para renderizar a página "about us"
def aboutus(request):
    return render(request, 'aboutus.html')

# Função para fazer upload de imagem
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Obtém o arquivo carregado
        image_file = request.FILES['image']
        # Define o caminho onde a imagem será salva (ajuste conforme necessário)
        upload_dir = os.path.join(str(settings.BASE_DIR), "static", "images", image_file.name)
        print("upload_dir: ", upload_dir)
        # Escreve o arquivo carregado na localização especificada
        with open(upload_dir, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        # Redireciona para uma página de sucesso
        return HttpResponseRedirect(f'/?img={image_file.name}')  # Redireciona para uma página de sucesso
    else:
        # Lida com o caso em que nenhum arquivo foi carregado
        return HttpResponseRedirect('/error/')  # Redireciona para uma página de erro

# Função para mudar a imagem exibida
def change_image(request):
    image_url = request.GET.get('img')
    return render(request, 'index.html', {'image_url': image_url})

# Função para converter uma imagem para escala de cinza
def convert_to_grayscale(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img = Image.open(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        # Converte a imagem para escala de cinza
        grayscale_img = im.convert_image_to_gray_scale(img)

        # Define o caminho onde a imagem em escala de cinza será salva
        grayscale_img_path = os.path.join(str(settings.BASE_DIR), "static", "images", "current_altered.png")
        print("grayscale_img_path: ", grayscale_img_path)
        # Salva a imagem em escala de cinza
        grayscale_img.save(grayscale_img_path)
        # Retorna uma resposta JSON com o caminho da imagem em escala de cinza
        return JsonResponse({'grayscale_image_path': '/static/images/current_altered.png'})
    else:
        return HttpResponse(status=400)

# Função para gerar o histograma de uma imagem
def generate_histogram(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img = Image.open(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        if img.mode == "L":
            # Gera o histograma para a imagem em escala de cinza
            histogram = im.generate_image_histogram(img)

            # Retorna o histograma como uma resposta JSON
            return JsonResponse({'imgType': 'grayscale', 'histogram': histogram})
        else:
            # Gera o histograma para cada canal da imagem HSV
            h, s, v = im.generate_image_histogram(img)
            return JsonResponse({'imgType': 'hsv', 'histogram_h': h, 'histogram_s': s, 'histogram_v': v})
    else:
        return HttpResponse(status=400)

# Função para gerar as características de Haralick de uma imagem
def generate_haralick_features(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img = cv2.imread(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        # Calcula as características de Haralick
        features = im.haralick_gray_scale(img, distances=[1, 2, 4, 8], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], api_call=True)

        return JsonResponse(features, safe=False)
    else:
        return HttpResponse(status=400)

# Função para calcular os momentos de Hu de uma imagem
def hu_moments(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img_type = data_dict['type']

        # Calcula os momentos de Hu para a imagem em escala de cinza
        if img_type == 'gray':
            binary_img, hu_moments_values = im.calculate_hu_moments(os.path.join(str(settings.BASE_DIR), "static", "images", image_url), img_type)
            binary_img_path = os.path.join(str(settings.BASE_DIR), "static", "images", "current_altered.png")

            # Salva a imagem binária
            cv2.imwrite(binary_img_path, binary_img)

            return JsonResponse({'binary_image_path': '/static/images/current_altered.png', 'hu_moments': hu_moments_values.tolist()})
        elif img_type == 'color':
            hu_moments_b, hu_moments_g, hu_moments_r = im.calculate_hu_moments(os.path.join(str(settings.BASE_DIR), "static", "images", image_url), img_type)

            return JsonResponse({'hu_moments_b': hu_moments_b.tolist(), 'hu_moments_g': hu_moments_g.tolist(), 'hu_moments_r': hu_moments_r.tolist()})
    else:
        return HttpResponse(status=400)

# Função para classificar uma imagem
def classify_image(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url'] and data_dict['model']:
        image_url = data_dict['image_url']
        img = cv2.imread(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        # Classifica a imagem usando modelos pré-treinados de XGBoost ou EfficientNet
        clazz = None
        clazz_proba = None

        if data_dict['model'] == 'xgboostBinary':
            clazz = xgb.predict(xgb.get_model('binary', os.path.join(str(settings.BASE_DIR), "static", "binaryModel.json")), img)
        elif data_dict['model'] == 'xgboostMulticlass':
            model, label_encoder = xgb.get_model('multiclass', os.path.join(str(settings.BASE_DIR), "static", "multiclassModel.json"), label_encoder_path=os.path.join(str(settings.BASE_DIR), "static", "label_encoder.npy"))
            clazz = label_encoder.inverse_transform(xgb.predict(model, img))
        elif data_dict['model'] == 'efficientNetMulticlass':
            return JsonResponse({'img_class': enetmulti.classify_image(os.path.join(str(settings.BASE_DIR), "static", "images", image_url), os.path.join(str(settings.BASE_DIR), "static", "multiclassEfficientNet.pth"))})
        elif data_dict['model'] == 'efficientNetBinary':
            return JsonResponse({'img_class': enetbinary.classify_image(os.path.join(str(settings.BASE_DIR), "static", "images", image_url), os.path.join(str(settings.BASE_DIR), "static", "binaryEfficientNet.pth"))})

        return JsonResponse({'img_class': str(clazz.tolist()[0])})
    else:
        return HttpResponse(status=400)
