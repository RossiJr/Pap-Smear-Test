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


def myview(request):
    return render(request, 'index.html')


def modelpage(request):
    return render(request, 'modelpage.html')


def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded file
        image_file = request.FILES['image']
        # Define the path where you want to save the uploaded image
        # Adjust the path according to your project structure
        upload_dir = os.path.join(str(settings.BASE_DIR), "static", "images", image_file.name)
        print("upload_dir: ", upload_dir)
        # Write the uploaded file to the specified location
        with open(upload_dir, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        # Redirect to a success page or perform any other action
        return HttpResponseRedirect(f'/?img={image_file.name}')  # Redirect to a success page
    else:
        # Handle the case when no file is uploaded
        return HttpResponseRedirect('/error/')  # Redirect to an error page


def change_image(request):
    image_url = request.GET.get('img')
    return render(request, 'index.html', {'image_url': image_url})


def convert_to_grayscale(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img = Image.open(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        # Convert the image to grayscale
        grayscale_img = im.convert_image_to_gray_scale(img)

        # Define the path where you want to save the grayscale image
        grayscale_img_path = os.path.join(str(settings.BASE_DIR), "static", "images", "current_altered.png")
        print("grayscale_img_path: ", grayscale_img_path)
        # Update this with your desired path
        grayscale_img.save(grayscale_img_path)
        # Return a JSON response with the path to the grayscale image
        return JsonResponse({'grayscale_image_path': '/static/images/current_altered.png'})
    else:
        return HttpResponse(status=400)


def generate_histogram(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img = Image.open(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        if img.mode == "L":
            # Generate the histogram for the image
            histogram = im.generate_image_histogram(img)

            # Return the histogram as a JSON response
            return JsonResponse({'imgType': 'grayscale', 'histogram': histogram})
        else:
            h, s, v = im.generate_image_histogram(img)
            return JsonResponse({'imgType': 'hsv', 'histogram_h': h, 'histogram_s': s, 'histogram_v': v})
    else:
        return HttpResponse(status=400)


def generate_haralick_features(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img = cv2.imread(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        features = im.haralick_gray_scale(img, distances=[1, 2, 4, 8], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                          api_call=True)

        return JsonResponse(features, safe=False)
    else:
        return HttpResponse(status=400)


def hu_moments(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img_type = data_dict['type']
        # Calculate Hu Moments for the grayscale image

        if img_type == 'gray':
            binary_img, hu_moments_values = im.calculate_hu_moments(
                os.path.join(str(settings.BASE_DIR), "static", "images", image_url), img_type)
            binary_img_path = os.path.join(str(settings.BASE_DIR), "static", "images", "current_altered.png")

            cv2.imwrite(binary_img_path, binary_img)

            return JsonResponse({'binary_image_path': '/static/images/current_altered.png',
                                 'hu_moments': hu_moments_values.tolist()})
        elif img_type == 'color':
            hu_moments_b, hu_moments_g, hu_moments_r = im.calculate_hu_moments(
                os.path.join(str(settings.BASE_DIR), "static", "images", image_url), img_type)

            return JsonResponse({'hu_moments_b': hu_moments_b.tolist(), 'hu_moments_g': hu_moments_g.tolist(),
                                 'hu_moments_r': hu_moments_r.tolist()})
    else:
        return HttpResponse(status=400)


def classify_image(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url'] and data_dict['model']:
        image_url = data_dict['image_url']
        img = cv2.imread(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        # Classify the image using the pre-trained XGBoost models
        clazz = None
        clazz_proba = None

        if data_dict['model'] == 'xgboostBinary':
            clazz = xgb.predict(
                xgb.get_model('binary', os.path.join(str(settings.BASE_DIR), "static", "binaryModel.json")), img)
        elif data_dict['model'] == 'xgboostMulticlass':
            model, label_encoder = xgb.get_model('multiclass',
                                                 os.path.join(str(settings.BASE_DIR), "static", "multiclassModel.json"),
                                                 label_encoder_path=os.path.join(str(settings.BASE_DIR), "static",
                                                                                 "label_encoder.npy"))
            clazz = label_encoder.inverse_transform(xgb.predict(model, img))
        elif data_dict['model'] == 'efficientNetMulticlass':
            return JsonResponse(
                {'img_class': enetmulti.classify_image(
                    os.path.join(str(settings.BASE_DIR), "static", "images", image_url),
                    os.path.join(str(settings.BASE_DIR), "static", "multiclassEfficientNet.pth"))})
        elif data_dict['model'] == 'efficientNetBinary':
            return JsonResponse(
                {'img_class': enetbinary.classify_image(
                    os.path.join(str(settings.BASE_DIR), "static", "images", image_url),
                    os.path.join(str(settings.BASE_DIR), "static", "binaryEfficientNet.pth"))})

        return JsonResponse({'img_class': str(clazz.tolist()[0])})
    else:
        return HttpResponse(status=400)


def get_model_classification_report(request):
    if request.method == 'GET' and 'model' in request.GET:
        model = request.GET.get('model')
        if model == 'xgboostBinary':
            return JsonResponse({
                'negative': {
                    'precision': 0.85,
                    'recall': 0.95,
                    'f1_score': 0.90,
                    'support': 849
                },
                'positive': {
                    'precision': 0.74,
                    'recall': 0.47,
                    'f1_score': 0.58,
                    'support': 268
                }
            })
        elif model == 'xgboostMulticlass':
            return JsonResponse({
                'ASC_H': {
                    'precision': 1,
                    'recall': 0.10,
                    'f1_score': 0.30,
                    'support': 17
                },
                'ASC_US': {
                    'precision': 0.67,
                    'recall': 0.08,
                    'f1_score': 0.14,
                    'support': 78
                },
                'HSIL': {
                    'precision': 0.63,
                    'recall': 0.64,
                    'f1_score': 0.63,
                    'support': 58
                },
                'LSIL': {
                    'precision': 0.62,
                    'recall': 0.10,
                    'f1_score': 0.17,
                    'support': 99
                },
                'negative': {
                    'precision': 0.83,
                    'recall': 0.99,
                    'f1_score': 0.90,
                    'support': 849
                },
                'SCC': {
                    'precision': 0.33,
                    'recall': 0.19,
                    'f1_score': 0.24,
                    'support': 16
                },
            })
        elif model == 'efficientNetMulticlass':
            return JsonResponse({
                'ASC_H': {
                    'precision': 0.81,
                    'recall': 0.80,
                    'f1_score': 0.80,
                    'support': 69
                },
                'ASC_US': {
                    'precision': 0.72,
                    'recall': 0.69,
                    'f1_score': 0.70,
                    'support': 283
                },
                'HSIL': {
                    'precision': 0.81,
                    'recall': 0.90,
                    'f1_score': 0.85,
                    'support': 255
                },
                'LSIL': {
                    'precision': 0.83,
                    'recall': 0.69,
                    'f1_score': 0.75,
                    'support': 394
                },
                'negative': {
                    'precision': 0.82,
                    'recall': 0.96,
                    'f1_score': 0.89,
                    'support': 391
                },
                'SCC': {
                    'precision': 0.86,
                    'recall': 0.70,
                    'f1_score': 0.77,
                    'support': 63
                },
            })
        elif model == 'efficientNetBinary':
            return JsonResponse({
                'positive': {
                    'precision': 0.99,
                    'recall': 0.98,
                    'f1_score': 0.99,
                    'support': 1097
                },
                'negative': {
                    'precision': 0.95,
                    'recall': 0.97,
                    'f1_score': 0.96,
                    'support': 358
                }
            })
