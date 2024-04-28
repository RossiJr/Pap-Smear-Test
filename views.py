import cv2
from PIL import Image
from django.contrib.sites import requests
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.shortcuts import render
import os
import json

from io import BytesIO

from Test import settings
from operations.image_manipulation import convert_image_to_gray_scale, generate_image_histogram, haralick_gray_scale


def myview(request):
    return render(request, 'index.html')


def aboutus(request):
    return render(request, 'aboutus.html')


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
        grayscale_img = convert_image_to_gray_scale(img)

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
            histogram = generate_image_histogram(img)

            # Return the histogram as a JSON response
            return JsonResponse({'imgType': 'grayscale', 'histogram': histogram})
        else:
            h, s, v = generate_image_histogram(img)
            return JsonResponse({'imgType': 'hsv', 'histogram_h': h, 'histogram_s': s, 'histogram_v': v})
    else:
        return HttpResponse(status=400)


def generate_haralick_features(request):
    data_dict = json.loads(request.body.decode("utf-8"))
    if request.method == 'POST' and data_dict['image_url']:
        image_url = data_dict['image_url']
        img = Image.open(os.path.join(str(settings.BASE_DIR), "static", "images", image_url))

        features = haralick_gray_scale(img)
        # Save a cv2 image
        cv2_image = features['image']
        cv2_image_path = os.path.join(str(settings.BASE_DIR), "static", "images", "current_altered.png")
        cv2.imwrite(cv2_image_path, cv2_image)

        return JsonResponse({'img_path': '/static/images/current_altered.png',
                             'contrast': features['contrast'],
                             'dissimilarity': features['dissimilarity'],
                             'homogeneity': features['homogeneity'],
                             'energy': features['energy'],
                             'correlation': features['correlation']})
    else:
        return HttpResponse(status=400)
