from django.urls import path

import views

urlpatterns = [
    path('', views.myview),
    path('aboutus/', views.aboutus),
    path('api/upload/', views.upload_image, name='upload_image'),
    path('img/', views.change_image, name='change_image'),
    path('api/convert_to_grayscale/', views.convert_to_grayscale, name='convert_to_grayscale'),
    path('api/generate_haralick_features/', views.generate_haralick_features, name='generate_haralick_features'),
    path('api/generate_histogram/', views.generate_histogram, name='generate_histogram'),
    path('api/hu_moments/', views.hu_moments, name='hu_moments'),
]
