from django.urls import path
from . import views

urlpatterns = [
    path('', views.detect_image, name='home'),  # Define uma view para o caminho vazio
    path('detect/', views.detect_image, name='detect_image'),
]
