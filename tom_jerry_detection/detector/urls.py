from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('start-video-processing/', views.start_video_processing, name='start_video_processing'),
    path('get-next-prediction/', views.get_next_prediction, name='get_next_prediction'),
]
