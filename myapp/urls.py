from django.urls import path

from . import views
urlpatterns = [
    path('m1', views.index, name='index'),
    path('handle_image', views.handle_image, name='handle_image')
]