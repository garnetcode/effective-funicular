from django.urls import path
from . import views

urlpatterns = [
    path('networks/', views.NetworkList.as_view(), name='network-list'),
    path('networks/<str:network_id>/', views.NetworkDetail.as_view(), name='network-detail'),
    path('networks/<str:network_id>/learn_text/', views.LearnText.as_view(), name='learn-text'),
    path('networks/<str:network_id>/organize/', views.Organize.as_view(), name='organize'),
    path('networks/<str:network_id>/structure/', views.NetworkStructure.as_view(), name='network-structure'),
]
