from django.urls import path
from . import views

urlpatterns = [
    path('agents/', views.AgentList.as_view(), name='agent-list'),
    path('agents/<str:agent_id>/', views.AgentDetail.as_view(), name='agent-detail'),
    path('agents/<str:agent_id>/learn/', views.Learn.as_view(), name='learn'),
    path('agents/<str:agent_id>/structure/', views.AgentStructure.as_view(), name='agent-structure'),
    path('agents/<str:agent_id>/select_action/', views.SelectAction.as_view(), name='select-action'),
]
