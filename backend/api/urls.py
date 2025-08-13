from django.urls import path
from . import views

urlpatterns = [
    path('agents/', views.AgentList.as_view(), name='agent-list'),
    path('agents/<str:agent_id>/', views.AgentDetail.as_view(), name='agent-detail'),
    path('agents/<str:agent_id>/learn_associative/', views.LearnAssociative.as_view(), name='learn-associative'),
    path('agents/<str:agent_id>/organize_memory/', views.OrganizeMemory.as_view(), name='organize-memory'),
    path('agents/<str:agent_id>/structure/', views.AgentStructure.as_view(), name='agent-structure'),
    path('agents/<str:agent_id>/select_action/', views.SelectAction.as_view(), name='select-action'),
]
