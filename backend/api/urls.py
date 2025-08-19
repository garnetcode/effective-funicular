from django.urls import path
from . import views

urlpatterns = [
    path('environments/', views.EnvironmentList.as_view(), name='environment-list'),
    path('cortex_specifications/', views.CortexSpecificationList.as_view(), name='cortex-specification-list'),
    path('agents/', views.AgentList.as_view(), name='agent-list'),
    path('agents/<str:agent_id>/', views.AgentDetail.as_view(), name='agent-detail'),
    path('agents/<str:agent_id>/learn_associative/', views.LearnAssociative.as_view(), name='learn-associative'),
    path('agents/<str:agent_id>/organize_memory/', views.OrganizeMemory.as_view(), name='organize-memory'),
    path('agents/<str:agent_id>/consolidate_memories/', views.ConsolidateMemories.as_view(), name='consolidate-memories'),
    path('agents/<str:agent_id>/structure/', views.AgentStructure.as_view(), name='agent-structure'),
    path('agents/<str:agent_id>/probe_activity/', views.ProbeActivity.as_view(), name='probe-activity'),
    path('agents/<str:agent_id>/select_action/', views.SelectAction.as_view(), name='select-action'),
    path('agents/<str:agent_id>/start_training/', views.StartTraining.as_view(), name='start-training'),
]
