from django.urls import re_path
from api import consumers

websocket_urlpatterns = [
    re_path(r"ws/training/(?P<agent_id>[\w-]+)/$", consumers.TrainingConsumer.as_asgi()),
]
