from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Route for the new BrainMonitor
    re_path(r'ws/brain/$', consumers.BrainConsumer.as_asgi()),

    # Original routes
    re_path(r'ws/agent/(?P<session_id>[\w-]+)/$', consumers.AgentConsumer.as_asgi()),
    re_path(r'ws/session_viewer/(?P<session_id>[\w-]+)/$', consumers.SessionViewerConsumer.as_asgi()),
    re_path(r'ws/day_trader_data/$', consumers.DayTraderDataConsumer.as_asgi()),
    re_path(r'ws/unified/(?P<session_id>[\w-]+)/$', consumers.UnifiedSessionConsumer.as_asgi()),

    # Legacy route
    re_path(r'ws/game/(?P<session_id>[\w-]+)/$', consumers.GameConsumer.as_asgi()),
]
