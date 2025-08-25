from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .signals import agent_data_signal
import json

@receiver(agent_data_signal)
def broadcast_agent_data(sender, **kwargs):
    data = kwargs['data']
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        'brain',
        {
            'type': 'send_brain_data',
            'data': data
        }
    )
