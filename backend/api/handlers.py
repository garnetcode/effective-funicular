from django.dispatch import receiver
from channels.layers import get_channel_layer
from .signals import agent_data_signal
import json
import asyncio

@receiver(agent_data_signal)
def broadcast_agent_data(sender, **kwargs):
    """
    Handles the agent_data_signal.
    Since the sender (`train_agent` command) is running in an async context,
    we can't use async_to_sync. We need to get the running event loop
    and schedule our async task on it.
    """
    data = kwargs['data']
    channel_layer = get_channel_layer()

    async def send_to_group():
        await channel_layer.group_send(
            'brain',
            {
                'type': 'send_brain_data',
                'data': data
            }
        )

    # Get the currently running event loop and run the coroutine
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_to_group())
    except RuntimeError:
        # If no event loop is running, we can fall back to async_to_sync
        # This makes the handler more robust for different contexts.
        from asgiref.sync import async_to_sync
        async_to_sync(send_to_group)()
