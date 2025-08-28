import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

class BrainConsumer(AsyncWebsocketConsumer):
    """
    This consumer handles WebSocket connections for broadcasting
    real-time training metrics and other agent-related events.
    """
    async def connect(self):
        """
        Called when the websocket is trying to connect.
        """
        self.room_group_name = 'brain_monitoring'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()
        logger.info(f"BrainConsumer connected to group '{self.room_group_name}'")

    async def disconnect(self, close_code):
        """
        Called when the WebSocket closes for any reason.
        """
        logger.info(f"BrainConsumer disconnecting with code: {close_code}")
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Note: This consumer does not need a `receive` method as it's only for broadcasting
    # from the server to the client. Client-side actions should be handled via HTTP requests.

    async def training_update(self, event):
        """
        Handler for messages sent to the 'brain_monitoring' group.
        It forwards the message directly to the WebSocket client.
        """
        message_data = event.get('data', {})

        # Send message to WebSocket
        await self.send(text_data=json.dumps(message_data))
        logger.debug(f"Sent training_update to client: {message_data}")
