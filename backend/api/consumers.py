import json
from channels.generic.websocket import AsyncWebsocketConsumer

class TrainingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.agent_id = self.scope["url_route"]["kwargs"]["agent_id"]
        self.room_group_name = f"training_{self.agent_id}"

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        print(f"WebSocket connected for agent: {self.agent_id}")

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        print(f"WebSocket disconnected for agent: {self.agent_id}")


    # Receive message from WebSocket (e.g., if UI wants to send commands)
    async def receive(self, text_data):
        # For now, we just log it. This could be used for pause/resume commands.
        print(f"Received message from client for {self.agent_id}: {text_data}")

    # Receive message from room group (from the training loop)
    async def training_message(self, event):
        message = event["message"]

        # Send message to WebSocket
        await self.send(text_data=json.dumps(message))
        print(f"Sent message to client for {self.agent_id}: {message}")
