import json
from channels.generic.websocket import AsyncWebsocketConsumer

class BrainConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_group_name = 'brain'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        pass

    async def send_brain_data(self, event):
        brain_data = event['data']
        await self.send(text_data=json.dumps(brain_data))
