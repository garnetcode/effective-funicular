import asyncio
import websockets
import json

class ColosseumConnector:
    """
    Manages the WebSocket connection and communication with the Colosseum game server.
    """
    def __init__(self, game_name, token, host="127.0.0.1", port=8765):
        self.uri = f"ws://{host}:{port}/ws/game/{game_name}/?token={token}"
        self.websocket = None

    async def connect(self):
        """Connects to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            print(f"Successfully connected to Colosseum server at {self.uri}")
        except Exception as e:
            print(f"Failed to connect to Colosseum server: {e}")
            raise

    async def close(self):
        """Closes the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            print("Connection to Colosseum server closed.")

    async def send_action(self, action):
        """
        Sends an action to the server.

        Args:
            action: The action to be taken by the agent.
        """
        message = {
            "type": "action",
            "action": action
        }
        await self.websocket.send(json.dumps(message))

    async def receive_message(self):
        """
        Waits for and returns the next message from the server.
        """
        try:
            message = await self.websocket.recv()
            return json.loads(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed by server: {e}")
            return None
