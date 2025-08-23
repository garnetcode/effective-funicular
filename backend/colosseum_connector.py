import asyncio
import json
import aiohttp
import websockets
import logging

logger = logging.getLogger(__name__)

class ColosseumConnector:
    """
    Manages communication with a Colosseum game server for a single session,
    handling both HTTP for session management and WebSockets for real-time gameplay.
    """
    def __init__(self, environment_id, agent_tag, host="127.0.0.1", http_port=8000, ws_port=8000):
        self.http_base_url = f"http://{host}:{http_port}/api"
        self.ws_base_url = f"ws://{host}:{ws_port}/ws"
        self.environment_id = environment_id
        self.agent_tag = agent_tag
        self.session_id = None
        self.websocket = None

    async def create_session(self):
        """Creates a new game session via the HTTP API."""
        url = f"{self.http_base_url}/sessions/create/"
        payload = {
            "environment_id": self.environment_id,
            "agent_tag": self.agent_tag,
            "agent_name": f"ChimeraAgent-{self.agent_tag}",
            "agent_type": "ai"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data.get("success"):
                        self.session_id = data.get("session_id")
                        logger.info(f"Successfully created session: {self.session_id}")
                        return data
                    else:
                        logger.error(f"Failed to create session: {data}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error creating session: {e}")
            return None

    async def connect_websocket(self):
        """Connects to the session's WebSocket endpoint."""
        if not self.session_id:
            logger.error("Cannot connect WebSocket without a session ID.")
            return False

        # Sanitize the session ID for the WebSocket URL by removing hyphens
        sanitized_session_id = self.session_id.replace('-', '')
        uri = f"{self.ws_base_url}/session/{sanitized_session_id}/"
        try:
            self.websocket = await websockets.connect(uri)
            logger.info(f"Successfully connected to WebSocket: {uri}")
            return True
        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid WebSocket URI: {uri}")
            return False
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed unexpectedly: {e}")
            return False

    async def join_session(self):
        """Sends the agent.join message to formally join the session."""
        if not self.websocket:
            return None

        join_message = {
            "type": "agent.join",
            "agent_tag": self.agent_tag,
            "environment_id": self.environment_id,
            "session_id": self.session_id
        }
        await self.websocket.send(json.dumps(join_message))
        response = await self.receive_message()
        if response and response.get("type") == "agent.joined":
            logger.info(f"Agent {self.agent_tag} successfully joined session {self.session_id}")
            return response
        else:
            logger.error(f"Failed to join session, response: {response}")
            return None

    async def send_action(self, action):
        """Sends an agent action to the server."""
        if not self.websocket:
            return

        action_message = {
            "type": "agent.action",
            "action": int(action),
            "session_id": self.session_id
        }
        await self.websocket.send(json.dumps(action_message))

    async def receive_message(self):
        """Receives and parses a single message from the WebSocket."""
        if not self.websocket:
            return None
        try:
            message = await self.websocket.recv()
            return json.loads(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed while waiting for message.")
            return None

    async def close(self):
        """Closes the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed.")
