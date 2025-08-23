import asyncio
import json
import uuid
import websockets
import logging

logger = logging.getLogger(__name__)

class ColosseumConnector:
    """
    Manages communication with a Colosseum game server for a single session,
    handling WebSocket connection and gameplay.
    """
    def __init__(self, environment_id, agent_tag, host="127.0.0.1", http_port=8000, ws_port=8000):
        # http_port is no longer used but kept for compatibility with existing configs
        self.ws_base_url = f"ws://{host}:{ws_port}/ws"
        self.environment_id = environment_id
        self.agent_tag = agent_tag
        self.session_id = None
        self.websocket = None

    async def connect(self):
        """
        Establishes a connection to the Colosseum server by generating a new
        session ID, creating a WebSocket connection, and joining the session.
        Returns True on success, False on failure.
        """
        # 1. Generate a session ID on the client-side
        self.session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {self.session_id}")

        # 2. Connect to the WebSocket endpoint
        uri = f"{self.ws_base_url}/session/{self.session_id}/"
        try:
            # The `create_protocol` argument is deprecated. Instead, pass custom
            # headers using `extra_headers`.
            self.websocket = await websockets.connect(
                uri,
                extra_headers={"Origin": "http://localhost:3000"}
            )
            logger.info(f"Successfully connected to WebSocket: {uri}")
        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid WebSocket URI: {uri}")
            return None
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed unexpectedly: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during WebSocket connection: {e}", exc_info=True)
            return None

        # 3. Join the session by sending an 'agent.join' message
        try:
            join_message = {
                "type": "agent.join",
                "agent_tag": self.agent_tag,
                "environment_id": self.environment_id,
            }
            await self.websocket.send(json.dumps(join_message))
            response = await self.receive_message()

            if response and response.get("type") == "agent.joined":
                logger.info(f"Agent {self.agent_tag} successfully joined session {self.session_id}")
                return response
            else:
                error_detail = response.get('message', 'No details provided') if response else "No response from server"
                logger.error(f"Failed to join session. Server response: {error_detail}")
                await self.close()
                return None
        except Exception as e:
            logger.error(f"An error occurred while trying to join the session: {e}", exc_info=True)
            await self.close()
            return None

    async def send_action(self, action):
        """Sends an agent action to the server."""
        if not self.websocket or self.websocket.closed:
            logger.error("Cannot send action, WebSocket is not connected.")
            return

        try:
            action_message = {
                "type": "agent.action",
                "action": int(action),
                "agent_tag": self.agent_tag
            }
            await self.websocket.send(json.dumps(action_message))
        except Exception as e:
            logger.error(f"Failed to send action: {e}", exc_info=True)


    async def receive_message(self):
        """Receives and parses a single message from the WebSocket."""
        if not self.websocket or self.websocket.closed:
            logger.warning("Cannot receive message, WebSocket is not connected.")
            return None
        try:
            message = await self.websocket.recv()
            return json.loads(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed while waiting for message.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from message: {e}")
            return None

    async def close(self):
        """Closes the WebSocket connection."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            logger.info("WebSocket connection closed.")
        self.websocket = None
