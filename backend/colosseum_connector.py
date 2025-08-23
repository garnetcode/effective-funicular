import asyncio
import json
import websockets
import logging
import aiohttp

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

        uri = f"{self.ws_base_url}/session/{self.session_id}/"
        try:
            self.websocket = await websockets.connect(
                uri,
                additional_headers={"Origin": "http://localhost:3000"}
            )
            logger.info(f"Successfully connected to WebSocket: {uri}")
            return True
        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid WebSocket URI: {uri}")
            return False
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed unexpectedly: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during WebSocket connection: {e}", exc_info=True)
            return False

    async def send_message(self, message):
        """Sends a raw JSON message to the server."""
        if not self.websocket:
            logger.error("Cannot send message, WebSocket is not connected.")
            return
        try:
            await self.websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Cannot send message, connection is closed.")
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)

    async def join_session(self):
        """Sends the agent.join message to formally join the session."""
        if not self.websocket:
            logger.error("Cannot join session, WebSocket is not connected.")
            return None
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
                return None
        except Exception as e:
            logger.error(f"An error occurred while trying to join the session: {e}", exc_info=True)
            return None

    async def send_action(self, action):
        """Sends an agent action to the server."""
        if not self.websocket:
            logger.error("Cannot send action, WebSocket is not connected.")
            return
        try:
            action_message = {
                "type": "agent.action",
                "action": int(action),
                "agent_tag": self.agent_tag
            }
            await self.websocket.send(json.dumps(action_message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Cannot send action, connection is closed.")
        except Exception as e:
            logger.error(f"Failed to send action: {e}", exc_info=True)

    async def reset_environment(self):
        """
        Sends a reset message and waits for confirmation. It assumes the
        message buffer has been drained by the caller.
        """
        logger.info("Sending agent.reset message.")
        reset_message = {"type": "agent.reset"}
        await self.send_message(reset_message)

        # Now, wait for the 'environment.reset' confirmation with a reasonable timeout.
        try:
            response = await asyncio.wait_for(self.receive_message(), timeout=10.0) # 10-second timeout
            if response and response.get("type") == "environment.reset":
                logger.info("Environment reset successfully.")
                return response
            else:
                logger.error(f"Failed to reset environment: received unexpected message {response}")
                return None
        except asyncio.TimeoutError:
            logger.error("Failed to reset environment: Did not receive 'environment.reset' confirmation within timeout.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while waiting for reset confirmation: {e}")
            return None

    async def receive_message(self):
        """
        Receives, parses, and returns a single message from the WebSocket,
        ignoring broadcast messages of the agent's own actions.
        """
        if not self.websocket:
            logger.warning("Cannot receive message, WebSocket is not connected.")
            return None

        while True: # Loop until a relevant message is received
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                # Check if the message is a reflection of our own action
                is_own_action = (
                    data.get("type") == "action.taken" and
                    data.get("agent_tag") == self.agent_tag
                )

                if is_own_action:
                    # In a more complex scenario, the server might send a direct confirmation
                    # in addition to a broadcast. For now, we assume the broadcast is the
                    # only response, so we must process it. If the protocol changes,
                    # this is where we would 'continue' to ignore it.
                    # logger.info(f"Received own action broadcast: {data}")
                    pass # For now, we still need this message for obs, reward, done.

                return data # Return the first message received

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed while waiting for message.")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from message: {e}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred in receive_message: {e}")
                return None

    async def close(self):
        """Closes the WebSocket connection."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket connection closed.")
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection was already closed.")
            except Exception as e:
                logger.error(f"Error while closing WebSocket: {e}", exc_info=True)
        self.websocket = None
