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
                    # Try to parse JSON, but handle cases where the error response isn't JSON
                    try:
                        data = await response.json()
                    except json.JSONDecodeError:
                        data = await response.text()

                    if response.status >= 400:
                        logger.error(f"HTTP error creating session. Status: {response.status}. Response: {data}")
                        return None

                    if data.get("success"):
                        self.session_id = data.get("session_id")
                        logger.debug(f"Successfully created session: {self.session_id}")
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
            logger.debug(f"Successfully connected to WebSocket: {uri}")
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
        """Sends a raw JSON message to the server, with reconnection logic."""
        if not self.websocket or not self.websocket.open:
            logger.warning(f"WebSocket closed or not available when trying to send {message.get('type')}. Reconnecting.")
            if not await self.reconnect():
                logger.error(f"Cannot send message of type {message.get('type')}, reconnection failed.")
                return

        message_str = json.dumps(message)
        try:
            await self.websocket.send(message_str)
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed on send_message for type {message.get('type')}. Reconnecting.")
            if await self.reconnect():
                logger.info("Reconnected successfully. Retrying send_message.")
                await self.websocket.send(message_str)
            else:
                logger.error(f"Cannot send message of type {message.get('type')}, reconnection failed.")
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
                logger.debug(f"Agent {self.agent_tag} successfully joined session {self.session_id}")
                return response
            else:
                error_detail = response.get('message', 'No details provided') if response else "No response from server"
                logger.error(f"Failed to join session. Server response: {error_detail}")
                return None
        except Exception as e:
            logger.error(f"An error occurred while trying to join the session: {e}", exc_info=True)
            return None

    async def send_action(self, action):
        """Sends an agent action to the server, with reconnection logic."""
        if not self.websocket:
            logger.error("Cannot send action, WebSocket is not connected.")
            return

        action_message_str = json.dumps({
            "type": "agent.action",
            "action": int(action),
            "agent_tag": self.agent_tag
        })

        try:
            await self.websocket.send(action_message_str)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed on send_action. Attempting to reconnect...")
            if await self.reconnect():
                logger.info("Reconnected successfully. Retrying send_action.")
                await self.websocket.send(action_message_str) # Retry once after reconnect
            else:
                logger.error("Cannot send action, reconnection failed.")

    async def reset_environment(self):
        """
        Sends a reset message and waits for confirmation, ignoring any
        other messages that may be in the buffer.
        """
        logger.debug("Sending agent.reset message.")
        reset_message = {"type": "agent.reset"}
        await self.send_message(reset_message)

        # Wait for the 'environment.reset' confirmation, ignoring other messages.
        start_time = asyncio.get_event_loop().time()
        timeout = 30.0  # Increased timeout for slower environments
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                # Use a shorter timeout for each recv() call within the main loop
                response = await asyncio.wait_for(self.receive_message(), timeout=1.0)

                if response and response.get("type") == "environment.reset":
                    logger.debug("Environment reset successfully.")
                    return response
                elif response:
                    # Log other messages received while waiting
                    logger.debug(f"Ignoring buffered message of type '{response.get('type')}' while waiting for reset confirmation. Content: {response}")
                # If response is None (due to connection closed), the loop will continue and eventually time out.

            except asyncio.TimeoutError:
                # This is a timeout for a single receive, not the whole operation.
                # It's okay, we just continue waiting for the confirmation.
                logger.debug("Timeout waiting for a message, will try again...")
                continue
            except Exception as e:
                logger.error(f"An unexpected error occurred while waiting for reset confirmation: {e}")
                return None # Exit on other errors

        # If the while loop finishes, it means the main timeout was exceeded
        logger.error("Failed to reset environment: Did not receive 'environment.reset' confirmation within the total timeout.")
        return None

    async def receive_message(self):
        """
        Receives, parses, and returns a single message from the WebSocket,
        with reconnection logic. It ignores broadcasts of the agent's own actions.
        """
        while True:  # Loop to ignore self-action broadcasts and find a relevant message
            message_data = await self._receive_one_message_with_retry()
            if message_data is None:
                return None  # Return None if receiving fails permanently

            # Check if the message is a reflection of our own action
            is_own_action = (
                message_data.get("type") == "action.taken" and
                message_data.get("agent_tag") == self.agent_tag
            )

            if is_own_action:
                # This is a broadcast of our own action. We process it because it contains
                # the environment's response (obs, reward, done). In other protocols,
                # we might ignore this and wait for a direct response.
                return message_data
            else:
                # If it's not our action, it's a relevant message.
                return message_data

    async def _receive_one_message_with_retry(self):
        """Helper that attempts to receive a single message, with one retry after reconnecting."""
        if not self.websocket:
            logger.error("Cannot receive message, WebSocket is not connected.")
            if not await self.reconnect():
                return None # If reconnect fails, give up

        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=60.0)
            logger.debug(f"Raw message received: {message}")
            return json.loads(message)
        except asyncio.TimeoutError:
            logger.error("Timeout receiving message. The server may be unresponsive.")
            # Attempt a reconnect, as the connection might be stale
            asyncio.create_task(self.reconnect())
            return None # Return None to indicate failure
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed on receive. Attempting to reconnect...")
            if await self.reconnect():
                logger.info("Reconnected successfully. Retrying receive.")
                try:
                    message = await self.websocket.recv()
                    return json.loads(message)
                except websockets.exceptions.ConnectionClosed:
                    logger.error("Connection closed again immediately after reconnect.")
                    return None
            else:
                logger.error("Cannot receive message, reconnection failed.")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from message: {e}. Attempting to re-parse for double-encoding.")
            try:
                # This handles cases where the message is a JSON string that itself contains a JSON string,
                # e.g., '"{\\"type\\": \\"game.over\\"}"'
                unwrapped_message = json.loads(message)
                if isinstance(unwrapped_message, str):
                    return json.loads(unwrapped_message)
                else:
                    # If the unwrapped message isn't a string, we can't parse it further.
                    logger.error("Message unwrapped to a non-string type, cannot re-parse.")
                    return None
            except Exception as inner_e:
                logger.error(f"Failed to re-parse double-encoded message: {inner_e}")
                return None

    async def reconnect(self, max_retries=3, delay=1):
        """
        Attempts to close the existing connection and re-establish a new one.
        Includes retries with exponential backoff.
        """
        logger.info("Connection lost. Attempting to reconnect...")
        await self.close()  # Close the old connection gracefully

        for i in range(max_retries):
            logger.info(f"Reconnection attempt {i + 1}/{max_retries}...")
            try:
                if await self.connect_websocket():
                    logger.info("WebSocket reconnected. Attempting to re-join session...")
                    # Re-joining is crucial for the server to recognize the new connection.
                    if await self.join_session():
                        logger.info("Successfully re-joined session.")
                        return True
                    else:
                        logger.warning("Reconnected to WebSocket, but failed to re-join session.")
                        # Close the new connection as it's not usable without joining
                        await self.close()

                # If connect_websocket or join_session fails, wait before retrying
                sleep_time = delay * (2 ** i)
                logger.info(f"Reconnection attempt failed. Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"An unexpected error occurred during reconnection attempt {i + 1}: {e}", exc_info=True)
                sleep_time = delay * (2 ** i)
                logger.info(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

        logger.error("Failed to reconnect after multiple attempts.")
        return False

    async def close(self):
        """Closes the WebSocket connection."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.debug("WebSocket connection closed.")
            except websockets.exceptions.ConnectionClosed:
                logger.debug("WebSocket connection was already closed.")
            except Exception as e:
                logger.error(f"Error while closing WebSocket: {e}", exc_info=True)
        self.websocket = None
