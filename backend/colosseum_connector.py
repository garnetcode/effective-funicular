import asyncio
import json
import websockets
import logging
import aiohttp
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColosseumConnector:
    """
    Manages communication with a Colosseum game server for a single session,
    handling both HTTP for session management and WebSockets for real-time gameplay.

    This class is designed to be used by an AI agent to connect to and play in a
    Colosseum environment.
    """
    def __init__(self, environment_id, agent_tag, host="127.0.0.1", port=8002):
        """
        Initializes the connector.
        Args:
            environment_id (str): The ID of the gymnasium environment to play in (e.g., "CartPole-v1").
            agent_tag (str): A unique identifier for the agent.
            host (str): The hostname or IP address of the Colosseum server.
            port (int): The port for both HTTP and WebSocket connections.
        """
        self.http_base_url = f"http://{host}:{port}/api"
        self.ws_base_url = f"ws://{host}:{port}/ws"
        self.environment_id = environment_id
        self.agent_tag = agent_tag
        self.session_id = None
        self.websocket = None
        self.is_reconnecting = False
        self.reconnect_attempts = 0

    async def create_session(self):
        """Creates a new game session via the HTTP API."""
        url = f"{self.http_base_url}/sessions/create/"
        payload = {
            "environment_id": self.environment_id,
            "agent_name": f"ConnectorAgent-{self.agent_tag}",
            "agent_type": "ai"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data.get("success") and data.get("session_id"):
                        self.session_id = data["session_id"]
                        logger.info(f"Successfully created session: {self.session_id}")
                        return data
                    else:
                        logger.error(f"Failed to create session. Server response: {data}")
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
            # The Origin header is sometimes needed to bypass CORS checks, especially in local dev.
            self.websocket = await websockets.connect(
                uri,
                extra_headers={"Origin": "http://localhost:3000"}
            )
            logger.info(f"Successfully connected to WebSocket: {uri}")
            self.reconnect_attempts = 0 # Reset on successful connection
            return True
        except (websockets.exceptions.InvalidURI,
                websockets.exceptions.ConnectionClosed,
                OSError) as e: # OSError can happen if the server is not running
            logger.error(f"Failed to connect to WebSocket at {uri}: {e}")
            return False

    async def join_session(self):
        """Sends the agent.join message to formally join the session and receive the initial state."""
        if not self.websocket:
            logger.error("Cannot join session, WebSocket is not connected.")
            return None
        try:
            join_message = {
                "type": "agent.join",
                "agent_tag": self.agent_tag,
                "environment_id": self.environment_id,
            }
            await self.send_message(join_message)
            response = await self.receive_message()

            if response and response.get("type") == "agent.joined":
                logger.info(f"Agent '{self.agent_tag}' successfully joined session '{self.session_id}'")
                return response
            else:
                error_detail = response.get('message', 'No details provided') if response else "No response from server"
                logger.error(f"Failed to join session. Server response: {error_detail}")
                return None
        except Exception as e:
            logger.error(f"An error occurred while trying to join the session: {e}", exc_info=True)
            return None

    async def send_action(self, action):
        """
        Sends an agent action to the server.
        The action can be an integer, float, or list/tuple, depending on the environment's action space.
        """
        action_message = {
            "type": "agent.action",
            "action": action, # No longer casting to int, allows for continuous actions
            "agent_tag": self.agent_tag
        }
        await self.send_message(action_message)

    async def receive_message(self):
        """Receives and parses a single message from the WebSocket, with reconnection logic."""
        if not self.websocket or self.websocket.closed:
            logger.warning("Cannot receive message, WebSocket is not connected or closed.")
            await self.handle_reconnect()
            return None
        try:
            message = await self.websocket.recv()
            return json.loads(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed while waiting for message. Attempting to reconnect.")
            await self.handle_reconnect()
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from message: {e}")
            return None

    async def send_message(self, message):
        """Sends a raw JSON message to the server, with reconnection logic."""
        if not self.websocket or self.websocket.closed:
            logger.error("Cannot send message, WebSocket is not connected or closed.")
            await self.handle_reconnect()
            # After attempting to reconnect, try sending again if the connection is back.
            if self.websocket and not self.websocket.closed:
                 await self.websocket.send(json.dumps(message))
            return

        try:
            await self.websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Cannot send message, connection is closed. Attempting to reconnect.")
            await self.handle_reconnect()
            if self.websocket and not self.websocket.closed:
                await self.websocket.send(json.dumps(message))

    async def reset_environment(self):
        """Resets the environment via the HTTP API and returns the new initial observation."""
        if not self.session_id:
            logger.error("Cannot reset environment without a session ID.")
            return None
        url = f"{self.http_base_url}/sessions/{self.session_id}/reset/"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data.get("success"):
                        logger.info(f"Successfully reset environment for session: {self.session_id}")
                        return data
                    else:
                        logger.error(f"Failed to reset environment. Server response: {data}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error resetting environment: {e}")
            return None

    async def handle_reconnect(self):
        """Handles WebSocket reconnection with exponential backoff."""
        if self.is_reconnecting:
            return
        self.is_reconnecting = True

        await self.close()

        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 30)
        logger.info(f"Connection lost. Attempting to reconnect in {wait_time} seconds (Attempt {self.reconnect_attempts})...")
        await asyncio.sleep(wait_time)

        try:
            if await self.connect_websocket():
                if await self.join_session():
                    logger.info("Successfully reconnected and rejoined session.")
                else:
                    logger.error("Failed to rejoin session after reconnecting.")
            else:
                logger.error("Failed to re-establish WebSocket connection.")
        finally:
            self.is_reconnecting = False

    async def close(self):
        """Closes the WebSocket connection."""
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.close()
                logger.info("WebSocket connection closed.")
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection was already closed.")
        self.websocket = None

async def run_random_agent(environment_id, agent_tag):
    """
    Example of how to use the ColosseumConnector to run an agent
    that takes random actions in an environment.
    """
    logger.info(f"Starting random agent for environment: {environment_id}")
    # Use port 8002 if you are running the backend server with `python manage.py runserver 0.0.0.0:8002`
    connector = ColosseumConnector(environment_id, agent_tag, port=8000)

    # 1. Create a session
    session_info = await connector.create_session()
    if not session_info:
        logger.error("Could not create session. Exiting.")
        return

    # 2. Connect to the WebSocket
    if not await connector.connect_websocket():
        logger.error("Could not connect to WebSocket. Exiting.")
        return

    # 3. Join the session and get the initial state
    initial_state = await connector.join_session()
    if not initial_state:
        logger.error("Could not join session. Exiting.")
        await connector.close()
        return

    action_space_shape = initial_state.get("action_space_shape")
    if not action_space_shape:
        logger.error("Could not determine action space from initial state.")
        await connector.close()
        return

    logger.info(f"Action space shape: {action_space_shape}")

    # 4. Main game loop
    try:
        while True:
            # For this example, we take a random action.
            # A real agent would use the observation to decide on an action.
            if isinstance(action_space_shape, int): # Discrete space
                action = random.randint(0, action_space_shape - 1)
            else: # Continuous space (like Pendulum) - send a random value in range [-2.0, 2.0]
                action = [random.uniform(-2.0, 2.0)]

            logger.info(f"Sending action: {action}")
            await connector.send_action(action)

            # Receive the result of the action
            state = await connector.receive_message()
            if not state:
                logger.warning("Did not receive state, might be reconnecting. Skipping loop.")
                continue

            if state.get("type") == "error":
                logger.error(f"Received error from server: {state.get('message')}")
                break

            if state.get("type") == "action.taken":
                reward = state.get("reward", 0)
                done = state.get("done", False)
                total_reward = state.get("total_reward")
                logger.info(f"Received state: Reward={reward:.2f}, Total Reward={total_reward:.2f}, Done={done}")

                if done:
                    logger.info("Game over!")
                    if state.get("terminated"):
                        logger.info(f"Agent reached the goal! Final Score: {total_reward:.2f}")
                    elif state.get("truncated"):
                         logger.info(f"Agent timed out. Final Score: {total_reward:.2f}")
                    break

            elif state.get("type") == "game.over": # Another way the game can end
                total_reward = state.get('final_reward')
                logger.info(f"Game over message received! Final Score: {total_reward:.2f}")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # 5. Clean up
        logger.info("Closing connection.")
        await connector.close()

if __name__ == "__main__":
    # --- Configuration ---
    # Change this to the environment you want to test
    # ENV_ID = "CartPole-v1"
    # ENV_ID = "MountainCar-v0"
    ENV_ID = "Pendulum-v1"
    AGENT_TAG = f"random-agent-{random.randint(1000, 9999)}"

    try:
        asyncio.run(run_random_agent(ENV_ID, AGENT_TAG))
    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError):
        logger.error("\nConnection to the server failed. Please ensure the Colosseum backend is running.")
