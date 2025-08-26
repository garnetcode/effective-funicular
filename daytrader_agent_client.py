import asyncio
import websockets
import json
import uuid
import random

class DayTraderAgent:
    def __init__(self, agent_tag="jules-trader-v2"):
        self.agent_tag = agent_tag
        self.session_id = str(uuid.uuid4())
        self.unified_uri = f"ws://localhost:8000/ws/unified/{self.session_id}/"
        self.data_uri = "ws://localhost:8000/ws/day_trader_data/"
        self.action_ws = None
        self.data_ws = None
        self.trade_is_open = False

    async def connect(self):
        """Establishes WebSocket connections."""
        try:
            self.action_ws = await websockets.connect(self.unified_uri)
            print(f"Connected to action server at {self.unified_uri}")
            self.data_ws = await websockets.connect(self.data_uri)
            print(f"Connected to data server at {self.data_uri}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    async def join_session(self):
        """Joins the Day Trader session."""
        join_payload = {
            "type": "agent.join",
            "agent_tag": self.agent_tag,
            "environment_id": "DayTrader-v1"
        }
        await self.action_ws.send(json.dumps(join_payload))
        print(f"Sent agent.join for session {self.session_id}")
        # Wait for the confirmation message from the action server
        response = await self.action_ws.recv()
        print(f"Join response: {response}")

    async def get_observation(self):
        """Requests a new observation from the data server."""
        payload = {
            "type": "get_observation",
            "session_id": self.session_id
        }
        await self.data_ws.send(json.dumps(payload))
        print("Requested new observation.")

    async def send_action(self, action):
        """Sends a trading action to the action server."""
        payload = {
            "type": "agent.action",
            "action": action
        }
        await self.action_ws.send(json.dumps(payload))
        print(f"Sent action: {action}")
        # The action response will come back on this socket
        response = await self.action_ws.recv()
        print(f"Action response: {response}")


    def make_decision(self, observation):
        """Makes a simple trading decision based on the observation."""
        # This is a placeholder for a real trading strategy.
        # For now, it will randomly buy or sell.
        # Action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE

        if self.trade_is_open:
            # If a trade is open, we can't do anything else until it's closed.
            print("Decision: HOLD (waiting for trade to close)")
            return 0 # HOLD

        # A simple random strategy
        decision = random.choice([1, 2]) # 1=BUY, 2=SELL
        print(f"Decision: {'BUY' if decision == 1 else 'SELL'}")
        return decision

    async def trading_loop(self):
        """The main trading loop."""
        # 1. Initial observation
        await self.get_observation()

        # 2. Main loop
        while True:
            try:
                # Listen for messages from the data server
                message_str = await self.data_ws.recv()
                message = json.loads(message_str)
                msg_type = message.get("type")

                print(f"\n--- Received message from data server: {msg_type} ---")
                # print(f"Full message: {message}")

                if msg_type == "observation_update":
                    if self.trade_is_open:
                        print("Received observation while trade is open. Ignoring.")
                        continue

                    observation = message.get("observation")
                    action = self.make_decision(observation)

                    if action in [1, 2]: # BUY or SELL
                        self.trade_is_open = True
                        await self.send_action(action)
                    else: # HOLD
                        # If we decide to hold, wait a bit then get a new observation
                        await asyncio.sleep(5)
                        await self.get_observation()

                elif msg_type == "trade.closed":
                    print("Trade closed! Getting new observation to make next decision.")
                    self.trade_is_open = False
                    await asyncio.sleep(1) # Small delay before requesting new observation
                    await self.get_observation()

                elif msg_type == "episode.terminated":
                    print("Episode terminated. Ending trading session.")
                    break

                elif msg_type == "error":
                    print(f"Error from server: {message.get('message')}")
                    if "Environment not found" in message.get('message'):
                         print("This might be because the server-side bug has not been fixed. See instructions.")
                    break

            except websockets.exceptions.ConnectionClosed:
                print("Connection closed.")
                break
            except Exception as e:
                print(f"An error occurred in the trading loop: {e}")
                break

    async def run(self):
        """Connects and runs the trading loop."""
        if await self.connect():
            await self.join_session()
            await self.trading_loop()
            await self.action_ws.close()
            await self.data_ws.close()

if __name__ == "__main__":
    agent = DayTraderAgent()
    asyncio.run(agent.run())
