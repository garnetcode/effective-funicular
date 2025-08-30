import json
import logging
import asyncio
import redis.asyncio as aioredis
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

class BrainConsumer(AsyncWebsocketConsumer):
    """
    This consumer handles WebSocket connections for providing real-time
    agent state by polling a Redis cache.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis = None
        self.poll_task = None
        self.connected = False
        self.last_known_states = {}
        # Map Redis keys to the message types the frontend expects
        self.key_to_message_type = {
            "chimera_environments": "environments_update",
            "chimera_graph_state": "graph_update",
            "chimera_training_metrics": "training_metrics",
            "chimera_episode_metrics": "training_metrics", # Both use the same handler
            "chimera_action_update": "action_update",
            "chimera_actor_state": "actor_state_update"
        }
        self.redis_keys_to_poll = list(self.key_to_message_type.keys())

    async def connect(self):
        """Called when the websocket is trying to connect."""
        await self.accept()
        logger.info("BrainConsumer connected.")
        self.connected = True
        try:
            self.redis = aioredis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
            await self.redis.ping()
            logger.info("Consumer successfully connected to Redis.")
            # Start the polling task
            self.poll_task = asyncio.create_task(self.poll_redis_for_updates())
        except Exception as e:
            logger.error(f"Consumer could not connect to Redis: {e}. Closing connection.")
            await self.close()

    async def disconnect(self, close_code):
        """Called when the WebSocket closes."""
        logger.info(f"BrainConsumer disconnecting with code: {close_code}")
        self.connected = False
        if self.poll_task:
            self.poll_task.cancel()
        if self.redis:
            await self.redis.close()

    async def poll_redis_for_updates(self):
        """Periodically polls Redis for changes and sends updates to the client."""
        while self.connected:
            try:
                for key in self.redis_keys_to_poll:
                    current_state_json = await self.redis.get(key)
                    if current_state_json:
                        # Check if the state has changed since the last time we sent it
                        if self.last_known_states.get(key) != current_state_json:
                            self.last_known_states[key] = current_state_json
                            # Use the mapping to get the correct message type
                            message_type = self.key_to_message_type.get(key)
                            if message_type:
                                message = {
                                    "type": message_type,
                                    "payload": json.loads(current_state_json)
                                }
                                await self.send(text_data=json.dumps(message))

                # Wait for the next poll interval
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                logger.info("Redis polling task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error during Redis polling: {e}")
                # Wait before retrying to avoid spamming errors
                await asyncio.sleep(5.0)
