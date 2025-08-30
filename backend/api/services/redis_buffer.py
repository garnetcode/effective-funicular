import redis
import pickle
import numpy as np

class RedisBuffer:
    """
    A replay buffer that uses a Redis list to store experiences, allowing
    decoupled communication between actor and learner processes.
    """
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, list_key='replay_buffer', max_size=50000):
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)
        self.list_key = list_key
        self.max_size = max_size

    def push(self, experience):
        """
        Pushes a single experience to the Redis buffer.
        """
        try:
            serialized_experience = pickle.dumps(experience)
            self.redis_client.rpush(self.list_key, serialized_experience)
            # Trim the list to enforce the max_size limit
            self.redis_client.ltrim(self.list_key, -self.max_size, -1)
        except Exception as e:
            print(f"Error pushing experience to Redis: {e}")

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the Redis buffer efficiently.
        """
        try:
            buffer_size = self.redis_client.llen(self.list_key)
            if buffer_size < batch_size:
                return []

            # Generate random indices to sample
            indices = np.random.choice(buffer_size, batch_size, replace=False)

            # Use a pipeline to efficiently retrieve multiple elements
            pipe = self.redis_client.pipeline()
            for index in indices:
                # Convert numpy int64 to python int
                pipe.lindex(self.list_key, int(index))

            serialized_experiences = pipe.execute()

            # Deserialize the experiences
            batch = [pickle.loads(exp) for exp in serialized_experiences if exp is not None]

            return batch
        except Exception as e:
            print(f"Error sampling from Redis: {e}")
            return []

    def __len__(self):
        """
        Returns the current number of experiences in the buffer.
        """
        try:
            return self.redis_client.llen(self.list_key)
        except Exception as e:
            print(f"Error getting Redis buffer length: {e}")
            return 0

    def clear(self):
        """
        Deletes the replay buffer list from Redis.
        """
        try:
            self.redis_client.delete(self.list_key)
            print(f"Cleared Redis buffer with key: {self.list_key}")
        except Exception as e:
            print(f"Error clearing Redis buffer: {e}")
