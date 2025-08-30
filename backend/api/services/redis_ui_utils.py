import redis
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

try:
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_client.ping()
    logger.info("Successfully connected to Redis for UI updates.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Could not connect to Redis for UI updates: {e}")
    redis_client = None

def update_ui_state_in_redis(key, data):
    if redis_client is None: return
    try:
        payload = json.dumps(data, cls=NumpyJSONEncoder)
        redis_client.set(key, payload)
    except Exception as e:
        logger.warning(f"Failed to update UI state in Redis for key {key}: {e}")
