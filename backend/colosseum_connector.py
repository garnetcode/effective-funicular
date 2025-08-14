import requests
import json

class ColosseumConnector:
    """
    Manages the HTTP communication with the Colosseum game server API.
    """
    def __init__(self, base_url="http://localhost:8000/api"):
        self.base_url = base_url

    def _post(self, endpoint, data=None):
        """Helper method for POST requests."""
        try:
            response = requests.post(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def _get(self, endpoint):
        """Helper method for GET requests."""
        try:
            response = requests.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def join_session(self, agent_tag, agent_name, environment_id, agent_type="ai"):
        """Agent joins a session."""
        data = {
            "agent_tag": agent_tag,
            "agent_name": agent_name,
            "environment_id": environment_id,
            "agent_type": agent_type
        }
        return self._post("/agent/join/", data)

    def take_action(self, session_id, agent_tag, action):
        """Agent takes an action in the current session."""
        data = {
            "session_id": session_id,
            "agent_tag": agent_tag,
            "action": action
        }
        return self._post("/agent/action/", data)

    def leave_session(self, session_id, agent_tag):
        """Agent leaves the current session."""
        data = {
            "session_id": session_id,
            "agent_tag": agent_tag
        }
        return self._post("/agent/leave/", data)

    def list_sessions(self):
        """Lists all active game sessions."""
        return self._get("/sessions/")

    def get_session(self, session_id):
        """Get detailed information about a specific session."""
        return self._get(f"/sessions/{session_id}/")

    def list_environments(self):
        """List all available environments."""
        return self._get("/environments/")

if __name__ == '__main__':
    # Example usage based on the provided API specification
    connector = ColosseumConnector()

    # Step 1: Join Environment
    join_data = connector.join_session(
        agent_tag='my-agent-v1',
        agent_name='My Deep Q Agent',
        environment_id='cartpole-v1'
    )

    if join_data and join_data.get("success"):
        session_id = join_data['session_id']
        observation = join_data['observation']
        print(f"Joined session {session_id} with initial observation: {observation}")

        # Simple agent logic: take action 0 until done
        is_done = False
        total_reward = 0
        while not is_done:
            action_data = connector.take_action(session_id, 'my-agent-v1', 0) # Example action: 0
            if action_data and action_data.get("success"):
                observation = action_data['observation']
                reward = action_data['reward']
                is_done = action_data['is_done']
                total_reward += reward
                print(f"Step: {action_data['step']}, Reward: {reward}, Done: {is_done}")
            else:
                print("Failed to take action.")
                break

        # Step 3: Leave Session
        leave_data = connector.leave_session(session_id, 'my-agent-v1')
        if leave_data and leave_data.get("success"):
            print(f"Left session. Final reward: {leave_data['final_reward']}")
        else:
            print("Failed to leave session.")
    else:
        print("Failed to join session.")

    # Other example usages:
    print("\nListing active sessions:")
    sessions = connector.list_sessions()
    if sessions:
        print(json.dumps(sessions, indent=2))

    print("\nListing available environments:")
    environments = connector.list_environments()
    if environments:
        print(json.dumps(environments, indent=2))
