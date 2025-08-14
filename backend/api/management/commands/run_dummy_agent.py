import time
import random
from django.core.management.base import BaseCommand
from colosseum_connector import ColosseumConnector

class Command(BaseCommand):
    help = 'Runs a dummy agent to interact with the Colosseum API.'

    def add_arguments(self, parser):
        parser.add_argument('--agent-tag', type=str, default='dummy-agent-1', help='Tag for the agent.')
        parser.add_argument('--environment', type=str, default='cartpole-v1', help='Environment to run the agent in.')
        parser.add_argument('--base-url', type=str, default='http://localhost:8000/api', help='Base URL of the Colosseum API.')

    def handle(self, *args, **options):
        agent_tag = options['agent_tag']
        environment = options['environment']
        base_url = options['base_url']

        self.stdout.write(self.style.SUCCESS(f"Starting dummy agent '{agent_tag}' for environment '{environment}'"))

        connector = ColosseumConnector(base_url=base_url)

        # 1. Join a session
        join_data = connector.join_session(
            agent_tag=agent_tag,
            agent_name=f"Dummy Agent {agent_tag}",
            environment_id=environment
        )

        if not join_data or not join_data.get('success'):
            self.stdout.write(self.style.ERROR('Failed to join session.'))
            return

        session_id = join_data['session_id']
        self.stdout.write(self.style.SUCCESS(f"Successfully joined session: {session_id}"))

        # 2. Play the game
        total_reward = 0
        step_count = 0
        is_done = False

        while not is_done:
            # Determine a random action
            if "cartpole" in environment.lower():
                action = random.randint(0, 1)
            elif "mountaincar" in environment.lower():
                action = random.randint(0, 2)
            elif "acrobot" in environment.lower():
                action = random.randint(0, 2)
            elif "lunarlander" in environment.lower():
                action = random.randint(0, 3)
            else:
                action = random.randint(0, 1) # Default action space

            action_data = connector.take_action(session_id, agent_tag, action)

            if not action_data or not action_data.get('success'):
                self.stdout.write(self.style.ERROR('Failed to take action.'))
                break

            reward = action_data.get('reward', 0)
            is_done = action_data.get('is_done', False)
            total_reward += reward
            step_count += 1

            self.stdout.write(f"Step {step_count}: Action={action}, Reward={reward}, Total Reward={total_reward:.2f}, Done={is_done}")

            # Optional: add a small delay to make the simulation observable
            time.sleep(0.05)

        self.stdout.write(self.style.SUCCESS(f"Episode finished after {step_count} steps with a total reward of {total_reward:.2f}."))

        # 3. Leave the session
        leave_data = connector.leave_session(session_id, agent_tag)
        if leave_data and leave_data.get('success'):
            self.stdout.write(self.style.SUCCESS('Successfully left the session.'))
        else:
            self.stdout.write(self.style.ERROR('Failed to leave the session.'))
