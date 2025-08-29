import unittest
from unittest.mock import patch, MagicMock
import sys

# We need to add the backend directory to the path to import train
sys.path.append('backend')
import train

class TrainScriptTests(unittest.TestCase):

    @patch('train.main')
    @patch('train.argparse.ArgumentParser')
    def test_curriculum_argument_parsing(self, mock_arg_parser, mock_main):
        """
        Tests that the --env-curriculum argument is parsed correctly into a list.
        """
        # --- Setup Mocks ---
        # Mock the return value of parse_args()
        mock_args = MagicMock()
        mock_args.env_curriculum = "CartPole-v1,MountainCar-v0"
        mock_args.agent_id = None
        mock_args.total_steps = 100
        mock_args.steps_per_env = 50
        mock_args.batch_size = 10
        mock_args.lr = 0.001
        mock_args.gamma = 0.99
        mock_args.force_new = False
        mock_args.no_stag = False

        mock_parser_instance = mock_arg_parser.return_value
        mock_parser_instance.parse_args.return_value = mock_args

        # --- Call the function that contains the parsing logic ---
        train.parse_args_and_run()

        # --- Assertions ---
        # Check that our patched main was called with the mocked args.
        mock_main.assert_called_once_with(mock_args)

if __name__ == '__main__':
    unittest.main()
