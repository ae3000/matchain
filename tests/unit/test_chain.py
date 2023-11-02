import sys
from unittest.mock import MagicMock, patch

import matchain.base
import matchain.chain
import matchain.config
from tests.utils_for_tests import DefaultDataPaths, TestBase


class TestChain(TestBase):

    def test_main(self):
        file_config = DefaultDataPaths.get_file_config_chains()
        args = ['any_program_name', '--config', file_config]
        board = matchain.base.PinBoard()
        mock = MagicMock(return_value=board)

        with patch.object(sys, 'argv', args):
            with patch.object(matchain.chain, 'run', mock):
                matchain.chain.main()

        self.assertEqual(mock.call_count, 7)

    def test_run(self):
        config = self.get_config('dg')
        exp_commands = config['chain']
        mock = MagicMock()

        with patch.object(matchain.chain, 'execute_command', mock):
            matchain.chain.run(config)

        act_commands = []
        for args in mock.call_args_list:
            pos_command_arg = 0
            command = args[0][pos_command_arg]
            act_commands.append(command)

        self.assertListEqual(act_commands, exp_commands)

    def test_execute_command_with_unknown_command(self):
        board = matchain.base.PinBoard()
        board.config = self.get_config('dg')
        self.assertRaises(RuntimeError, matchain.chain.execute_command,
                          'unknown', board)
