"""
Tests for CLI interface
"""

import io
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call
import argparse
import os

# Add the parent directory to sys.path to import cli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import main, factor_command, benchmark_command, run_tests_command, info_command


class TestCLI(unittest.TestCase):
    """Test cases for CLI interface"""

    def setUp(self):
        """Set up test fixtures"""
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        """Clean up after tests"""
        sys.argv = self.original_argv

    def test_info_command(self):
        """Test info command output"""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            args = argparse.Namespace()
            info_command(args)
            
            output = mock_stdout.getvalue()
            self.assertIn("Prime Resonance Field (RFH3)", output)
            self.assertIn("UOR Foundation", output)
            self.assertIn("Available commands:", output)

    @patch('cli.RFH3')
    def test_factor_command_basic(self, mock_rfh3_class):
        """Test basic factor command functionality"""
        # Mock RFH3 instance
        mock_rfh3 = MagicMock()
        mock_rfh3.factor.return_value = (11, 13)
        mock_rfh3_class.return_value = mock_rfh3

        # Create test args
        args = argparse.Namespace(
            numbers=['143'],
            timeout=60.0,
            verbose=False,
            verify=False,
            strict=False,
            no_learning=False,
            no_hierarchical=False
        )

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            factor_command(args)
            
            output = mock_stdout.getvalue()
            self.assertIn("Factoring 143", output)
            self.assertIn("143 = 11 × 13", output)
            mock_rfh3.factor.assert_called_once_with(143, timeout=60.0)

    @patch('cli.RFH3')
    def test_factor_command_with_verification(self, mock_rfh3_class):
        """Test factor command with verification enabled"""
        mock_rfh3 = MagicMock()
        mock_rfh3.factor.return_value = (11, 13)
        mock_rfh3_class.return_value = mock_rfh3

        args = argparse.Namespace(
            numbers=['143'],
            timeout=60.0,
            verbose=False,
            verify=True,
            strict=False,
            no_learning=False,
            no_hierarchical=False
        )

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            factor_command(args)
            
            output = mock_stdout.getvalue()
            self.assertIn("✓ Verified", output)

    @patch('cli.RFH3')
    def test_factor_command_verification_failure(self, mock_rfh3_class):
        """Test factor command verification failure"""
        mock_rfh3 = MagicMock()
        mock_rfh3.factor.return_value = (11, 12)  # Wrong factors
        mock_rfh3_class.return_value = mock_rfh3

        args = argparse.Namespace(
            numbers=['143'],
            timeout=60.0,
            verbose=False,
            verify=True,
            strict=False,
            no_learning=False,
            no_hierarchical=False
        )

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('sys.exit') as mock_exit:
                factor_command(args)
                
                output = mock_stdout.getvalue()
                self.assertIn("❌ Verification failed!", output)
                mock_exit.assert_called_once_with(1)

    @patch('cli.RFH3')
    def test_factor_command_multiple_numbers(self, mock_rfh3_class):
        """Test factoring multiple numbers"""
        mock_rfh3 = MagicMock()
        mock_rfh3.factor.side_effect = [(11, 13), (17, 19)]
        mock_rfh3_class.return_value = mock_rfh3

        args = argparse.Namespace(
            numbers=['143', '323'],
            timeout=60.0,
            verbose=False,
            verify=False,
            strict=False,
            no_learning=False,
            no_hierarchical=False
        )

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            factor_command(args)
            
            output = mock_stdout.getvalue()
            self.assertIn("143 = 11 × 13", output)
            self.assertIn("323 = 17 × 19", output)
            self.assertIn("Summary: 2 successful factorizations", output)

    @patch('cli.RFH3')
    def test_factor_command_verbose_mode(self, mock_rfh3_class):
        """Test factor command in verbose mode"""
        mock_rfh3 = MagicMock()
        mock_rfh3.factor.return_value = (11, 13)
        mock_rfh3.print_stats = MagicMock()
        mock_rfh3_class.return_value = mock_rfh3

        args = argparse.Namespace(
            numbers=['143'],
            timeout=60.0,
            verbose=True,
            verify=False,
            strict=False,
            no_learning=False,
            no_hierarchical=False
        )

        with patch('sys.stdout', new_callable=io.StringIO):
            factor_command(args)
            
            # Should call print_stats in verbose mode
            mock_rfh3.print_stats.assert_called_once()

    @patch('cli.RFH3')
    def test_factor_command_error_handling(self, mock_rfh3_class):
        """Test factor command error handling"""
        mock_rfh3 = MagicMock()
        mock_rfh3.factor.side_effect = ValueError("Test error")
        mock_rfh3_class.return_value = mock_rfh3

        args = argparse.Namespace(
            numbers=['143'],
            timeout=60.0,
            verbose=False,
            verify=False,
            strict=False,
            no_learning=False,
            no_hierarchical=False
        )

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            factor_command(args)
            
            output = mock_stdout.getvalue()
            self.assertIn("Error factoring 143", output)

    @patch('cli.RFH3')
    def test_factor_command_strict_mode(self, mock_rfh3_class):
        """Test factor command in strict mode"""
        mock_rfh3 = MagicMock()
        mock_rfh3.factor.side_effect = ValueError("Test error")
        mock_rfh3_class.return_value = mock_rfh3

        args = argparse.Namespace(
            numbers=['143'],
            timeout=60.0,
            verbose=False,
            verify=False,
            strict=True,
            no_learning=False,
            no_hierarchical=False
        )

        with patch('sys.stdout', new_callable=io.StringIO):
            with patch('sys.exit') as mock_exit:
                factor_command(args)
                mock_exit.assert_called_once_with(1)

    def test_factor_command_invalid_number(self):
        """Test factor command with invalid small number"""
        args = argparse.Namespace(
            numbers=['2'],
            timeout=60.0,
            verbose=False,
            verify=False,
            strict=False,
            no_learning=False,
            no_hierarchical=False
        )

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('cli.RFH3'):
                factor_command(args)
                
                output = mock_stdout.getvalue()
                self.assertIn("Error: 2 is too small", output)

    def test_benchmark_command(self):
        """Test benchmark command"""
        args = argparse.Namespace(
            quick=True,
            extensive=False,
            save_results=None,
            compare=False
        )

        # Test that benchmark command runs successfully
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            benchmark_command(args)
            
            output = mock_stdout.getvalue()
            # Should contain benchmark output
            self.assertIn("Running Quick Benchmark", output)
            self.assertIn("BENCHMARK SUMMARY", output)

    @patch('subprocess.run')
    def test_test_command_basic(self, mock_subprocess):
        """Test basic test command"""
        mock_subprocess.return_value.returncode = 0
        
        args = argparse.Namespace(
            verbose=False,
            coverage=False,
            unit=False,
            integration=False
        )

        with patch('sys.exit') as mock_exit:
            run_tests_command(args)
            
            mock_subprocess.assert_called_once_with(
                ["python", "-m", "pytest", "tests/"], 
                check=False
            )
            mock_exit.assert_called_once_with(0)

    @patch('subprocess.run')
    def test_test_command_verbose(self, mock_subprocess):
        """Test test command with verbose flag"""
        mock_subprocess.return_value.returncode = 0
        
        args = argparse.Namespace(
            verbose=True,
            coverage=False,
            unit=False,
            integration=False
        )

        with patch('sys.exit'):
            run_tests_command(args)
            
            expected_cmd = ["python", "-m", "pytest", "-v", "tests/"]
            mock_subprocess.assert_called_once_with(expected_cmd, check=False)

    @patch('subprocess.run')
    def test_test_command_coverage(self, mock_subprocess):
        """Test test command with coverage"""
        mock_subprocess.return_value.returncode = 0
        
        args = argparse.Namespace(
            verbose=False,
            coverage=True,
            unit=False,
            integration=False
        )

        with patch('sys.exit'):
            run_tests_command(args)
            
            expected_cmd = [
                "python", "-m", "pytest", 
                "--cov=prime_resonance_field", "--cov-report=term",
                "tests/"
            ]
            mock_subprocess.assert_called_once_with(expected_cmd, check=False)

    @patch('subprocess.run')
    def test_test_command_unit_tests(self, mock_subprocess):
        """Test test command for unit tests only"""
        mock_subprocess.return_value.returncode = 0
        
        args = argparse.Namespace(
            verbose=False,
            coverage=False,
            unit=True,
            integration=False
        )

        with patch('sys.exit'):
            run_tests_command(args)
            
            expected_cmd = ["python", "-m", "pytest", "-m", "unit", "tests/"]
            mock_subprocess.assert_called_once_with(expected_cmd, check=False)

    @patch('subprocess.run')
    def test_test_command_pytest_not_found(self, mock_subprocess):
        """Test test command when pytest is not found"""
        mock_subprocess.side_effect = FileNotFoundError()
        
        args = argparse.Namespace(
            verbose=False,
            coverage=False,
            unit=False,
            integration=False
        )

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('sys.exit') as mock_exit:
                run_tests_command(args)
                
                output = mock_stdout.getvalue()
                self.assertIn("pytest not found", output)
                mock_exit.assert_called_once_with(1)

    def test_main_no_command(self):
        """Test main function with no command (should show info)"""
        sys.argv = ['rfh3']
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('sys.exit'):
                main()
                
                output = mock_stdout.getvalue()
                self.assertIn("Prime Resonance Field (RFH3)", output)

    @patch('cli.factor_command')
    def test_main_factor_command(self, mock_factor_command):
        """Test main function with factor command"""
        sys.argv = ['rfh3', 'factor', '143']
        
        with patch('sys.exit'):
            main()
            
            mock_factor_command.assert_called_once()

    def test_argument_parser_factor(self):
        """Test argument parser for factor command"""
        sys.argv = ['rfh3', 'factor', '143', '--timeout', '30', '--verbose', '--verify']
        
        # Mock the factor_command to capture args
        captured_args = None
        
        def capture_args(args):
            nonlocal captured_args
            captured_args = args
        
        with patch('cli.factor_command', side_effect=capture_args):
            with patch('sys.exit'):
                main()
        
        self.assertEqual(captured_args.numbers, ['143'])
        self.assertEqual(captured_args.timeout, 30.0)
        self.assertTrue(captured_args.verbose)
        self.assertTrue(captured_args.verify)

    def test_argument_parser_benchmark(self):
        """Test argument parser for benchmark command"""
        sys.argv = ['rfh3', 'benchmark', '--quick', '--save-results', 'test.json']
        
        captured_args = None
        
        def capture_args(args):
            nonlocal captured_args
            captured_args = args
        
        with patch('cli.benchmark_command', side_effect=capture_args):
            with patch('sys.exit'):
                main()
        
        self.assertTrue(captured_args.quick)
        self.assertEqual(captured_args.save_results, 'test.json')


class TestCLIConfiguration(unittest.TestCase):
    """Test CLI configuration options"""

    @patch('cli.RFH3')
    def test_factor_command_configuration_options(self, mock_rfh3_class):
        """Test that CLI options properly configure RFH3"""
        mock_rfh3 = MagicMock()
        mock_rfh3.factor.return_value = (11, 13)
        mock_rfh3_class.return_value = mock_rfh3

        args = argparse.Namespace(
            numbers=['143'],
            timeout=30.0,
            verbose=True,
            verify=False,
            strict=False,
            no_learning=True,
            no_hierarchical=True
        )

        with patch('sys.stdout', new_callable=io.StringIO):
            factor_command(args)

        # Check that RFH3 was instantiated with proper config
        mock_rfh3_class.assert_called_once()
        config = mock_rfh3_class.call_args[0][0]
        
        # Check timeout affects max_iterations
        self.assertEqual(config.max_iterations, 30000000)  # 30 * 1000000
        
        # Check learning disabled
        self.assertFalse(config.learning_enabled)
        
        # Check hierarchical search disabled
        self.assertFalse(config.hierarchical_search)


if __name__ == "__main__":
    unittest.main()
