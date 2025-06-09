"""
Tests for StateManager component
"""

import json
import os
import sys
import tempfile
import unittest

from prime_resonance_field.core import StateManager


class TestStateManager(unittest.TestCase):
    """Test cases for StateManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = StateManager(checkpoint_interval=10)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.manager.checkpoint_interval, 10)
        self.assertEqual(len(self.manager.sliding_window), 0)
        self.assertEqual(len(self.manager.checkpoints), 0)
        self.assertEqual(self.manager.iteration_count, 0)
        self.assertEqual(self.manager.best_resonance, 0.0)
        self.assertEqual(self.manager.best_position, 0)

    def test_update_functionality(self):
        """Test state update functionality"""
        # Add some data points
        self.manager.update(10, 0.5)
        self.manager.update(20, 0.8)
        self.manager.update(30, 0.3)

        self.assertEqual(self.manager.iteration_count, 3)
        self.assertEqual(len(self.manager.sliding_window), 3)
        self.assertEqual(self.manager.best_resonance, 0.8)
        self.assertEqual(self.manager.best_position, 20)

    def test_sliding_window_maxlen(self):
        """Test sliding window respects maximum length"""
        # Add more than maxlen items
        for i in range(1500):
            self.manager.update(i, 0.1)

        # Should be capped at 1000 (default maxlen)
        self.assertEqual(len(self.manager.sliding_window), 1000)
        self.assertEqual(self.manager.iteration_count, 1500)

    def test_checkpoint_creation(self):
        """Test checkpoint creation"""
        # Add enough updates to trigger checkpoint
        for i in range(15):
            self.manager.update(i, 0.1 * i)

        # Should have created at least one checkpoint
        self.assertGreater(len(self.manager.checkpoints), 0)

        # Check checkpoint content
        checkpoint = self.manager.checkpoints[-1]
        self.assertIn("iteration", checkpoint)
        self.assertIn("best_resonance", checkpoint)
        self.assertIn("best_position", checkpoint)
        self.assertIn("timestamp", checkpoint)

    def test_get_statistics(self):
        """Test statistics computation"""
        # Add some test data
        test_data = [(10, 0.5), (20, 0.8), (30, 0.3), (40, 0.9), (50, 0.2)]
        for x, res in test_data:
            self.manager.update(x, res)

        stats = self.manager.get_statistics()

        self.assertIn("iterations", stats)
        self.assertIn("best_resonance", stats)
        self.assertIn("best_position", stats)
        self.assertIn("mean_recent_resonance", stats)
        self.assertIn("std_recent_resonance", stats)

        self.assertEqual(stats["iterations"], 5)
        self.assertEqual(stats["best_resonance"], 0.9)
        self.assertEqual(stats["best_position"], 40)

        # Check mean calculation
        expected_mean = sum(res for _, res in test_data) / len(test_data)
        self.assertAlmostEqual(stats["mean_recent_resonance"], expected_mean, places=5)

    def test_reset_functionality(self):
        """Test reset functionality"""
        # Add some data
        for i in range(5):
            self.manager.update(i, 0.1 * i)

        # Reset
        self.manager.reset()

        # Should be back to initial state
        self.assertEqual(self.manager.iteration_count, 0)
        self.assertEqual(len(self.manager.sliding_window), 0)
        self.assertEqual(self.manager.best_resonance, 0.0)
        self.assertEqual(self.manager.best_position, 0)


class TestStateManagerPersistence(unittest.TestCase):
    """Test persistence functionality of StateManager"""

    def test_save_and_load(self):
        """Test save/load to file"""
        manager = StateManager(checkpoint_interval=5)

        # Add some data
        test_data = [(10, 0.5), (20, 0.8), (30, 0.3)]
        for x, res in test_data:
            manager.update(x, res)

        # Force a checkpoint
        manager.checkpoint()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            manager.save_to_file(temp_file)

            # Create new manager and load
            new_manager = StateManager()
            new_manager.resume_from_file(temp_file)

            # Should have same state
            self.assertEqual(new_manager.iteration_count, manager.iteration_count)
            self.assertEqual(new_manager.best_resonance, manager.best_resonance)
            self.assertEqual(new_manager.best_position, manager.best_position)

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_save_file_format(self):
        """Test that saved file has correct format"""
        manager = StateManager(checkpoint_interval=5)

        # Add data and create checkpoint
        manager.update(10, 0.5)
        manager.checkpoint()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            manager.save_to_file(temp_file)

            # Read and verify file format
            with open(temp_file, "r") as f:
                data = json.load(f)

            self.assertIn("checkpoints", data)
            self.assertIn("final_state", data)

            final_state = data["final_state"]
            self.assertIn("iteration", final_state)
            self.assertIn("best_resonance", final_state)
            self.assertIn("best_position", final_state)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
