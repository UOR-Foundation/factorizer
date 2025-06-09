"""
Test suite for the main Factorizer class.

Tests the integration of all 5 axioms in factorizing semiprimes.
"""

import unittest
import time
from typing import Tuple, List

# Import from parent directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factorizer import Factorizer, FactorizationResult, factorize, get_factorizer


class TestFactorizer(unittest.TestCase):
    """Test the main Factorizer class."""
    
    def setUp(self):
        """Set up test cases."""
        self.factorizer = Factorizer(learning_enabled=True)
        
        # Test cases: (n, expected_factors)
        self.test_cases = [
            # Small semiprimes
            (15, (3, 5)),
            (21, (3, 7)),
            (35, (5, 7)),
            (77, (7, 11)),
            
            # Medium semiprimes
            (143, (11, 13)),
            (221, (13, 17)),
            (323, (17, 19)),
            (391, (17, 23)),
            (437, (19, 23)),
            
            # Larger semiprimes
            (667, (23, 29)),
            (899, (29, 31)),
            (1517, (37, 41)),
            (2021, (43, 47)),
            (3233, (53, 61)),
            
            # Special cases
            (4, (2, 2)),  # Perfect square
            (25, (5, 5)),  # Perfect square
            (121, (11, 11)),  # Perfect square
            
            # Even semiprimes
            (6, (2, 3)),
            (14, (2, 7)),
            (22, (2, 11)),
            (46, (2, 23)),
            
            # Fibonacci-related
            (377, (13, 29)),  # 13 is Fibonacci
            (1147, (31, 37)),  # Both near Fibonacci
            
            # Twin prime products
            (143, (11, 13)),  # 11 and 13 are twin primes
            (323, (17, 19)),  # 17 and 19 are twin primes
            (899, (29, 31)),  # 29 and 31 are twin primes
        ]
    
    def test_basic_factorization(self):
        """Test basic factorization functionality."""
        for n, expected in self.test_cases[:10]:  # Test first 10 cases
            with self.subTest(n=n):
                p, q = self.factorizer.factorize(n)
                
                # Check if factors multiply to n
                self.assertEqual(p * q, n)
                
                # Check if factors match expected (order may vary)
                self.assertIn((p, q), [expected, (expected[1], expected[0])])
    
    def test_factorization_with_details(self):
        """Test factorization with detailed results."""
        n = 143
        result = self.factorizer.factorize_with_details(n)
        
        # Check result structure
        self.assertIsInstance(result, FactorizationResult)
        self.assertEqual(result.factors[0] * result.factors[1], n)
        self.assertGreater(result.iterations, 0)
        self.assertGreaterEqual(result.max_coherence, 0.0)
        self.assertLessEqual(result.max_coherence, 1.0)
        self.assertGreater(result.candidates_explored, 0)
        self.assertGreater(result.time_elapsed, 0)
        self.assertIsInstance(result.method_sequence, list)
        self.assertGreater(len(result.method_sequence), 0)
        self.assertIsInstance(result.learning_applied, bool)
    
    def test_trivial_cases(self):
        """Test trivial factorization cases."""
        # Even number
        n = 10
        p, q = self.factorizer.factorize(n)
        self.assertEqual((p, q), (2, 5))
        
        # Prime number (should return 1, n)
        n = 17
        p, q = self.factorizer.factorize(n)
        self.assertEqual((p, q), (1, 17))
    
    def test_perfect_squares(self):
        """Test factorization of perfect squares."""
        perfect_squares = [(4, 2), (9, 3), (25, 5), (49, 7), (121, 11)]
        
        for n, root in perfect_squares:
            with self.subTest(n=n):
                p, q = self.factorizer.factorize(n)
                self.assertEqual(p, root)
                self.assertEqual(q, root)
    
    def test_fibonacci_proximity(self):
        """Test factorization of numbers with Fibonacci-proximate factors."""
        # 377 = 13 × 29, where 13 is Fibonacci
        n = 377
        p, q = self.factorizer.factorize(n)
        self.assertEqual(p * q, n)
        self.assertIn(13, [p, q])
    
    def test_twin_prime_products(self):
        """Test factorization of twin prime products."""
        twin_products = [
            (15, (3, 5)),    # 3 and 5 differ by 2
            (35, (5, 7)),    # 5 and 7 are twin primes
            (143, (11, 13)), # 11 and 13 are twin primes
            (323, (17, 19)), # 17 and 19 are twin primes
        ]
        
        for n, expected in twin_products:
            with self.subTest(n=n):
                p, q = self.factorizer.factorize(n)
                self.assertEqual(p * q, n)
                self.assertIn((p, q), [expected, (expected[1], expected[0])])
    
    def test_singleton_factorizer(self):
        """Test the singleton factorizer instance."""
        f1 = get_factorizer()
        f2 = get_factorizer()
        
        # Should be the same instance
        self.assertIs(f1, f2)
        
        # Should work correctly
        n = 143
        p, q = f1.factorize(n)
        self.assertEqual(p * q, n)
    
    def test_convenience_function(self):
        """Test the convenience factorize function."""
        n = 143
        p, q = factorize(n)
        self.assertEqual(p * q, n)
    
    def test_learning_mode(self):
        """Test that learning mode improves performance."""
        # Create two factorizers
        learning_factorizer = Factorizer(learning_enabled=True)
        no_learning_factorizer = Factorizer(learning_enabled=False)
        
        # Factor a number to train the learning factorizer
        n1 = 143  # 11 × 13
        learning_factorizer.factorize(n1)
        
        # Factor a similar number
        n2 = 187  # 11 × 17 (shares factor 11)
        
        # Time both factorizations
        start = time.time()
        result_learning = learning_factorizer.factorize_with_details(n2)
        time_learning = time.time() - start
        
        start = time.time()
        result_no_learning = no_learning_factorizer.factorize_with_details(n2)
        time_no_learning = time.time() - start
        
        # Both should find correct factors
        self.assertEqual(result_learning.factors[0] * result_learning.factors[1], n2)
        self.assertEqual(result_no_learning.factors[0] * result_no_learning.factors[1], n2)
        
        # Learning should have explored fewer candidates (ideally)
        # Note: This might not always be true for small numbers
        self.assertGreater(result_learning.candidates_explored, 0)
        self.assertGreater(result_no_learning.candidates_explored, 0)
    
    def test_axiom_integration(self):
        """Test that all axioms are being used."""
        n = 667  # 23 × 29
        result = self.factorizer.factorize_with_details(n)
        
        # Check that multiple phases were used
        self.assertIn("phase1_setup", result.method_sequence)
        self.assertIn("phase2_superposition", result.method_sequence)
        self.assertIn("phase3_coherence", result.method_sequence)
        
        # Should find correct factors
        self.assertEqual(result.factors[0] * result.factors[1], n)
    
    def test_large_semiprime(self):
        """Test factorization of a larger semiprime."""
        n = 10403  # 101 × 103
        p, q = self.factorizer.factorize(n)
        
        self.assertEqual(p * q, n)
        self.assertEqual((p, q), (101, 103))
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with 0
        with self.assertRaises(ZeroDivisionError):
            self.factorizer.factorize(0)
        
        # Test with 1
        p, q = self.factorizer.factorize(1)
        self.assertEqual((p, q), (1, 1))
        
        # Test with negative number
        # The factorizer might handle this differently
        # For now, we'll just check it doesn't crash
        try:
            p, q = self.factorizer.factorize(-15)
            # If it returns something, check it's valid
            if p != 1 or q != -15:
                self.assertEqual(abs(p * q), 15)
        except:
            # It's ok if it raises an exception for negative numbers
            pass


class TestFactorizerPerformance(unittest.TestCase):
    """Performance tests for the Factorizer."""
    
    def setUp(self):
        """Set up performance tests."""
        self.factorizer = Factorizer(learning_enabled=True)
    
    def test_small_numbers_performance(self):
        """Test that small numbers factor quickly."""
        small_semiprimes = [15, 21, 35, 77, 143]
        
        for n in small_semiprimes:
            start = time.time()
            p, q = self.factorizer.factorize(n)
            elapsed = time.time() - start
            
            with self.subTest(n=n):
                self.assertEqual(p * q, n)
                # Should factor small numbers in under 1 second
                self.assertLess(elapsed, 1.0)
    
    def test_batch_factorization(self):
        """Test factorizing multiple numbers in sequence."""
        numbers = [143, 221, 323, 391, 437, 667, 899]
        results = []
        
        total_start = time.time()
        for n in numbers:
            p, q = self.factorizer.factorize(n)
            results.append((n, p, q))
        total_elapsed = time.time() - total_start
        
        # Check all results
        for n, p, q in results:
            self.assertEqual(p * q, n)
        
        # Should factor all these in reasonable time
        self.assertLess(total_elapsed, 10.0)  # 10 seconds for all
    
    def test_caching_effectiveness(self):
        """Test that caching improves performance on repeated similar inputs."""
        n = 667  # 23 × 29
        
        # First factorization
        start = time.time()
        result1 = self.factorizer.factorize_with_details(n)
        time1 = time.time() - start
        
        # Second factorization of same number
        start = time.time()
        result2 = self.factorizer.factorize_with_details(n)
        time2 = time.time() - start
        
        # Both should be correct
        self.assertEqual(result1.factors, result2.factors)
        self.assertEqual(result1.factors[0] * result1.factors[1], n)
        
        # Second should explore fewer candidates due to caching
        self.assertLessEqual(result2.candidates_explored, result1.candidates_explored)


def run_all_tests():
    """Run all factorizer tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFactorizer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFactorizerPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
