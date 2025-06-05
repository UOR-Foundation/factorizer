"""
Tests for Prime Geodesic functionality
Validates prime coordinate navigation and geodesic paths
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom1.prime_geodesic import PrimeGeodesic
from axiom1.prime_core import primes_up_to

def test_prime_geodesic_init():
    """Test PrimeGeodesic initialization"""
    n = 210  # 2 × 3 × 5 × 7
    geo = PrimeGeodesic(n)
    assert geo.n == 210
    
    # Check prime coordinates
    # 210 = 0 mod 2, 0 mod 3, 0 mod 5, 0 mod 7, 1 mod 11, etc.
    assert geo.coord[0] == 0  # 210 % 2
    assert geo.coord[1] == 0  # 210 % 3
    assert geo.coord[2] == 0  # 210 % 5
    assert geo.coord[3] == 0  # 210 % 7
    assert geo.coord[4] == 1  # 210 % 11
    
    print("✓ PrimeGeodesic initialization and coordinates")

def test_pull_calculation():
    """Test gravitational pull calculation"""
    # n = 30 = 2 × 3 × 5
    geo = PrimeGeodesic(30)
    
    # Pull at 2 should be high (shares factor 2 with 30)
    pull_2 = geo._pull(2)
    assert pull_2 > 0
    
    # Pull at 6 = 2×3 should be very high (shares 2 and 3 with 30)
    pull_6 = geo._pull(6)
    assert pull_6 > pull_2
    
    # Pull at 7 should be 0 (no shared factors with 30)
    pull_7 = geo._pull(7)
    assert pull_7 == 0
    
    # Pull at 15 = 3×5 should be high
    pull_15 = geo._pull(15)
    assert pull_15 > 0
    
    print("✓ Prime pull calculations")

def test_pull_strength_ordering():
    """Test that pull strength follows expected ordering"""
    n = 420  # 2² × 3 × 5 × 7
    geo = PrimeGeodesic(n)
    
    # More shared prime factors = stronger pull
    pull_2 = geo._pull(2)    # shares 2
    pull_6 = geo._pull(6)    # shares 2, 3
    pull_14 = geo._pull(14)  # shares 2, 7
    pull_21 = geo._pull(21)  # shares 3, 7
    
    # 6 and 14 should have stronger pull than just 2
    assert pull_6 > pull_2
    assert pull_14 > pull_2
    
    # Numbers with no shared factors should have zero pull
    assert geo._pull(11) == 0
    assert geo._pull(13) == 0
    
    print("✓ Pull strength ordering")

def test_geodesic_walk_basic():
    """Test basic geodesic walking"""
    n = 35  # 5 × 7
    geo = PrimeGeodesic(n)
    
    # Starting from 2, should move toward factors
    path = geo.walk(2, steps=10)
    assert len(path) >= 1
    assert path[0] == 2
    
    # Path should move toward positions with higher pull
    for i in range(1, len(path)):
        # Each step should be within ±3 of previous
        assert abs(path[i] - path[i-1]) <= 3
    
    print("✓ Basic geodesic walking")

def test_geodesic_finds_factors():
    """Test that geodesic can find factors"""
    test_cases = [
        (15, 3),   # 3 × 5
        (21, 3),   # 3 × 7
        (35, 5),   # 5 × 7
        (77, 7),   # 7 × 11
    ]
    
    for n, expected_factor in test_cases:
        geo = PrimeGeodesic(n)
        # Start near the factor
        path = geo.walk(expected_factor - 3, steps=10)
        # Should find the factor in the path
        assert expected_factor in path, f"Expected to find {expected_factor} for n={n}"
    
    print("✓ Geodesic finds factors")

def test_geodesic_convergence():
    """Test that geodesic converges to stable points"""
    n = 91  # 7 × 13
    geo = PrimeGeodesic(n)
    
    # Walk should eventually stabilize
    path = geo.walk(2, steps=30)
    
    # Check if path stabilizes (same position repeated)
    if len(path) > 5:
        # Look for convergence in last few steps
        last_positions = path[-5:]
        # If converged, positions should repeat
        unique_last = len(set(last_positions))
        assert unique_last <= 3, "Path should converge to few positions"
    
    print("✓ Geodesic convergence")

def test_geodesic_respects_bounds():
    """Test that geodesic respects boundaries"""
    n = 100
    geo = PrimeGeodesic(n)
    
    # Walk from various starting points
    for start in [2, 5, 8, 9]:
        path = geo.walk(start, steps=20)
        for pos in path:
            assert 2 <= pos <= 10, f"Position {pos} out of bounds for n={n}"
    
    print("✓ Geodesic respects bounds")

def test_edge_cases():
    """Test edge cases"""
    # Small prime
    geo = PrimeGeodesic(7)
    path = geo.walk(2, steps=5)
    assert all(2 <= p <= 2 for p in path)  # sqrt(7) ≈ 2.6
    
    # Perfect square
    geo = PrimeGeodesic(49)  # 7²
    path = geo.walk(5, steps=10)
    assert 7 in path or path[-1] == 7, "Should find or approach factor 7"
    
    # Prime number (no factors except 1 and itself)
    geo = PrimeGeodesic(97)
    path = geo.walk(2, steps=20)
    # Should still produce valid path even with no factors
    assert all(2 <= p <= 9 for p in path)  # sqrt(97) ≈ 9.8
    
    print("✓ Edge cases handled")

def test_deterministic_paths():
    """Test that geodesic paths are deterministic"""
    n = 143  # 11 × 13
    
    # Run multiple times from same starting point
    paths = []
    for _ in range(3):
        geo = PrimeGeodesic(n)
        path = geo.walk(5, steps=15)
        paths.append(path)
    
    # All paths should be identical
    for i in range(1, len(paths)):
        assert paths[i] == paths[0], "Geodesic paths should be deterministic"
    
    print("✓ Deterministic geodesic paths")

def run_all_tests():
    """Run all prime geodesic tests"""
    print("Testing Prime Geodesic (Axiom 1)...")
    print("-" * 40)
    
    test_prime_geodesic_init()
    test_pull_calculation()
    test_pull_strength_ordering()
    test_geodesic_walk_basic()
    test_geodesic_finds_factors()
    test_geodesic_convergence()
    test_geodesic_respects_bounds()
    test_edge_cases()
    test_deterministic_paths()
    
    print("-" * 40)
    print("All Prime Geodesic tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
