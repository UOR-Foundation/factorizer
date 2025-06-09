"""
Tests for Spectral Navigation functionality
Validates gradient navigation and harmonic jumps
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4.spectral_navigation import (
    coherence_gradient,
    gradient_ascent,
    multi_path_search,
    harmonic_jump,
    find_coherence_peaks,
    navigate_to_factor,
    coherence_flow_lines
)
from axiom4.adaptive_observer import MultiScaleObserver

def test_coherence_gradient():
    """Test coherence gradient calculation"""
    n = 35  # 5 × 7
    observer = MultiScaleObserver(n)
    
    # Gradient at factor
    grad_5 = coherence_gradient(n, 5, observer)
    assert isinstance(grad_5, float)
    
    # Gradient at non-factor
    grad_4 = coherence_gradient(n, 4, observer)
    assert isinstance(grad_4, float)
    
    # Edge cases
    assert coherence_gradient(n, 1, observer) == 0  # Out of bounds
    assert coherence_gradient(n, 10, observer) == 0  # Too large
    
    print("✓ Coherence gradient calculation")

def test_gradient_ascent():
    """Test gradient ascent navigation"""
    n = 77  # 7 × 11
    observer = MultiScaleObserver(n)
    
    # Start near a factor
    path = gradient_ascent(n, 6, observer, max_steps=10)
    
    assert isinstance(path, list)
    assert len(path) > 0
    assert path[0] == 6  # Should start where we specified
    
    # All positions should be valid
    root = 8  # floor(sqrt(77))
    for pos in path:
        assert 2 <= pos <= root
    
    print("✓ Gradient ascent navigation")

def test_multi_path_search():
    """Test multi-path gradient search"""
    n = 143  # 11 × 13
    observer = MultiScaleObserver(n)
    
    starts = [5, 7, 9, 11]
    endpoints = multi_path_search(n, starts, observer, max_paths=3)
    
    assert isinstance(endpoints, list)
    assert len(endpoints) <= 3  # Respects max_paths
    
    # Check format
    for pos, coh in endpoints:
        assert isinstance(pos, int)
        assert isinstance(coh, float)
        assert 2 <= pos <= 11
        assert 0 <= coh <= 1
    
    # Should be sorted by coherence
    coherences = [c for _, c in endpoints]
    assert coherences == sorted(coherences, reverse=True)
    
    print("✓ Multi-path gradient search")

def test_harmonic_jump():
    """Test harmonic jump mechanism"""
    n = 100
    
    # Different stuck counts give different jumps
    jumps = []
    for stuck in range(4):
        new_pos = harmonic_jump(n, 5, stuck)
        jumps.append(new_pos)
        assert 2 <= new_pos <= 10
    
    # Should produce different positions
    assert len(set(jumps)) > 1
    
    print("✓ Harmonic jump mechanism")

def test_find_coherence_peaks():
    """Test coherence peak finding"""
    n = 55  # 5 × 11
    observer = MultiScaleObserver(n)
    
    peaks = find_coherence_peaks(n, observer, resolution=20)
    
    assert isinstance(peaks, list)
    # Should find some peaks
    assert len(peaks) > 0
    
    # All peaks should be valid positions
    root = 7  # floor(sqrt(55))
    for peak in peaks:
        assert 2 <= peak <= root
    
    print("✓ Coherence peak finding")

def test_navigate_to_factor():
    """Test factor navigation"""
    n = 21  # 3 × 7
    observer = MultiScaleObserver(n)
    
    # Start at 2, should find a factor
    factor = navigate_to_factor(n, 2, observer, max_iterations=20)
    
    # Should find 3 or 7
    assert factor in [3, 7, None]
    
    # If found, should be valid factor
    if factor:
        assert n % factor == 0
        assert factor > 1
    
    print("✓ Navigate to factor")

def test_coherence_flow_lines():
    """Test coherence flow line generation"""
    n = 35  # 5 × 7
    observer = MultiScaleObserver(n)
    
    flow_lines = coherence_flow_lines(n, observer, num_lines=5)
    
    assert isinstance(flow_lines, list)
    assert len(flow_lines) <= 5
    
    # Each flow line should be a valid path
    root = 5  # floor(sqrt(35))
    for line in flow_lines:
        assert isinstance(line, list)
        assert len(line) > 1  # Should have movement
        for pos in line:
            assert 2 <= pos <= root
    
    print("✓ Coherence flow lines")

def test_navigation_determinism():
    """Test that navigation is deterministic"""
    n = 91  # 7 × 13
    observer = MultiScaleObserver(n)
    
    # Multiple gradient calculations should be same
    grads = []
    for _ in range(3):
        g = coherence_gradient(n, 7, observer)
        grads.append(g)
    assert all(g == grads[0] for g in grads)
    
    # Gradient ascent should be deterministic
    paths = []
    for _ in range(3):
        p = gradient_ascent(n, 5, observer, max_steps=5)
        paths.append(p)
    assert all(p == paths[0] for p in paths)
    
    print("✓ Navigation determinism")

def test_edge_cases():
    """Test navigation edge cases"""
    # Small number
    n = 6  # 2 × 3
    observer = MultiScaleObserver(n)
    path = gradient_ascent(n, 2, observer)
    assert path[0] == 2
    
    # Prime number
    n = 17
    observer = MultiScaleObserver(n)
    peaks = find_coherence_peaks(n, observer, resolution=10)
    assert isinstance(peaks, list)
    
    # Perfect square
    n = 49  # 7 × 7
    observer = MultiScaleObserver(n)
    factor = navigate_to_factor(n, 6, observer, max_iterations=10)
    if factor:
        assert factor == 7
    
    print("✓ Navigation edge cases")

def test_harmonic_jump_patterns():
    """Test different harmonic jump patterns"""
    n = 200
    current = 10
    
    # Test all jump types
    jumps = []
    for stuck in range(8):  # Test multiple cycles
        jump = harmonic_jump(n, current, stuck)
        jumps.append(jump)
        assert 2 <= jump <= 14  # sqrt(200) ≈ 14
    
    # Should have variety in jumps
    unique_jumps = set(jumps)
    assert len(unique_jumps) >= 3
    
    print("✓ Harmonic jump patterns")

def run_all_tests():
    """Run all spectral navigation tests"""
    print("Testing Spectral Navigation (Axiom 4)...")
    print("-" * 40)
    
    test_coherence_gradient()
    test_gradient_ascent()
    test_multi_path_search()
    test_harmonic_jump()
    test_find_coherence_peaks()
    test_navigate_to_factor()
    test_coherence_flow_lines()
    test_navigation_determinism()
    test_edge_cases()
    test_harmonic_jump_patterns()
    
    print("-" * 40)
    print("All Spectral Navigation tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
