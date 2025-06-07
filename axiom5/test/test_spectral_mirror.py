"""
Tests for Spectral Mirror functionality
Validates spectral reflection and mirror point finding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom5.spectral_mirror import (
    SpectralMirror,
    find_mirror_points,
    inverse_spectral_map,
    recursive_mirror,
    create_mirror_field,
    detect_mirror_symmetries,
    spectral_modulated_search
)

def test_spectral_mirror_init():
    """Test SpectralMirror initialization"""
    n = 143  # 11 × 13
    mirror = SpectralMirror(n)
    
    assert mirror.n == 143
    assert mirror.root == 11
    assert len(mirror.n_spectrum) > 0
    assert len(mirror.mirror_cache) == 0
    
    print("✓ SpectralMirror initialization")

def test_spectral_distance():
    """Test spectral distance calculation"""
    n = 77
    mirror = SpectralMirror(n)
    
    # Distance to self should be 0
    dist_self = mirror.spectral_distance(5, 5)
    assert dist_self == 0.0
    
    # Distance should be symmetric
    dist_12 = mirror.spectral_distance(5, 7)
    dist_21 = mirror.spectral_distance(7, 5)
    assert abs(dist_12 - dist_21) < 0.001
    
    # Distance should be positive
    assert dist_12 >= 0
    
    print("✓ Spectral distance calculation")

def test_find_mirror_point():
    """Test mirror point finding"""
    n = 35  # 5 × 7
    mirror = SpectralMirror(n)
    
    # Find mirror of factor
    mirror_5 = mirror.find_mirror_point(5)
    assert 2 <= mirror_5 <= 5  # sqrt(35) ≈ 5.9
    
    # Second call should use cache (test by calling again)
    mirror_5_cached = mirror.find_mirror_point(5)
    assert mirror_5 == mirror_5_cached
    
    # Test non-factor
    mirror_4 = mirror.find_mirror_point(4)
    assert 2 <= mirror_4 <= 5
    
    print("✓ Mirror point finding")

def test_spectral_reflection():
    """Test spectral reflection"""
    n = 55  # 5 × 11
    mirror = SpectralMirror(n)
    
    # Reflect various positions
    reflect_3 = mirror.spectral_reflection(3)
    reflect_5 = mirror.spectral_reflection(5)
    
    assert 2 <= reflect_3 <= 7  # sqrt(55) ≈ 7.4
    assert 2 <= reflect_5 <= 7
    
    # Different positions should have different reflections (usually)
    # But not always, so just check they're valid
    assert isinstance(reflect_3, int)
    assert isinstance(reflect_5, int)
    
    print("✓ Spectral reflection")

def test_recursive_mirror():
    """Test recursive mirroring"""
    n = 91  # 7 × 13
    mirror = SpectralMirror(n)
    
    # Apply recursive mirroring
    mirrors = mirror.recursive_mirror(7, depth=3)
    
    assert isinstance(mirrors, list)
    assert mirrors[0] == 7  # Should start with input
    assert len(mirrors) <= 4  # depth + 1 or until cycle
    
    # All should be valid positions
    root = 9  # floor(sqrt(91)) = 9
    for m in mirrors:
        assert 2 <= m <= root
    
    print("✓ Recursive mirroring")

def test_find_mirror_points_func():
    """Test find_mirror_points function"""
    n = 143
    positions = [5, 7, 9, 11]
    
    pairs = find_mirror_points(n, positions)
    
    assert isinstance(pairs, list)
    # Each pair should be (original, mirror)
    for orig, mirr in pairs:
        assert orig in positions
        assert orig != mirr
        assert 2 <= mirr <= 11
    
    print("✓ Find mirror points function")

def test_inverse_spectral_map():
    """Test inverse spectral mapping"""
    n = 77
    # Use spectrum of a known position
    from axiom3 import spectral_vector
    target = spectral_vector(7)
    
    candidates = inverse_spectral_map(n, target)
    
    assert isinstance(candidates, list)
    assert len(candidates) <= 10
    
    # All should be valid positions
    for pos in candidates:
        assert 2 <= pos <= 8
    
    print("✓ Inverse spectral mapping")

def test_create_mirror_field():
    """Test mirror field creation"""
    n = 35
    field = create_mirror_field(n, resolution=10)
    
    assert isinstance(field, dict)
    assert len(field) > 0
    
    # Check all mappings
    for pos, mirror in field.items():
        assert 2 <= pos <= 5
        assert 2 <= mirror <= 5
    
    print("✓ Mirror field creation")

def test_detect_mirror_symmetries():
    """Test mirror symmetry detection"""
    n = 55
    
    symmetries = detect_mirror_symmetries(n)
    
    assert isinstance(symmetries, list)
    # Each symmetry is (pos1, pos2, strength)
    for pos1, pos2, strength in symmetries:
        assert 2 <= pos1 <= 7
        assert 2 <= pos2 <= 7
        assert 0 <= strength <= 1
    
    # Should be sorted by strength
    if len(symmetries) > 1:
        strengths = [s for _, _, s in symmetries]
        assert strengths == sorted(strengths, reverse=True)
    
    print("✓ Mirror symmetry detection")

def test_spectral_modulated_search():
    """Test spectral modulated search"""
    n = 91
    base_positions = [5, 7, 9]
    
    modulated = spectral_modulated_search(n, base_positions)
    
    assert isinstance(modulated, list)
    assert len(modulated) >= len(base_positions)
    
    # Should include base positions
    for base in base_positions:
        assert base in modulated
    
    # All should be valid
    for pos in modulated:
        assert 2 <= pos <= 9
    
    print("✓ Spectral modulated search")

def test_mirror_determinism():
    """Test that mirror operations are deterministic"""
    n = 143
    
    # Create two mirrors
    mirror1 = SpectralMirror(n)
    mirror2 = SpectralMirror(n)
    
    # Same operations should give same results
    m1_7 = mirror1.find_mirror_point(7)
    m2_7 = mirror2.find_mirror_point(7)
    assert m1_7 == m2_7
    
    r1_5 = mirror1.spectral_reflection(5)
    r2_5 = mirror2.spectral_reflection(5)
    assert r1_5 == r2_5
    
    print("✓ Mirror determinism")

def run_all_tests():
    """Run all spectral mirror tests"""
    print("Testing Spectral Mirror (Axiom 5)...")
    print("-" * 40)
    
    test_spectral_mirror_init()
    test_spectral_distance()
    test_find_mirror_point()
    test_spectral_reflection()
    test_recursive_mirror()
    test_find_mirror_points_func()
    test_inverse_spectral_map()
    test_create_mirror_field()
    test_detect_mirror_symmetries()
    test_spectral_modulated_search()
    test_mirror_determinism()
    
    print("-" * 40)
    print("All Spectral Mirror tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
