"""
Tests for Quantum Tools functionality
Validates quantum tunneling, harmonic amplification, and spectral folding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4.quantum_tools import (
    QuantumTunnel,
    harmonic_amplify,
    SpectralFolder,
    quantum_superposition_collapse,
    entangle_positions
)

def test_quantum_tunnel_init():
    """Test QuantumTunnel initialization"""
    n = 100
    tunnel = QuantumTunnel(n)
    
    assert tunnel.n == 100
    assert tunnel.root == 10
    
    print("✓ QuantumTunnel initialization")

def test_quantum_tunnel_exit():
    """Test quantum tunnel exit finding"""
    n = 143  # 11 × 13
    tunnel = QuantumTunnel(n)
    
    # Test exit from blocked position
    blocked = 5
    exit_pos = tunnel.exit(blocked, width=30)
    
    assert exit_pos > blocked
    assert exit_pos <= 11
    
    # Should prefer Fibonacci or prime exits
    # (might be 8 or 7 depending on what's available)
    
    print("✓ Quantum tunnel exit")

def test_tunnel_sequence():
    """Test tunnel sequence generation"""
    n = 200
    tunnel = QuantumTunnel(n)
    
    sequence = tunnel.tunnel_sequence(start=5, max_tunnels=3)
    
    assert isinstance(sequence, list)
    assert len(sequence) <= 3
    
    # Each position should be further than previous
    for i in range(1, len(sequence)):
        assert sequence[i] > sequence[i-1]
    
    print("✓ Tunnel sequence generation")

def test_harmonic_amplify():
    """Test harmonic amplification"""
    n = 100
    x = 5
    
    harmonics = harmonic_amplify(n, x)
    
    assert isinstance(harmonics, list)
    assert len(harmonics) > 0
    
    # Should include some multiples
    assert 10 in harmonics  # 2×5
    
    # All harmonics should be valid
    root = 10
    for h in harmonics:
        assert 2 <= h <= root
    
    # Should be sorted
    assert harmonics == sorted(harmonics)
    
    print("✓ Harmonic amplification")

def test_spectral_folder_init():
    """Test SpectralFolder initialization"""
    n = 64
    folder = SpectralFolder(n)
    
    assert folder.n == 64
    assert folder.root == 8
    assert isinstance(folder.points, list)
    
    # Should include powers of 2
    assert 2 in folder.points
    assert 4 in folder.points
    assert 8 in folder.points
    
    # Should include some Fibonacci numbers
    assert 2 in folder.points or 3 in folder.points
    assert 5 in folder.points or 8 in folder.points
    
    print("✓ SpectralFolder initialization")

def test_spectral_folder_navigation():
    """Test spectral folder navigation"""
    n = 100
    folder = SpectralFolder(n)
    
    # Test next fold
    next_fold = folder.next_fold(5)
    assert next_fold > 5
    assert next_fold <= 10
    
    # Test nearest fold
    nearest = folder.nearest_fold(7)
    assert isinstance(nearest, int)
    assert 2 <= nearest <= 10
    
    print("✓ Spectral folder navigation")

def test_quantum_superposition_collapse():
    """Test quantum superposition collapse"""
    n = 77  # 7 × 11
    
    positions = [3, 5, 7, 9, 11]
    weights = [0.1, 0.3, 0.8, 0.2, 0.4]
    
    # Partial collapse
    collapsed = quantum_superposition_collapse(n, positions, weights, 
                                             collapse_factor=0.5)
    
    assert isinstance(collapsed, list)
    assert len(collapsed) < len(positions)
    assert len(collapsed) >= 1
    
    # Should keep high-weight positions
    assert 7 in collapsed  # Highest weight
    
    print("✓ Quantum superposition collapse")

def test_entangle_positions():
    """Test position entanglement"""
    n = 100
    pos1, pos2 = 4, 6
    
    entangled = entangle_positions(n, pos1, pos2)
    
    assert isinstance(entangled, list)
    assert len(entangled) > 0
    
    # Should include some combinations
    possible = {
        10,  # pos1 + pos2
        2,   # |pos1 - pos2|
        4,   # geometric mean ≈ 4.9
        4,   # harmonic mean = 4.8
    }
    
    # At least one combination should appear
    assert any(e in possible for e in entangled)
    
    # All should be valid
    for e in entangled:
        assert 2 <= e <= 10
    
    print("✓ Position entanglement")

def test_quantum_tools_determinism():
    """Test that quantum tools are deterministic"""
    n = 143
    
    # Harmonic amplification
    h1 = harmonic_amplify(n, 5)
    h2 = harmonic_amplify(n, 5)
    assert h1 == h2
    
    # Tunnel exit
    tunnel = QuantumTunnel(n)
    e1 = tunnel.exit(5, 30)
    e2 = tunnel.exit(5, 30)
    assert e1 == e2
    
    # Entanglement
    ent1 = entangle_positions(n, 3, 7)
    ent2 = entangle_positions(n, 3, 7)
    assert ent1 == ent2
    
    print("✓ Quantum tools determinism")

def test_edge_cases():
    """Test quantum tools edge cases"""
    # Small number
    n = 6
    tunnel = QuantumTunnel(n)
    exit_pos = tunnel.exit(2, width=10)
    assert exit_pos >= 2
    
    # Harmonic amplify with x=1
    harmonics = harmonic_amplify(10, 1)
    assert isinstance(harmonics, list)
    
    # Empty superposition collapse
    collapsed = quantum_superposition_collapse(10, [], [])
    assert collapsed == []
    
    # Entangle same position
    entangled = entangle_positions(20, 3, 3)
    assert 3 in entangled  # geometric mean = 3
    
    print("✓ Quantum tools edge cases")

def test_spectral_folder_patterns():
    """Test spectral folder patterns"""
    n = 128  # Power of 2
    folder = SpectralFolder(n)
    
    # Should have all small powers of 2
    powers = [2, 4, 8, 16, 32, 64]
    for p in powers:
        if p <= folder.root:
            assert p in folder.points
    
    # Test folding pattern
    assert len(folder.points) > 5  # Should have decent coverage
    
    print("✓ Spectral folder patterns")

def run_all_tests():
    """Run all quantum tools tests"""
    print("Testing Quantum Tools (Axiom 4)...")
    print("-" * 40)
    
    test_quantum_tunnel_init()
    test_quantum_tunnel_exit()
    test_tunnel_sequence()
    test_harmonic_amplify()
    test_spectral_folder_init()
    test_spectral_folder_navigation()
    test_quantum_superposition_collapse()
    test_entangle_positions()
    test_quantum_tools_determinism()
    test_edge_cases()
    test_spectral_folder_patterns()
    
    print("-" * 40)
    print("All Quantum Tools tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
