"""
Tests for Failure Analysis functionality
Validates failure pattern tracking and adaptive strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom5.failure_analysis import (
    FailureMemory,
    analyze_failure_patterns,
    detect_spectral_nulls,
    adaptive_strategy,
    inverse_failure_search,
    failure_gradient
)

def test_failure_memory_init():
    """Test FailureMemory initialization"""
    memory = FailureMemory(memory_size=50)
    
    assert memory.memory_size == 50
    assert len(memory.failed_positions) == 0
    assert len(memory.failure_patterns) == 0
    assert len(memory.dead_end_regions) == 0
    assert len(memory.spectral_nulls) == 0
    
    print("✓ FailureMemory initialization")

def test_record_failure():
    """Test failure recording"""
    memory = FailureMemory()
    
    # Record failures
    memory.record_failure(n=77, position=4, method='axiom1', coherence_value=0.2)
    memory.record_failure(n=77, position=6, method='axiom2', coherence_value=0.05)
    
    assert len(memory.failure_patterns) == 2
    assert 4 in memory.failed_positions
    assert 6 in memory.spectral_nulls  # Low coherence
    
    print("✓ Failure recording")

def test_analyze_position():
    """Test position characteristic analysis"""
    memory = FailureMemory()
    
    # Analyze even position
    chars_4 = memory._analyze_position(4, 100)
    assert chars_4['even'] == True
    assert chars_4['divisible_by_3'] == False
    assert chars_4['power_of_2'] == True  # 4 = 2^2
    
    # Analyze odd position
    chars_15 = memory._analyze_position(15, 100)
    assert chars_15['even'] == False
    assert chars_15['divisible_by_3'] == True
    assert chars_15['divisible_by_5'] == True
    
    print("✓ Position analysis")

def test_identify_dead_ends():
    """Test dead end region identification"""
    memory = FailureMemory()
    
    # Create dense failure region
    for pos in [5, 6, 7, 8, 9, 10]:
        memory.record_failure(n=100, position=pos, method='axiom1')
    
    # Also some sparse failures
    memory.record_failure(n=100, position=20, method='axiom1')
    memory.record_failure(n=100, position=30, method='axiom1')
    
    dead_ends = memory.identify_dead_ends(threshold=5)
    
    assert len(dead_ends) > 0
    # Should identify 5-10 region as dead end
    assert any(start <= 5 and end >= 10 for start, end in dead_ends)
    
    print("✓ Dead end identification")

def test_failure_probability():
    """Test failure probability estimation"""
    memory = FailureMemory()
    
    # Record failures
    memory.record_failure(n=77, position=5, method='axiom1')
    memory.record_failure(n=77, position=10, method='axiom2')
    
    # Identify dead ends
    memory.dead_end_regions = [(8, 12)]
    
    # Test probabilities
    prob_5 = memory.get_failure_probability(5)
    assert prob_5 > 0.8  # Direct failure
    
    prob_6 = memory.get_failure_probability(6)
    assert 0.1 < prob_6 < 0.8  # Near failure
    
    prob_10 = memory.get_failure_probability(10)
    assert prob_10 > 0.7  # In dead end
    
    prob_50 = memory.get_failure_probability(50)
    assert prob_50 < 0.2  # Far from failures
    
    print("✓ Failure probability estimation")

def test_analyze_failure_patterns():
    """Test failure pattern analysis"""
    memory = FailureMemory()
    
    # Add varied failures
    memory.record_failure(n=100, position=4, method='axiom1', coherence_value=0.3)
    memory.record_failure(n=100, position=6, method='axiom1', coherence_value=0.2)
    memory.record_failure(n=100, position=8, method='axiom2', coherence_value=0.1)
    memory.record_failure(n=100, position=15, method='axiom3', coherence_value=0.4)
    
    patterns = analyze_failure_patterns(memory)
    
    assert isinstance(patterns, dict)
    assert 'method_axiom1_fail_rate' in patterns
    assert patterns['method_axiom1_fail_rate'] == 0.5  # 2/4
    assert 'char_even_fail_rate' in patterns
    assert patterns['char_even_fail_rate'] > 0.5  # 3/4 are even
    assert 'avg_failure_coherence' in patterns
    
    print("✓ Failure pattern analysis")

def test_detect_spectral_nulls_func():
    """Test spectral null detection"""
    # Reset global cache to ensure clean state
    import axiom5.meta_acceleration_cache
    axiom5.meta_acceleration_cache._meta_cache = None
    
    memory = FailureMemory()
    n = 77
    
    # Add low-coherence failures
    memory.record_failure(n=n, position=4, method='axiom1', coherence_value=0.05)
    memory.record_failure(n=n, position=6, method='axiom2', coherence_value=0.08)
    memory.record_failure(n=n, position=10, method='axiom3', coherence_value=0.5)
    
    nulls = detect_spectral_nulls(n, memory, coherence_threshold=0.1)
    
    assert isinstance(nulls, list)
    assert 4 in nulls
    assert 6 in nulls
    assert 10 not in nulls  # Coherence too high
    
    print("✓ Spectral null detection")

def test_adaptive_strategy():
    """Test adaptive strategy based on failures"""
    memory = FailureMemory()
    n = 91
    
    # Record many even failures
    for pos in [4, 6, 8, 10, 12]:
        memory.record_failure(n=n, position=pos, method='axiom1')
    
    base_candidates = [3, 4, 5, 6, 7, 8, 9, 10]
    adapted = adaptive_strategy(n, memory, base_candidates)
    
    assert isinstance(adapted, list)
    # Should filter out some failed positions
    assert 4 not in adapted or 6 not in adapted
    
    # Should have some odd positions
    odd_count = sum(1 for c in adapted if c % 2 == 1)
    assert odd_count > 0
    
    print("✓ Adaptive strategy")

def test_inverse_failure_search():
    """Test inverse failure search"""
    memory = FailureMemory()
    n = 143
    
    # Record failures with patterns
    for pos in [4, 6, 8, 10, 12]:  # Many evens
        memory.record_failure(n=n, position=pos, method='axiom1')
    
    anti_positions = inverse_failure_search(n, memory)
    
    assert isinstance(anti_positions, list)
    assert len(anti_positions) > 0
    
    # Should avoid failed positions
    for pos in anti_positions:
        assert pos not in [4, 6, 8, 10, 12]
    
    # Should include odd positions
    odd_count = sum(1 for p in anti_positions if p % 2 == 1)
    assert odd_count > len(anti_positions) // 2
    
    print("✓ Inverse failure search")

def test_failure_gradient():
    """Test failure gradient field"""
    memory = FailureMemory()
    n = 55
    
    # Add failures
    memory.record_failure(n=n, position=3, method='axiom1')
    memory.record_failure(n=n, position=4, method='axiom2')
    memory.dead_end_regions = [(3, 5)]
    
    gradient = failure_gradient(n, memory)
    
    assert isinstance(gradient, dict)
    
    # Positions far from failures should have high gradient
    assert gradient.get(7, 0) > gradient.get(4, 1)
    
    # Dead end positions should have low gradient
    assert gradient.get(4, 1) < 0.2
    
    print("✓ Failure gradient field")

def test_failure_determinism():
    """Test that failure analysis is deterministic"""
    memory1 = FailureMemory()
    memory2 = FailureMemory()
    
    # Same failures
    for mem in [memory1, memory2]:
        mem.record_failure(n=77, position=5, method='axiom1', coherence_value=0.3)
        mem.record_failure(n=77, position=7, method='axiom2', coherence_value=0.1)
    
    # Analysis should be same
    patterns1 = analyze_failure_patterns(memory1)
    patterns2 = analyze_failure_patterns(memory2)
    
    assert patterns1 == patterns2
    
    print("✓ Failure analysis determinism")

def run_all_tests():
    """Run all failure analysis tests"""
    print("Testing Failure Analysis (Axiom 5)...")
    print("-" * 40)
    
    test_failure_memory_init()
    test_record_failure()
    test_analyze_position()
    test_identify_dead_ends()
    test_failure_probability()
    test_analyze_failure_patterns()
    test_detect_spectral_nulls_func()
    test_adaptive_strategy()
    test_inverse_failure_search()
    test_failure_gradient()
    test_failure_determinism()
    
    print("-" * 40)
    print("All Failure Analysis tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
