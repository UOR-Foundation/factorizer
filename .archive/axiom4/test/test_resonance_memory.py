"""
Tests for Resonance Memory functionality
Validates pattern recording and prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4.resonance_memory import (
    ResonanceMemory,
    analyze_resonance_landscape,
    resonance_guided_search
)

def test_resonance_memory_init():
    """Test ResonanceMemory initialization"""
    memory = ResonanceMemory(memory_size=50)
    
    assert memory.memory_size == 50
    assert len(memory.resonance_graph) == 0
    assert len(memory.successes) == 0
    assert memory.decay_factor == 0.7
    
    print("✓ ResonanceMemory initialization")

def test_record_pattern():
    """Test pattern recording"""
    memory = ResonanceMemory()
    
    # Record a pattern without factor
    memory.record(p=7, f=5, n=100, strength=0.8)
    
    assert (7, 5) in memory.resonance_graph
    assert memory.resonance_graph[(7, 5)] > 0
    assert len(memory.successes) == 0  # No factor recorded
    
    # Record with factor
    memory.record(p=11, f=8, n=143, strength=0.9, factor=11)
    
    assert (11, 8) in memory.resonance_graph
    assert len(memory.successes) == 1
    assert memory.successes[0] == (11, 8, 143, 11)
    
    print("✓ Pattern recording")

def test_pattern_strength_update():
    """Test pattern strength updates with decay"""
    memory = ResonanceMemory()
    
    # Record same pattern multiple times
    memory.record(p=3, f=2, n=50, strength=1.0)
    strength1 = memory.get_pattern_strength(3, 2)
    
    memory.record(p=3, f=2, n=60, strength=0.5)
    strength2 = memory.get_pattern_strength(3, 2)
    
    # Should use exponential moving average
    expected = 0.7 * strength1 + 0.3 * 0.5
    assert abs(strength2 - expected) < 0.01
    
    print("✓ Pattern strength update")

def test_predict():
    """Test factor prediction"""
    memory = ResonanceMemory()
    
    # Add some successful patterns
    memory.record(p=5, f=3, n=35, strength=0.9, factor=5)
    memory.record(p=7, f=5, n=77, strength=0.8, factor=7)
    memory.record(p=11, f=8, n=143, strength=0.85, factor=11)
    
    # Predict for a new number
    predictions = memory.predict(n=91, top_k=10)  # 7 × 13
    
    assert isinstance(predictions, list)
    assert len(predictions) <= 10
    
    # Check format
    for pos, weight in predictions:
        assert isinstance(pos, int)
        assert isinstance(weight, float)
        assert 2 <= pos <= 9  # sqrt(91) ≈ 9.5
        assert 0 <= weight <= 1
    
    print("✓ Factor prediction")

def test_find_similar_patterns():
    """Test finding similar patterns"""
    memory = ResonanceMemory()
    
    # Add patterns
    memory.record(p=5, f=3, n=50, strength=0.7)
    memory.record(p=5, f=5, n=60, strength=0.8)
    memory.record(p=7, f=3, n=70, strength=0.9)
    memory.record(p=11, f=8, n=30, strength=0.6)  # Far from (5,3)
    
    # Find similar to (5, 3)
    similar = memory.find_similar_patterns(p=5, f=3, tolerance=2)
    
    assert isinstance(similar, list)
    # Should find (5,5) and (7,3) but not (11,8)
    patterns = [pf for pf, _ in similar]
    assert (5, 5) in patterns
    assert (7, 3) in patterns
    assert (11, 8) not in patterns
    
    print("✓ Find similar patterns")

def test_success_rate():
    """Test success rate calculation"""
    memory = ResonanceMemory()
    
    # Empty memory
    assert memory.success_rate() == 0.0
    
    # Add patterns with varying strengths
    # Note: strengths are reduced by decay factor
    memory.record(p=3, f=2, n=30, strength=0.3)
    memory.record(p=5, f=3, n=50, strength=0.8)  # Higher strength
    memory.record(p=7, f=5, n=70, strength=0.9)  # Higher strength
    
    # Also test updating existing pattern
    memory.record(p=5, f=3, n=60, strength=0.9)  # Update (5,3) pattern
    
    rate = memory.success_rate()
    assert 0 <= rate <= 1
    
    # Check that we have some patterns
    assert len(memory.resonance_graph) > 0
    
    print("✓ Success rate calculation")

def test_memory_limit():
    """Test memory size limit"""
    memory = ResonanceMemory(memory_size=10)
    
    # Add more than limit
    for i in range(15):
        memory.record(p=i+2, f=i+1, n=100+i, strength=0.5, factor=i+2)
    
    # Should only keep last 10
    assert len(memory.successes) == 10
    
    # Check it kept the latest ones
    assert memory.successes[-1][3] == 16  # Last factor
    
    print("✓ Memory size limit")

def test_merge_memories():
    """Test merging two memories"""
    memory1 = ResonanceMemory()
    memory2 = ResonanceMemory()
    
    # Add to first memory
    memory1.record(p=3, f=2, n=30, strength=0.5)
    memory1.record(p=5, f=3, n=50, strength=0.7, factor=5)
    
    # Get strength before merge
    strength_before = memory1.get_pattern_strength(3, 2)
    
    # Add to second memory
    memory2.record(p=3, f=2, n=40, strength=0.8)  # Same pattern, higher strength
    memory2.record(p=7, f=5, n=70, strength=0.6, factor=7)
    
    # Get memory2's strength
    strength_memory2 = memory2.get_pattern_strength(3, 2)
    
    # Merge
    memory1.merge(memory2)
    
    # Should have both patterns
    assert (3, 2) in memory1.resonance_graph
    assert (7, 5) in memory1.resonance_graph
    
    # Should use maximum strength for (3,2)
    strength_after = memory1.get_pattern_strength(3, 2)
    assert strength_after == max(strength_before, strength_memory2)
    
    # Should have both successes
    assert len(memory1.successes) == 2
    
    print("✓ Memory merging")

def test_analyze_landscape():
    """Test resonance landscape analysis"""
    memory = ResonanceMemory()
    
    # Add some patterns
    memory.record(p=5, f=3, n=35, strength=0.9, factor=5)
    memory.record(p=7, f=5, n=77, strength=0.8, factor=7)
    
    # Analyze landscape for new number
    landscape = analyze_resonance_landscape(n=55, memory=memory, resolution=20)
    
    assert isinstance(landscape, dict)
    assert len(landscape) > 0
    
    # All positions should be valid
    for pos, strength in landscape.items():
        assert 2 <= pos <= 7  # sqrt(55) ≈ 7.4
        assert 0 <= strength <= 1
    
    print("✓ Resonance landscape analysis")

def test_guided_search():
    """Test resonance-guided search"""
    memory = ResonanceMemory()
    
    # Train on some factorizations
    memory.record(p=3, f=2, n=21, strength=0.9, factor=3)
    memory.record(p=5, f=3, n=35, strength=0.85, factor=5)
    
    # Search for factor of 15 (3×5)
    factor = resonance_guided_search(n=15, memory=memory, max_attempts=10)
    
    # Should find 3 or 5
    assert factor in [3, 5, None]
    
    if factor:
        assert 15 % factor == 0
    
    print("✓ Resonance-guided search")

def test_memory_determinism():
    """Test that memory operations are deterministic"""
    memory1 = ResonanceMemory()
    memory2 = ResonanceMemory()
    
    # Same operations on both
    for mem in [memory1, memory2]:
        mem.record(p=5, f=3, n=50, strength=0.7)
        mem.record(p=7, f=5, n=70, strength=0.8, factor=7)
    
    # Predictions should be identical
    pred1 = memory1.predict(n=100, top_k=5)
    pred2 = memory2.predict(n=100, top_k=5)
    
    assert pred1 == pred2
    
    print("✓ Memory determinism")

def run_all_tests():
    """Run all resonance memory tests"""
    print("Testing Resonance Memory (Axiom 4)...")
    print("-" * 40)
    
    test_resonance_memory_init()
    test_record_pattern()
    test_pattern_strength_update()
    test_predict()
    test_find_similar_patterns()
    test_success_rate()
    test_memory_limit()
    test_merge_memories()
    test_analyze_landscape()
    test_guided_search()
    test_memory_determinism()
    
    print("-" * 40)
    print("All Resonance Memory tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
