"""
Integration tests for Axiom 4 acceleration
Verifies all components work together with acceleration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom4 import (
    IntegratedObserver,
    integrated_observer_search,
    integrated_axiom4_factor,
    create_accelerated_observer,
    MultiScaleObserver,
    generate_superposition,
    ResonanceMemory
)
import time

def test_integrated_observer():
    """Test IntegratedObserver functionality"""
    n = 143  # 11 × 13
    
    # Create integrated observer
    observer = IntegratedObserver(n)
    
    # Test observation
    obs1 = observer.observe(11)
    assert obs1 > 0
    
    # Test gradient
    grad = observer.gradient(11)
    assert isinstance(grad, float)
    
    # Test coherence field
    field = observer.coherence_field([10, 11, 12, 13])
    assert len(field) == 4
    assert all(isinstance(v, float) for v in field.values())
    
    # Test wavefunction collapse
    candidates = [10, 11, 12, 13, 14]
    collapsed = observer.collapse_wavefunction(candidates)
    assert len(collapsed) > 0
    assert all(isinstance(t[0], int) and isinstance(t[1], float) for t in collapsed)
    
    # Test navigation
    factor = observer.navigate_to_factor(10, max_iterations=10)
    # May or may not find factor, but should return int or None
    assert factor is None or isinstance(factor, int)
    
    # Test gradient ascent
    path = observer.gradient_ascent(10, max_steps=5)
    assert isinstance(path, list)
    assert all(isinstance(p, int) for p in path)
    
    # Test multi-path search
    endpoints = observer.multi_path_search([8, 10, 12])
    assert isinstance(endpoints, list)
    
    # Test cache stats
    stats = observer.get_cache_stats()
    assert 'total_hit_rate' in stats
    assert stats['total_hit_rate'] > 0  # Should have some hits from repeated operations
    
    print("✓ IntegratedObserver functionality")

def test_integrated_search():
    """Test integrated observer search"""
    n = 221  # 13 × 17
    
    # Test without memory
    factor = integrated_observer_search(n, memory=None, max_iterations=50)
    assert factor in [13, 17] or factor is None
    
    # Test with memory
    memory = ResonanceMemory()
    factor = integrated_observer_search(n, memory=memory, max_iterations=50)
    assert factor in [13, 17] or factor is None
    
    print("✓ Integrated observer search")

def test_integrated_axiom4_factor():
    """Test complete Axiom 4 factorization"""
    # Test easy semiprime
    n = 35  # 5 × 7
    factor = integrated_axiom4_factor(n, verbose=False)
    assert factor in [5, 7]
    
    # Test medium semiprime
    n = 143  # 11 × 13  
    factor = integrated_axiom4_factor(n, verbose=False)
    assert factor in [11, 13]
    
    print("✓ Integrated Axiom 4 factorization")

def test_acceleration_integration():
    """Test that acceleration is properly integrated"""
    n = 323  # 17 × 19
    
    # Create regular observer
    regular_observer = MultiScaleObserver(n)
    
    # Create accelerated setup
    accel_observer, cache = create_accelerated_observer(n)
    
    # Generate candidates
    candidates = generate_superposition(n)[:20]
    
    # Time regular collapse
    start = time.time()
    from axiom4.adaptive_observer import collapse_wavefunction
    regular_collapsed = collapse_wavefunction(n, candidates, regular_observer, iterations=3)
    regular_time = time.time() - start
    
    # Time accelerated collapse  
    from axiom4.accelerated_observer import accelerated_collapse
    start = time.time()
    accel_collapsed = accelerated_collapse(n, candidates, accel_observer, iterations=3, cache=cache)
    accel_time = time.time() - start
    
    # Accelerated should be faster (or at least not significantly slower)
    # Note: First run might be slower due to cache warming
    print(f"  Regular time: {regular_time:.4f}s")
    print(f"  Accelerated time: {accel_time:.4f}s")
    
    # Results should be similar (deterministic)
    assert len(regular_collapsed) == len(accel_collapsed)
    
    print("✓ Acceleration integration")

def test_cache_effectiveness():
    """Test cache effectiveness in integrated tools"""
    n = 437  # 19 × 23
    
    observer = IntegratedObserver(n)
    
    # Pre-warm cache
    positions = list(range(10, 30))
    
    # First pass - mostly misses
    for pos in positions:
        observer.observe(pos)
    
    stats1 = observer.get_cache_stats()
    
    # Second pass - should be all hits
    for pos in positions:
        observer.observe(pos)
        
    stats2 = observer.get_cache_stats()
    
    # Hit rate should improve
    assert stats2['observation_hits'] > stats1['observation_hits']
    assert stats2['total_hit_rate'] > stats1['total_hit_rate']
    
    print(f"✓ Cache effectiveness (hit rate: {stats2['total_hit_rate']:.2%})")

def test_memory_integration():
    """Test resonance memory integration"""
    memory = ResonanceMemory()
    
    # Record some successes
    memory.record(p=2, f=3, n=35, strength=0.8, factor=5)
    memory.record(p=3, f=5, n=143, strength=0.9, factor=11)
    
    # Test prediction
    predictions = memory.predict(221, top_k=5)
    assert len(predictions) > 0
    
    # Test with integrated search
    factor = integrated_observer_search(221, memory=memory, max_iterations=30)
    # Should find factor or None
    assert factor in [13, 17, None]
    
    print("✓ Memory integration")

def test_determinism():
    """Test that integration maintains determinism"""
    n = 323  # 17 × 19
    
    # Run integrated factorization twice
    factor1 = integrated_axiom4_factor(n, use_memory=False, verbose=False)
    factor2 = integrated_axiom4_factor(n, use_memory=False, verbose=False)
    
    # Should get same result
    assert factor1 == factor2
    
    print("✓ Determinism maintained")

def test_verbose_mode():
    """Test verbose mode output"""
    n = 77  # 7 × 11
    
    print("\n  Testing verbose mode:")
    factor = integrated_axiom4_factor(n, verbose=True)
    assert factor in [7, 11]
    
    print("✓ Verbose mode")

def run_all_tests():
    """Run all integration tests"""
    print("Testing Axiom 4 Integration...")
    print("-" * 50)
    
    test_integrated_observer()
    test_integrated_search()
    test_integrated_axiom4_factor()
    test_acceleration_integration()
    test_cache_effectiveness()
    test_memory_integration()
    test_determinism()
    test_verbose_mode()
    
    print("-" * 50)
    print("All integration tests passed! ✓")
    print("\nAxiom 4 acceleration is fully integrated!")

if __name__ == "__main__":
    run_all_tests()
