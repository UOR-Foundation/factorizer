"""
Simple test for RFH3 implementation
"""

import time
from rfh3 import RFH3, RFH3Config

def test_simple():
    """Test RFH3 with simple cases"""
    
    # Create config with reduced iterations
    config = RFH3Config()
    config.max_iterations = 1000  # Limit iterations for testing
    config.hierarchical_search = False  # Disable hierarchical search for now
    config.log_level = 20  # INFO level
    
    rfh3 = RFH3(config)
    
    # Test very simple cases
    test_cases = [
        (6, (2, 3)),
        (15, (3, 5)),
        (35, (5, 7)),
        (143, (11, 13)),
        (323, (17, 19))
    ]
    
    print("Testing RFH3 Implementation (Simple Test)")
    print("=" * 50)
    
    for n, (p_true, q_true) in test_cases:
        try:
            start = time.time()
            p_found, q_found = rfh3.factor(n)
            elapsed = time.time() - start
            
            if {p_found, q_found} == {p_true, q_true}:
                print(f"✓ {n:6d} = {p_found:3d} × {q_found:3d} ({elapsed:.3f}s)")
            else:
                print(f"✗ {n:6d}: Expected {p_true} × {q_true}, got {p_found} × {q_found}")
        
        except Exception as e:
            print(f"✗ {n:6d}: FAILED - {str(e)}")
    
    print("=" * 50)
    
    # Test basic functionality
    print("\nTesting basic components:")
    
    # Test MultiScaleResonance
    from rfh3 import MultiScaleResonance
    analyzer = MultiScaleResonance()
    
    # Test coarse resonance
    res = analyzer.compute_coarse_resonance(11, 143)
    print(f"Coarse resonance(11, 143) = {res:.3f}")
    
    # Test full resonance
    res = analyzer.compute_resonance(11, 143)
    print(f"Full resonance(11, 143) = {res:.3f}")
    
    # Test LazyResonanceIterator
    from rfh3 import LazyResonanceIterator
    iterator = LazyResonanceIterator(143, analyzer)
    
    print("\nFirst 10 nodes from iterator:")
    for i, x in enumerate(iterator):
        if i >= 10:
            break
        res = analyzer.compute_coarse_resonance(x, 143)
        print(f"  x={x:3d}, resonance={res:.3f}")


if __name__ == "__main__":
    test_simple()
