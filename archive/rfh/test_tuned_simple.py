"""
Simple test for tuned RFH3 with problematic cases
"""

import time
from rfh3_tuned import RFH3Tuned, RFH3Config

def test_simple():
    """Test problematic cases"""
    
    config = RFH3Config()
    config.max_iterations = 100000
    config.hierarchical_search = True
    config.learning_enabled = True
    
    rfh3 = RFH3Tuned(config)
    
    # Test cases that were failing
    test_cases = [
        (39919, 191, 209),       # Was finding 11 × 3629
        (1095235506161, 1046527, 1046543),  # Was finding 137 × 7994419753
        (281475278503913, 16769023, 16785431),  # Was finding 13 × 21651944500301
    ]
    
    print("Testing problematic cases:")
    print("-" * 60)
    
    for n, p_true, q_true in test_cases:
        print(f"\nTesting n = {n}")
        print(f"Expected: {p_true} × {q_true}")
        
        try:
            start = time.time()
            p_found, q_found = rfh3.factor(n, timeout=30.0)
            elapsed = time.time() - start
            
            print(f"Found: {p_found} × {q_found}")
            print(f"Time: {elapsed:.3f}s")
            
            if {p_found, q_found} == {p_true, q_true}:
                print("✓ SUCCESS")
            else:
                print("✗ FAILED - Found different factors")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    print("\n" + "-" * 60)
    rfh3.print_stats()

if __name__ == "__main__":
    test_simple()
