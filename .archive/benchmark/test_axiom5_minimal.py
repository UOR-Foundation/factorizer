"""
Minimal test of Axiom 5 breakthrough benchmark
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Starting minimal Axiom 5 test...")

try:
    from axiom5 import MetaObserver, SpectralMirror, get_meta_cache
    print("✓ Axiom 5 imports successful")
    
    from axiom1 import primes_up_to
    print("✓ Axiom 1 imports successful")
    
    # Test basic functionality
    cache = get_meta_cache()
    print("✓ Meta cache initialized")
    
    # Test simple factorization
    n = 143  # 11 * 13
    mirror = SpectralMirror(n)
    print(f"✓ SpectralMirror created for n={n}")
    
    # Test cache
    cache.add_observation({
        'n': n, 'position': 11, 'coherence': 1.0,
        'axiom': 'test', 'factor': True
    })
    
    obs = cache.query_observations(min_coherence=0.9)
    print(f"✓ Cache has {len(obs)} observations")
    
    # Test meta-observer
    meta_obs = MetaObserver(n)
    meta_obs.observe_observation(11, 1.0, 'test', True)
    patterns = meta_obs.detect_observation_patterns()
    print(f"✓ Meta-observer detected patterns: {list(patterns.keys())}")
    
    print("\nAll basic tests passed! Full benchmark should work.")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
