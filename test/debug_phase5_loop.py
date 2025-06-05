#!/usr/bin/env python3
# Debug script to identify Phase 5 loop issue

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultra_accelerated_uor_factorizer import (
    MultiScaleObserver, SpectralFolder, QuantumTunnel,
    sharp_fold_candidates, FoldTopology, interference_extrema,
    FibonacciEntanglement, ResonanceMemory, PrimeCascade,
    prime_fib_interference, identify_resonance_source,
    harmonic_amplify
)

# Test the semiprimes that hung
test_semiprimes = [1189, 988027]  # 29×41 and 991×997

for n in test_semiprimes:
    print(f"\n{'='*60}")
    print(f"Testing Phase 5 behavior for n = {n}")
    root = int(n**0.5)
    print(f"Root: {root}")
    
    # Phase 5 setup
    obs = MultiScaleObserver(n)
    folder = SpectralFolder(n)
    tun = QuantumTunnel()
    
    print(f"Observer scales: {obs.scales}")
    print(f"Spectral folder points: {folder.points[:10]}...")  # First 10 points
    
    # Simulate Phase 5 loop with debug output
    x, stuck = 2, 0
    iteration = 0
    max_iterations = 100  # Limit for debugging
    
    print(f"\nPhase 5 loop simulation:")
    while stuck < 80 and iteration < max_iterations:
        iteration += 1
        
        # Check if factor found
        if n % x == 0:
            print(f"  Factor found at x={x} (iteration {iteration})")
            break
        
        # Check coherence
        coh = obs.coherence(x)
        if coh > 0.9:
            print(f"  High coherence {coh:.4f} at x={x} (iteration {iteration})")
            h_amps = harmonic_amplify(n, x)
            print(f"    Harmonic amplifications: {h_amps}")
            for h in h_amps:
                if n % h == 0:
                    print(f"    Factor found via harmonic: {h}")
                    break
        
        # Get next position
        nxt = folder.next_after(x)
        
        # Track stuck counter
        if nxt == x:
            stuck += 1
        else:
            stuck = 0
        
        # Quantum tunnel if stuck
        if stuck >= 40:
            old_x = x
            x = tun.exit(n, nxt)
            stuck %= 40
            print(f"  Quantum tunnel from {old_x} to {x} (iteration {iteration})")
        else:
            x = nxt
        
        # Debug output every 10 iterations
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: x={x}, stuck={stuck}, coherence={coh:.4f}")
    
    if iteration >= max_iterations:
        print(f"\nLoop exceeded {max_iterations} iterations - likely infinite loop!")
    
    # Check what's happening with SpectralFolder
    print(f"\nSpectralFolder analysis:")
    print(f"Total folding points: {len(folder.points)}")
    print(f"Last few points: {folder.points[-5:]}")
    
    # Test next_after behavior near the end
    test_x = folder.points[-1] if folder.points else root
    print(f"\nTesting next_after near end:")
    for i in range(5):
        next_x = folder.next_after(test_x)
        print(f"  next_after({test_x}) = {next_x}")
        if next_x == test_x:
            print(f"    -> Stuck at {test_x}!")
            break
        test_x = next_x
