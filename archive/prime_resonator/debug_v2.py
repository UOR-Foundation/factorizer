"""
Debug V2 to understand why resonance detection is failing
"""

import numpy as np
from pure_resonance_v2 import PureResonanceFactorizerV2
import math


def debug_v2_resonance(n, p, q):
    """Debug V2 resonance detection."""
    print(f"\nDebugging n = {n} = {p} × {q}")
    print(f"Bit length: {n.bit_length()}")
    print(f"sqrt(n) = {int(math.sqrt(n))}")
    print(f"p/sqrt(n) = {p/math.sqrt(n):.6f}")
    
    factorizer = PureResonanceFactorizerV2()
    
    # Generate signals
    signals = factorizer._generate_enhanced_signals(n)
    
    # Get resonance map
    resonance_map = factorizer._compute_interference_pattern(signals, n)
    
    # Check if factors are in the map
    p_in_map = p in resonance_map
    q_in_map = q in resonance_map
    
    print(f"\nFactor coverage:")
    print(f"  p={p} in resonance map: {p_in_map}")
    if p_in_map:
        print(f"    Resonance value: {resonance_map[p]:.6f}")
    print(f"  q={q} in resonance map: {q_in_map}")
    if q_in_map:
        print(f"    Resonance value: {resonance_map[q]:.6f}")
    
    # Find top resonances
    sorted_resonances = sorted(resonance_map.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 resonance positions:")
    for i, (pos, res) in enumerate(sorted_resonances[:10]):
        is_factor = n % pos == 0
        print(f"  {i+1}. pos={pos}, resonance={res:.6f} {'← FACTOR!' if is_factor else ''}")
    
    # Check sampling coverage
    sqrt_n = int(math.sqrt(n))
    num_samples = min(int(math.log2(n) ** 2), 1000)
    positions = factorizer._adaptive_full_range_sampling(sqrt_n, num_samples)
    
    print(f"\nSampling statistics:")
    print(f"  Total positions sampled: {len(positions)}")
    print(f"  Min position: {min(positions)}")
    print(f"  Max position: {max(positions)}")
    print(f"  Positions < 100000: {sum(1 for p in positions if p < 100000)}")
    
    # Find nearest sampled position to p
    if positions:
        nearest_to_p = min(positions, key=lambda x: abs(x - p))
        print(f"  Nearest position to p={p}: {nearest_to_p} (distance: {abs(nearest_to_p - p)})")
    
    # Analyze signal strengths at p (if sampled)
    if p_in_map:
        print(f"\nSignal analysis at p={p}:")
        
        # Find position index
        pos_list = sorted(resonance_map.keys())
        p_idx = pos_list.index(p) if p in pos_list else -1
        
        if p_idx >= 0:
            # Check each signal
            for signal_name, signal_data in signals.items():
                if len(signal_data) > 0:
                    # Map position to signal index
                    if signal_name == 'cross_correlation' and p_idx < len(signal_data):
                        value = signal_data[p_idx]
                        print(f"  {signal_name}: {value:.6f}")
                    elif signal_name == 'info_density' and p_idx < len(signal_data):
                        value = signal_data[p_idx]
                        print(f"  {signal_name}: {value:.6f}")


def main():
    """Debug test cases."""
    test_cases = [
        (65537, 4294967311),  # 49-bit
        (7125766127, 6958284019),  # 66-bit
    ]
    
    for p, q in test_cases:
        n = p * q
        debug_v2_resonance(n, p, q)
        print("\n" + "="*60)


if __name__ == "__main__":
    main()
