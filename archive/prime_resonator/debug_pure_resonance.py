"""
Debug script for Pure Resonance Factorizer
Analyzes why resonance detection is failing
"""

import numpy as np
import matplotlib.pyplot as plt
from pure_resonance import PureResonanceFactorizer
import math


def analyze_resonance_failure(n, p, q):
    """Analyze why resonance detection fails for a given semiprime."""
    print(f"\nAnalyzing n = {n} = {p} × {q}")
    print(f"Bit length: {n.bit_length()}")
    
    factorizer = PureResonanceFactorizer()
    
    # Generate signals
    signals = factorizer._generate_multi_domain_signals(n)
    
    # Get coherence map
    coherence_map = factorizer._compute_phase_coherence(signals, n)
    
    # Get positions
    sqrt_n = int(math.sqrt(n))
    num_positions = min(int(math.log(n) ** 2), 1000)
    positions = factorizer._golden_ratio_positions(sqrt_n, num_positions)
    
    # Find where p and q should be in our positions
    p_idx = min(range(len(positions)), key=lambda i: abs(positions[i] - p)) if p <= sqrt_n else -1
    q_idx = min(range(len(positions)), key=lambda i: abs(positions[i] - q)) if q <= sqrt_n else -1
    
    # Find max coherence
    max_idx = np.argmax(coherence_map)
    max_coherence = coherence_map[max_idx]
    max_position = positions[max_idx]
    
    print(f"\nResults:")
    print(f"  Max coherence: {max_coherence:.6f} at position {max_position}")
    print(f"  Expected factors: p={p}, q={q}")
    print(f"  Actual factor p={p} {'is' if p_idx >= 0 else 'NOT'} in positions")
    if p_idx >= 0:
        print(f"    Coherence at p: {coherence_map[p_idx]:.6f} (rank {sorted(coherence_map, reverse=True).index(coherence_map[p_idx]) + 1})")
    
    # Check individual signal coherences at the true factor
    if p <= sqrt_n:
        print(f"\nCoherence breakdown at p={p}:")
        test_pos = p
        
        # Prime signal coherence
        if 'log_prime' in signals and len(signals['log_prime']) > 0:
            prime_coh = factorizer._prime_signal_coherence(signals['log_prime'], n, test_pos)
            print(f"  Prime signal: {prime_coh:.6f}")
        
        # Phase coherence
        if 'phase' in signals and len(signals['phase']) > 0:
            phase_coh = factorizer._phase_alignment_coherence(signals['phase'], n, test_pos)
            print(f"  Phase alignment: {phase_coh:.6f}")
        
        # Autocorrelation coherence
        if 'autocorr' in signals and len(signals['autocorr']) > 0:
            auto_coh = factorizer._autocorr_coherence(signals['autocorr'], n, test_pos)
            print(f"  Autocorrelation: {auto_coh:.6f}")
        
        # Wavelet coherence
        if 'wavelet' in signals and len(signals['wavelet']) > 0:
            wavelet_coh = factorizer._wavelet_coherence(signals['wavelet'], n, test_pos)
            print(f"  Wavelet: {wavelet_coh:.6f}")
        
        # Entropy coherence
        if 'entropy' in signals and len(signals['entropy']) > 0:
            entropy_coh = factorizer._entropy_coherence(signals['entropy'], n, test_pos)
            print(f"  Entropy: {entropy_coh:.6f}")
    
    # Plot coherence map
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(positions, coherence_map)
    if p_idx >= 0:
        plt.axvline(p, color='r', linestyle='--', label=f'p={p}')
    if q_idx >= 0:
        plt.axvline(q, color='g', linestyle='--', label=f'q={q}')
    plt.axvline(max_position, color='b', linestyle=':', label=f'Max @ {max_position}')
    plt.xlabel('Position')
    plt.ylabel('Coherence')
    plt.title(f'Coherence Map for {n.bit_length()}-bit number')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in near sqrt(n)
    plt.subplot(1, 2, 2)
    zoom_range = int(sqrt_n * 0.2)
    zoom_indices = [i for i, pos in enumerate(positions) if abs(pos - sqrt_n) < zoom_range]
    if zoom_indices:
        zoom_positions = [positions[i] for i in zoom_indices]
        zoom_coherence = [coherence_map[i] for i in zoom_indices]
        plt.plot(zoom_positions, zoom_coherence)
        plt.axvline(sqrt_n, color='k', linestyle=':', alpha=0.5, label='sqrt(n)')
        if p <= sqrt_n + zoom_range:
            plt.axvline(p, color='r', linestyle='--', label=f'p={p}')
        plt.xlabel('Position')
        plt.ylabel('Coherence')
        plt.title('Zoomed near sqrt(n)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'resonance_debug_{n.bit_length()}bit.png')
    plt.close()
    
    return max_position == p or max_position == q


def main():
    """Analyze failures for different bit sizes."""
    test_cases = [
        (65537, 4294967311),  # 64-bit (but shown as 49-bit in output?)
        (7125766127, 6958284019),  # 66-bit
        (14076040031, 15981381943),  # 68-bit
    ]
    
    successes = 0
    for p, q in test_cases:
        n = p * q
        if analyze_resonance_failure(n, p, q):
            successes += 1
    
    print(f"\n\nSummary: {successes}/{len(test_cases)} successful detections")
    
    # Additional analysis: What makes a good resonance signal?
    print("\n\nSignal Analysis:")
    for p, q in test_cases[:1]:  # Just analyze first case in detail
        n = p * q
        factorizer = PureResonanceFactorizer()
        
        print(f"\nFor n = {n}:")
        print(f"  sqrt(n) = {int(math.sqrt(n))}")
        print(f"  p = {p}, q = {q}")
        print(f"  p/sqrt(n) = {p/math.sqrt(n):.6f}")
        print(f"  Golden ratio positions near p:")
        
        sqrt_n = int(math.sqrt(n))
        positions = factorizer._golden_ratio_positions(sqrt_n, 100)
        near_p = sorted(positions, key=lambda x: abs(x - p))[:5]
        for pos in near_p:
            print(f"    {pos} (distance: {abs(pos - p)})")


if __name__ == "__main__":
    main()
