"""
Debug tool for Prime Sieve - analyze why specific factors are missed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prime_sieve import PrimeSieve
from prime_sieve.core.prime_coordinate_system import PrimeCoordinateSystem
from prime_sieve.core.coherence_engine import CoherenceEngine
from prime_sieve.core.fibonacci_vortex import FibonacciVortex
from prime_sieve.core.interference_analyzer import InterferenceAnalyzer
from prime_sieve.core.quantum_sieve import QuantumSieve


def debug_factorization(n: int, expected_p: int, expected_q: int):
    """Debug why specific factors are not found."""
    print(f"\n{'='*60}")
    print(f"DEBUGGING: {n} = {expected_p} × {expected_q}")
    print(f"{'='*60}")
    
    # Initialize components
    coord_system = PrimeCoordinateSystem(n)
    coherence_engine = CoherenceEngine(n)
    vortex_engine = FibonacciVortex(n)
    interference_analyzer = InterferenceAnalyzer(n)
    quantum_sieve = QuantumSieve(n)
    
    # Create sieve instance
    sieve = PrimeSieve(enable_learning=False)
    
    # 1. Check if factors are in initial candidates
    print("\n1. CANDIDATE GENERATION:")
    candidates = sieve._generate_initial_candidates(
        coord_system, coherence_engine, vortex_engine, 
        interference_analyzer, n
    )
    
    print(f"   Total candidates generated: {len(candidates)}")
    print(f"   Expected factor {expected_p} in candidates: {expected_p in candidates}")
    print(f"   Expected factor {expected_q} in candidates: {expected_q in candidates}")
    
    # 2. Check coordinate alignment
    print("\n2. PRIME COORDINATE ANALYSIS:")
    if expected_p in candidates:
        alignment_p = coord_system.calculate_alignment(expected_p)
        print(f"   Alignment score for {expected_p}: {alignment_p.alignment_score:.6f}")
        print(f"   Zero alignments: {len(alignment_p.zero_alignments)}")
        print(f"   Pull value: {alignment_p.pull_value:.6f}")
    
    # 3. Check coherence
    print("\n3. COHERENCE ANALYSIS:")
    coherence = coherence_engine.calculate_coherence(expected_p, expected_q)
    print(f"   Coherence({expected_p}, {expected_q}): {coherence:.6f}")
    
    if expected_p in candidates:
        # Check field value
        field = coherence_engine.generate_coherence_field()
        field_value = field.get_value(expected_p)
        print(f"   Field value at {expected_p}: {field_value:.6f}")
    
    # 4. Check vortex properties
    print("\n4. FIBONACCI VORTEX ANALYSIS:")
    if expected_p in candidates:
        vortex_pull = vortex_engine._calculate_vortex_pull(expected_p)
        entanglement = vortex_engine._calculate_entanglement(expected_p)
        print(f"   Vortex pull at {expected_p}: {vortex_pull:.6f}")
        print(f"   Entanglement at {expected_p}: {entanglement:.6f}")
    
    # 5. Check interference
    print("\n5. INTERFERENCE ANALYSIS:")
    if expected_p in candidates:
        interference = interference_analyzer.calculate_interference(expected_p)
        print(f"   Interference at {expected_p}: {interference:.6f}")
    
    # 6. Apply dimensional sieves
    print("\n6. DIMENSIONAL SIEVE RESULTS:")
    filtered = sieve._apply_dimensional_sieves(
        candidates, coord_system, coherence_engine,
        vortex_engine, interference_analyzer, None
    )
    
    print(f"   Candidates after sieving: {len(filtered)}")
    print(f"   Expected factor {expected_p} survived: {expected_p in filtered}")
    print(f"   Expected factor {expected_q} survived: {expected_q in filtered}")
    
    # 7. Check quantum collapse
    print("\n7. QUANTUM COLLAPSE:")
    if filtered:
        refined = quantum_sieve.quantum_collapse(filtered)
        print(f"   Candidates after quantum collapse: {len(refined)}")
        print(f"   Expected factor {expected_p} in final: {expected_p in refined}")
        print(f"   Expected factor {expected_q} in final: {expected_q in refined}")
    
    # 8. Sample some candidates near the factors
    print("\n8. NEARBY CANDIDATES:")
    for delta in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        pos = expected_p + delta
        if pos in candidates:
            print(f"   Position {pos} (Δ={delta:+d}) is in candidates")
    
    # 9. Check why factors might be filtered out
    if expected_p not in candidates:
        print(f"\n9. WHY {expected_p} NOT IN CANDIDATES:")
        
        # Check each generation method
        sqrt_n = int(n**0.5)
        search_limit = min(sqrt_n, 100000)
        
        # Prime coordinates
        coord_aligned = coord_system.find_aligned_positions((2, search_limit))
        coord_positions = {a.position for a in coord_aligned[:500]}
        print(f"   In prime coordinate candidates: {expected_p in coord_positions}")
        
        # Coherence peaks
        field = coherence_engine.generate_coherence_field()
        peaks = set(field.get_peaks(0.3))
        print(f"   In coherence peaks: {expected_p in peaks}")
        
        # Vortex centers
        vortex_centers = vortex_engine.generate_vortex_centers()
        vortex_positions = {v.position for v in vortex_centers[:200]}
        print(f"   In vortex centers: {expected_p in vortex_positions}")
        
        # Interference extrema
        extrema = interference_analyzer.find_extrema()
        extrema_positions = {e.position for e in extrema[:200]}
        print(f"   In interference extrema: {expected_p in extrema_positions}")


if __name__ == "__main__":
    # Debug the failing case
    debug_factorization(1299709, 1117, 1163)
    
    # Also debug the other failing case
    debug_factorization(294409, 541, 544)
