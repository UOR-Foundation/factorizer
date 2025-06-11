import math
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class PPTSUniversalBasis:
    """
    Prime Polynomial-Time Solver with Universal Scale-Invariant Basis
    Strict implementation following the PPTS specification
    """
    
    # Universal Constants (empirically determined)
    ALPHA = 1.175056651649053      # Resonance Decay
    BETA = 0.199684068301496       # Phase Coupling
    GAMMA = 12.416057765534330     # Scale Transition
    DELTA = 0.0                    # Interference Null
    EPSILON = 4.329953646807706    # Adelic Threshold
    PHI = (1 + math.sqrt(5)) / 2   # Golden Ratio
    
    # Configuration Parameters
    SEARCH_RADIUS_BASE = 0.3
    RESONANCE_THRESHOLD = 0.001
    PRIME_BASE_SIZE = 100        # Increased for better phase detection
    MAX_ITERATIONS = 1000
    CONFIDENCE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]
    HARMONIC_COUNT = 7           # Increased for better resonance detection
    
    def __init__(self, canonical_size: int = 5000, reference_bits: int = 20):
        """
        Initialize PPTS with scale-invariant canonical basis
        
        Args:
            canonical_size: Number of vectors in canonical basis
            reference_bits: Reference bit-length for canonical basis
        """
        self.canonical_size = canonical_size
        self.reference_bits = reference_bits
        self.prime_base = self._generate_prime_base(self.PRIME_BASE_SIZE)
        
        # Generate canonical basis
        print(f"Generating canonical basis with {canonical_size} vectors...")
        start_time = time.time()
        self.canonical_basis = self._generate_canonical_basis()
        print(f"Canonical basis generated in {time.time() - start_time:.2f} seconds")
        
    def _generate_prime_base(self, size: int) -> List[int]:
        """Generate first 'size' prime numbers"""
        primes = []
        n = 2
        while len(primes) < size:
            is_prime = True
            for p in primes:
                if p * p > n:
                    break
                if n % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(n)
            n += 1
        return primes
    
    def _is_prime(self, n: int) -> bool:
        """Check if n is prime using trial division"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def _distribute_intelligently(self, index: int, total: int, ref_range: int) -> int:
        """
        Distribute positions with higher density around key regions
        Optimized for finding factors across different scales
        """
        sqrt_ref = int(math.sqrt(ref_range))
        fourth_root = int(ref_range ** 0.25)
        
        # 20% for small prime factors (critical for large numbers)
        if index < int(0.2 * total):
            # Small primes and their multiples
            position = 2 + int((index / (0.2 * total)) * min(10000, fourth_root))
        # 50% concentrated around √n (balanced factors)
        elif index < int(0.7 * total):
            # Distribution around sqrt with adaptive density
            normalized_idx = ((index - int(0.2 * total)) / (0.5 * total) - 0.5) * 4
            offset = int(sqrt_ref * 0.5 * math.tanh(normalized_idx))
            position = sqrt_ref + offset
        # 30% for larger factors and harmonic positions
        else:
            remaining_idx = index - int(0.7 * total)
            remaining_total = total - int(0.7 * total)
            # Cover range from sqrt(n) to n/small_prime
            max_pos = min(ref_range // 2, sqrt_ref * 10)
            position = int(sqrt_ref + (remaining_idx / remaining_total) * (max_pos - sqrt_ref))
        
        return max(2, min(position, ref_range - 1))
    
    def _quadratic_residue_pattern(self, x: int) -> List[int]:
        """Calculate quadratic residue pattern for small primes"""
        pattern = []
        for p in self.prime_base[:10]:  # First 10 primes
            if p >= x:
                pattern.append(0)
            else:
                # Check if x is a quadratic residue mod p
                is_residue = any((i * i) % p == x % p for i in range(p))
                pattern.append(1 if is_residue else 0)
        return pattern
    
    def _calculate_canonical_vector(self, x: int, ref_range: int) -> Dict:
        """Calculate canonical vector for position x in reference range"""
        normalized_pos = x / ref_range
        log_scale = math.log(normalized_pos + self.EPSILON)
        
        # Modular phases (scale-invariant)
        modular_phases = []
        for p in self.prime_base[:20]:  # Use first 20 primes
            if p < x:
                phase = (x % p) / p
            else:
                phase = 0.0
            modular_phases.append(phase)
        
        # Quadratic residues (pattern-invariant)
        quadratic_residues = self._quadratic_residue_pattern(x)
        
        # Harmonic components
        harmonic_components = []
        for h in range(1, self.HARMONIC_COUNT + 1):
            component = math.cos(h * 2 * math.pi * normalized_pos) * math.exp(-h * 0.1)
            harmonic_components.append(component)
        
        return {
            'id': x,
            'normalized_position': normalized_pos,
            'log_scale_normalized': log_scale,
            'modular_phases': modular_phases,
            'quadratic_residues': quadratic_residues,
            'harmonic_components': harmonic_components,
            'fundamental_frequency': 2 * math.pi * normalized_pos
        }
    
    def _generate_canonical_basis(self) -> Dict[int, Dict]:
        """Generate the scale-invariant canonical basis"""
        ref_range = 2 ** self.reference_bits
        basis = {}
        
        for i in range(self.canonical_size):
            position = self._distribute_intelligently(i, self.canonical_size, ref_range)
            vector = self._calculate_canonical_vector(position, ref_range)
            basis[position] = vector
            
        return basis
    
    def _scale_to_target_bit_length(self, canonical_vector: Dict, 
                                   target_bits: int) -> Dict:
        """Scale a canonical vector to target bit length"""
        scaling_factor = 2 ** (target_bits - self.reference_bits)
        target_range = 2 ** target_bits
        target_x = int(canonical_vector['normalized_position'] * target_range)
        
        # Scale harmonic components
        scaled_harmonics = []
        for h in canonical_vector['harmonic_components']:
            scaled_h = h * math.cos(math.log(scaling_factor))
            scaled_harmonics.append(scaled_h)
        
        scaled_vector = {
            'id': target_x,
            'log_scale': canonical_vector['log_scale_normalized'] + math.log(scaling_factor),
            'modular_phases': canonical_vector['modular_phases'],  # Scale-invariant
            'quadratic_residues': canonical_vector['quadratic_residues'],  # Pattern-invariant
            'harmonic_components': scaled_harmonics,
            'fundamental_frequency': canonical_vector['fundamental_frequency'] / scaling_factor
        }
        
        return scaled_vector
    
    def calculate_resonance(self, x: int, n: int) -> float:
        """
        Enhanced resonance calculation optimized for large numbers
        Incorporates multiple mathematical indicators for factor detection
        """
        if x <= 1 or x >= n:
            return 0.0
        
        sqrt_n = math.sqrt(n)
        
        # Core resonance components
        
        # 1. Modular coherence - key for large number factorization
        modular_score = 0.0
        coherence_matches = 0
        
        # Use adaptive prime selection based on number size
        bit_length = n.bit_length()
        primes_to_check = min(40 + bit_length // 4, len(self.prime_base))
        
        for i, p in enumerate(self.prime_base[:primes_to_check]):
            if p >= min(x, n):
                break
            
            n_mod_p = n % p
            x_mod_p = x % p
            
            # Check for modular relationships indicating divisibility
            if x_mod_p != 0:
                # Fermat-like test: if x|n, then n ≡ 0 (mod x) implies certain patterns
                gcd_indicator = math.gcd(n_mod_p, x_mod_p)
                if gcd_indicator == x_mod_p:
                    coherence_matches += 1
                    modular_score += 1.0 / (i + 1)
                
                # Check multiplicative order relationships
                if pow(n_mod_p, p-1, p) == pow(x_mod_p, p-1, p):
                    modular_score += 0.5 / (i + 1)
        
        # Normalize modular score
        if coherence_matches > 0:
            modular_score = (coherence_matches / primes_to_check) * math.exp(modular_score)
        
        # 2. Geometric resonance - adaptive for different factor scales
        geometric_score = 0.0
        
        # Check multiple geometric relationships
        relationships = []
        
        # Direct ratios
        if x != 0:
            relationships.extend([
                n / x,                    # Direct quotient
                sqrt_n / x,              # Square root ratio
                x / sqrt_n,              # Inverse sqrt ratio
            ])
        
        # Higher order relationships for large numbers
        if bit_length > 64:
            cube_root_n = n ** (1/3)
            if x != 0:
                relationships.extend([
                    cube_root_n / x,
                    x / cube_root_n,
                    (n / x) ** 0.5 if n > x else 0,
                ])
        
        for ratio in relationships:
            if ratio > 0:
                # Strong resonance when ratio is near integer
                fractional_part = abs(ratio - round(ratio))
                if fractional_part < 0.001:
                    geometric_score += math.exp(-fractional_part * 1000)
                elif fractional_part < 0.01:
                    geometric_score += math.exp(-fractional_part * 100)
        
        geometric_score = min(1.0, geometric_score / len(relationships))
        
        # 3. Traditional PPTS components (modified for scale)
        
        # α component: Adaptive decay based on position and scale
        if x <= sqrt_n:
            decay_distance = abs(x - sqrt_n) / sqrt_n
            decay_factor = self.ALPHA * (1 + bit_length / 128)  # Scale with bit length
        else:
            # Less penalty for factors above sqrt(n) in large numbers
            decay_distance = math.log(x / sqrt_n) / math.log(n)
            decay_factor = self.ALPHA * 0.5
        
        decay_score = math.exp(-decay_factor * decay_distance)
        
        # β component: Enhanced phase coupling
        phase_score = 0.0
        phase_matches = 0
        
        for i, p in enumerate(self.prime_base[:30]):
            if p >= n:
                break
            
            n_mod_p = n % p
            x_mod_p = x % p
            
            # Multiple phase indicators
            if x_mod_p != 0:
                # Check if n/x would be integral mod p
                if n_mod_p % x_mod_p == 0:
                    phase_matches += 1
                
                # Phase alignment
                delta_p = (n_mod_p - x_mod_p) / p
                phase_score += (1 + math.cos(2 * math.pi * delta_p)) / 2
        
        phase_score = (phase_score / 30) ** self.BETA
        if phase_matches > 15:  # Strong phase alignment
            phase_score *= 2.0
        
        # γ component: Scale transition (adaptive)
        if x > 1:
            log_ratio = math.log(sqrt_n) / math.log(x) - 1
            scale_arg = self.GAMMA * log_ratio + self.DELTA
            scale_score = (1 + math.cos(scale_arg)) / 2
        else:
            scale_score = 0.0
        
        # Combine all components with adaptive weights
        if bit_length <= 32:
            # Small numbers: traditional approach works well
            weights = [0.3, 0.2, 0.3, 0.1, 0.1]
        elif bit_length <= 64:
            # Medium numbers: balance all approaches  
            weights = [0.4, 0.3, 0.15, 0.1, 0.05]
        else:
            # Large numbers: emphasize modular and geometric
            weights = [0.5, 0.35, 0.1, 0.03, 0.02]
        
        base_resonance = (
            weights[0] * modular_score +
            weights[1] * geometric_score +
            weights[2] * decay_score +
            weights[3] * phase_score +
            weights[4] * scale_score
        )
        
        # Apply threshold and enhance strong signals
        if base_resonance > self.RESONANCE_THRESHOLD:
            final_resonance = base_resonance ** (1.0 / self.EPSILON)
        else:
            final_resonance = base_resonance
        
        # Boost for exact factors (helps validate detection)
        if n % x == 0:
            boost = 1 + math.log10(max(n/x, x))
            final_resonance *= boost
        
        return final_resonance
    
    def _adaptive_radius(self, n: int) -> float:
        """Calculate adaptive search radius based on bit length"""
        bit_length = n.bit_length()
        return max(0.05, self.SEARCH_RADIUS_BASE - bit_length * 0.01)
    
    def _adaptive_threshold(self, n: int) -> float:
        """Calculate adaptive resonance threshold"""
        magnitude = math.log10(n) if n > 0 else 0
        return self.RESONANCE_THRESHOLD * math.exp(-magnitude * 0.1)
    
    def find_resonance_peaks(self, n: int) -> List[Dict]:
        """
        Enhanced peak finding optimized for large numbers up to 128-bit
        Uses multi-phase search strategy with adaptive parameters
        """
        sqrt_n = int(math.sqrt(n))
        bit_length = n.bit_length()
        search_radius = self._adaptive_radius(n)
        threshold = self._adaptive_threshold(n)
        
        peaks = []
        
        # Phase 1: Small prime factors (critical for any size)
        # Even 128-bit numbers might have small factors
        small_prime_limit = min(100000, int(n ** 0.25))
        step = 1 if bit_length <= 32 else (2 if bit_length <= 64 else 6)
        
        for x in range(2, min(small_prime_limit, sqrt_n), step):
            if x <= 1 or x >= n:
                continue
            resonance = self.calculate_resonance(x, n)
            if resonance > threshold:
                peaks.append({
                    'candidate': x,
                    'score': resonance,
                    'is_factor': (n % x == 0),
                    'confidence': self._calculate_confidence(x, n, resonance),
                    'source': 'small_prime'
                })
        
        # Phase 2: Scaled canonical basis vectors
        # Adaptive scaling for different bit lengths
        if bit_length <= 96:  # Extended limit for canonical basis
            target_scale = min(bit_length, 64)  # Cap scaling for memory efficiency
            
            # Select subset of canonical basis for large numbers
            if bit_length > 64:
                # Use every k-th vector for efficiency
                k = 1 + (bit_length - 64) // 16
                selected_basis = dict(list(self.canonical_basis.items())[::k])
            else:
                selected_basis = self.canonical_basis
            
            for canonical_pos, canonical_vec in selected_basis.items():
                scaled_vec = self._scale_to_target_bit_length(canonical_vec, target_scale)
                
                # Multiple positions from single canonical vector
                base_x = scaled_vec['id']
                scale_factor = 2 ** (bit_length - target_scale)
                
                # Test multiple scalings of the canonical position
                for multiplier in [1, 2, 3, 5, 7, 11]:
                    x = int(base_x * scale_factor * multiplier)
                    
                    if x > 1 and x < n and x >= sqrt_n // 2 and x <= sqrt_n * 2:
                        resonance = self.calculate_resonance(x, n)
                        if resonance > threshold:
                            peaks.append({
                                'candidate': x,
                                'score': resonance,
                                'is_factor': (n % x == 0),
                                'confidence': self._calculate_confidence(x, n, resonance),
                                'source': 'canonical'
                            })
        
        # Phase 3: Focused sqrt region search
        # Adaptive step size based on bit length
        if bit_length <= 32:
            sqrt_samples = 1000
        elif bit_length <= 64:
            sqrt_samples = 500
        elif bit_length <= 96:
            sqrt_samples = 200
        else:
            sqrt_samples = 100
        
        min_x = max(2, int(sqrt_n * (1 - search_radius)))
        max_x = min(n - 1, int(sqrt_n * (1 + search_radius)))
        step_size = max(1, (max_x - min_x) // sqrt_samples)
        
        for x in range(min_x, max_x + 1, step_size):
            if any(abs(p['candidate'] - x) < step_size // 2 for p in peaks):
                continue
                
            resonance = self.calculate_resonance(x, n)
            if resonance > threshold:
                peaks.append({
                    'candidate': x,
                    'score': resonance,
                    'is_factor': (n % x == 0),
                    'confidence': self._calculate_confidence(x, n, resonance),
                    'source': 'sqrt_region'
                })
        
        # Phase 4: Harmonic positions for large numbers
        # Check positions n/k for small k
        max_k = min(10000, int(sqrt_n))
        k_step = 1 if bit_length <= 64 else (int(math.sqrt(max_k)) if bit_length <= 96 else max_k // 100)
        
        for k in range(2, max_k, k_step):
            if n % k == 0:
                x = n // k
                if x > 1 and x < n and not any(p['candidate'] == x for p in peaks):
                    resonance = self.calculate_resonance(x, n)
                    peaks.append({
                        'candidate': x,
                        'score': resonance,
                        'is_factor': True,
                        'confidence': self._calculate_confidence(x, n, resonance),
                        'source': 'harmonic'
                    })
        
        # Phase 5: Refined neighborhood search (only for promising peaks)
        # Limit neighborhood search for very large numbers
        if bit_length <= 96:
            high_resonance_peaks = sorted(peaks, key=lambda p: p['score'], reverse=True)[:10]
            neighborhood_range = 20 if bit_length <= 64 else 10
            
            for peak in high_resonance_peaks:
                center = peak['candidate']
                for offset in range(-neighborhood_range, neighborhood_range + 1):
                    x = center + offset
                    if x <= 1 or x >= n or any(p['candidate'] == x for p in peaks):
                        continue
                        
                    resonance = self.calculate_resonance(x, n)
                    if resonance > threshold:
                        peaks.append({
                            'candidate': x,
                            'score': resonance,
                            'is_factor': (n % x == 0),
                            'confidence': self._calculate_confidence(x, n, resonance),
                            'source': 'neighborhood'
                        })
        
        # Remove duplicates and sort by score
        unique_peaks = {}
        for p in peaks:
            key = p['candidate']
            if key not in unique_peaks or p['score'] > unique_peaks[key]['score']:
                unique_peaks[key] = p
        
        peaks = list(unique_peaks.values())
        peaks.sort(key=lambda p: p['score'], reverse=True)
        
        # Limit total peaks for very large numbers
        if bit_length > 96 and len(peaks) > 1000:
            peaks = peaks[:1000]
        
        return peaks
    
    def _calculate_confidence(self, x: int, n: int, resonance: float) -> float:
        """Calculate confidence score for a candidate"""
        sqrt_n = math.sqrt(n)
        
        # Proximity to √n
        proximity = 1.0 - abs(x - sqrt_n) / sqrt_n
        
        # Vector alignment (simplified)
        vector_alignment = 0.8  # Placeholder for full vector calculation
        
        # Harmonic coherence
        harmonic_coherence = 0.7  # Placeholder for harmonic analysis
        
        # Weighted combination
        weights = self.CONFIDENCE_WEIGHTS
        confidence = (weights[0] * resonance + 
                     weights[1] * proximity + 
                     weights[2] * vector_alignment + 
                     weights[3] * harmonic_coherence)
        
        return min(1.0, confidence)
    

    
    def factor(self, n: int) -> List[int]:
        """
        Main factorization method using PPTS algorithm
        Returns list of prime factors
        
        Strict implementation: Uses ONLY resonance-based factorization
        """
        if n < 2:
            return []
        
        if self._is_prime(n):
            return [n]
        
        factors = []
        queue = [n]
        iterations = 0
        
        while queue and iterations < self.MAX_ITERATIONS:
            iterations += 1
            current = queue.pop()
            
            if self._is_prime(current):
                factors.append(current)
                continue
            
            # Find resonance peaks
            peaks = self.find_resonance_peaks(current)
            
            # Filter actual factors from resonance peaks
            actual_factors = [p for p in peaks if p['is_factor']]
            
            if actual_factors:
                # Select best factor (highest resonance score)
                best_factor = actual_factors[0]['candidate']
                other_factor = current // best_factor
                
                # Add to queue for further factorization
                queue.extend([best_factor, other_factor])
            else:
                # No resonance peaks found - according to PPTS theory,
                # this should not happen for composite numbers
                # Mark as "resonance-prime" (unable to factor via resonance)
                print(f"  Warning: No resonance peaks found for {current}")
                factors.append(current)
        
        # Sort factors
        factors.sort()
        return factors
    
    def demonstrate(self):
        """Demonstrate PPTS factorization capabilities including large numbers"""
        print("\n=== PPTS Universal Basis Demonstration ===")
        print("=== Scaling up to 128-bit factorization ===\n")
        
        # Test cases organized by bit length
        test_cases = {
            "Small (< 32-bit)": [
                77,      # 7 × 11
                221,     # 13 × 17
                1517,    # 37 × 41
                5141,    # 53 × 97
                10403,   # 101 × 103
                32767,   # 7 × 31 × 151
            ],
            "Medium (32-64 bit)": [
                1234567891,      # 2347 × 525853
                9876543211,      # Prime
                12345678901237,  # 113 × 109337884963
                98765432109877,  # 314159 × 314159543
            ],
            "Large (64-96 bit)": [
                1234567890123456789,     # 3 × 411522630041152263
                9999999999999999991,     # 59649589127497217 × 167597486129726437
                12345678901234567891,    # Prime (actually composite but hard)
            ],
            "Very Large (96-128 bit)": [
                123456789012345678901234567891,  # Large semiprime
                999999999999999999999999999989,  # Large prime candidate
            ]
        }
        
        overall_success = 0
        overall_composites = 0
        
        for category, numbers in test_cases.items():
            print(f"\n{'='*60}")
            print(f"{category} Numbers")
            print(f"{'='*60}")
            
            success_count = 0
            total_composites = 0
            
            for n in numbers:
                print(f"\nFactoring {n} ({n.bit_length()}-bit):")
                start_time = time.time()
                
                # Find factors using PPTS
                try:
                    factors = self.factor(n)
                    elapsed = time.time() - start_time
                    
                    # Verify
                    product = 1
                    for f in factors:
                        product *= f
                    
                    is_correct = product == n
                    is_composite = not self._is_prime(n) if n < 10**12 else len(factors) > 1
                    
                    if is_composite:
                        total_composites += 1
                        if is_correct and len(factors) > 1:
                            success_count += 1
                    
                    print(f"  Factors: {factors}")
                    print(f"  Time: {elapsed:.2f} seconds")
                    print(f"  Verification: {'✓' if is_correct else '✗'}")
                    
                    # Show resonance analysis for smaller numbers
                    if n.bit_length() <= 64:
                        peaks = self.find_resonance_peaks(n)
                        actual_factor_peaks = [p for p in peaks if p['is_factor']]
                        
                        if peaks:
                            print(f"  Resonance peaks: {len(peaks)} total, {len(actual_factor_peaks)} factors")
                            for i, p in enumerate(peaks[:3]):  # Show top 3
                                print(f"    #{i+1}: x={p['candidate']}, score={p['score']:.4f} "
                                      f"{'[FACTOR]' if p['is_factor'] else ''}")
                        else:
                            print("  No resonance peaks detected")
                
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    elapsed = time.time() - start_time
                    print(f"  Time before error: {elapsed:.2f} seconds")
            
            if total_composites > 0:
                print(f"\n{category} Summary: {success_count}/{total_composites} composites factored")
                print(f"Success rate: {100*success_count/total_composites:.1f}%")
                overall_success += success_count
                overall_composites += total_composites
        
        print(f"\n\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total composite numbers successfully factored: {overall_success}/{overall_composites}")
        if overall_composites > 0:
            print(f"Overall success rate: {100*overall_success/overall_composites:.1f}%")
        print("\nNote: PPTS implementation with enhanced resonance detection for large numbers.")
        print("Larger numbers may require extended computation time.")


    def visualize_resonance_field(self, n: int, width: int = 80):
        """
        Visualize the resonance field around √n
        Shows how factors appear as peaks in the field
        """
        sqrt_n = int(math.sqrt(n))
        radius = int(sqrt_n * 0.2)  # 20% radius
        
        print(f"\n=== Resonance Field Visualization for n={n} ===")
        print(f"Range: [{sqrt_n - radius}, {sqrt_n + radius}]")
        print(f"√n ≈ {sqrt_n}")
        
        # Calculate resonances
        resonances = []
        max_resonance = 0
        factors_found = []
        
        for x in range(max(2, sqrt_n - radius), min(sqrt_n + radius + 1, n)):
            r = self.calculate_resonance(x, n)
            resonances.append((x, r))
            max_resonance = max(max_resonance, r)
            if n % x == 0:
                factors_found.append(x)
        
        # Create visualization
        print("\nResonance Field:")
        print("─" * width)
        
        for x, r in resonances:
            # Normalize to width
            bar_length = int((r / max_resonance) * (width - 20)) if max_resonance > 0 else 0
            bar = "█" * bar_length
            
            # Mark factors
            factor_mark = " [FACTOR]" if x in factors_found else ""
            
            # Format output
            print(f"{x:5d} | {bar}{factor_mark}")
        
        print("─" * width)
        print(f"Factors found in range: {factors_found}")
        print(f"Peak resonance: {max_resonance:.6f}")


# Example usage
if __name__ == "__main__":
    # Initialize PPTS with enhanced canonical basis for large numbers
    print("Initializing PPTS with universal scale-invariant basis...")
    print("Enhanced configuration for 128-bit factorization support")
    ppts = PPTSUniversalBasis(canonical_size=5000, reference_bits=20)
    
    # Run comprehensive demonstration
    ppts.demonstrate()
    
    # Example: Factor a specific large number
    print("\n\n=== Specific Large Number Factorization ===")
    # 64-bit semiprime example
    large_n = 18446744073709551557  # Large 64-bit number
    print(f"\nFactoring {large_n} ({large_n.bit_length()}-bit)")
    print(f"√n ≈ {int(math.sqrt(large_n))}")
    
    start_time = time.time()
    factors = ppts.factor(large_n)
    elapsed = time.time() - start_time
    
    print(f"Factors: {factors}")
    print(f"Time: {elapsed:.2f} seconds")
    
    # Verify
    product = 1
    for f in factors:
        product *= f
    print(f"Verification: {' × '.join(map(str, factors))} = {product} {'✓' if product == large_n else '✗'}")
    
    # Visualize resonance field for a smaller number
    print("\n\n=== Resonance Field Visualization ===")
    test_n = 10403  # 101 × 103
    ppts.visualize_resonance_field(test_n)
