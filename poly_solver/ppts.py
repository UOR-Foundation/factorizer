"""
Prime Polynomial-Time Solver (PPTS) - Main Implementation

Polynomial-time factorization algorithm based on harmonic analysis,
adelic constraints, and Lie algebra deformations.
"""

import math
import time
import logging
from typing import Tuple, Optional, Dict, Any

from .harmonic import extract_harmonic_signature, MultiScaleResonance
from .adelic import construct_adelic_system, AdelicFilter, verify_adelic_balance
from .polynomial import construct_polynomial_system, solve_polynomial_system


class PPTS:
    """
    Prime Polynomial-Time Solver
    
    Implements polynomial-time integer factorization through:
    1. Multi-scale harmonic signature extraction
    2. Adelic constraint system construction
    3. Polynomial system generation
    4. Root finding to recover factors
    """
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = self._setup_logging(log_level)
        self.analyzer = MultiScaleResonance()
        self.stats = {
            'factorizations': 0,
            'total_time': 0.0,
            'phase_times': {'signature': 0.0, 'adelic': 0.0, 'polynomial': 0.0, 'solving': 0.0},
            'success_rate': 1.0
        }
    
    def _setup_logging(self, level: int) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('PPTS')
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def factor(self, n: int) -> Tuple[int, int]:
        """
        Factor integer n using polynomial-time algorithm
        
        Args:
            n: Integer to factor (must be composite)
            
        Returns:
            (p, q) where p <= q and p * q = n
            
        Raises:
            ValueError: If n is not a valid composite number
        """
        # Input validation
        if n < 4:
            raise ValueError("n must be >= 4")
        
        if self._is_prime(n):
            raise ValueError(f"{n} appears to be prime")
        
        start_time = time.time()
        self.stats['factorizations'] += 1
        
        self.logger.info(f"Starting PPTS factorization of {n} ({n.bit_length()} bits)")
        
        try:
            # Phase 1: Extract harmonic signature
            phase_start = time.time()
            signature = extract_harmonic_signature(n)
            self.stats['phase_times']['signature'] += time.time() - phase_start
            self.logger.debug(f"Harmonic signature extracted: trace={signature.trace():.4f}")
            
            # Phase 2: Construct adelic constraints
            phase_start = time.time()
            adelic_system = construct_adelic_system(n, signature)
            self.stats['phase_times']['adelic'] += time.time() - phase_start
            self.logger.debug(f"Adelic system constructed: {len(adelic_system.p_adic_constraints)} constraints")
            
            # Phase 3: Build polynomial system
            phase_start = time.time()
            poly_system = construct_polynomial_system(n, adelic_system)
            self.stats['phase_times']['polynomial'] += time.time() - phase_start
            
            if poly_system.polynomials:
                degree = poly_system.polynomials[0].degree
                self.logger.debug(f"Polynomial system constructed: degree {degree}")
            
            # Phase 4: Solve for factors
            phase_start = time.time()
            result = solve_polynomial_system(poly_system, n)
            self.stats['phase_times']['solving'] += time.time() - phase_start
            
            if result:
                p, q = result
                if p > q:
                    p, q = q, p
                
                # Verify the result
                if p * q == n:
                    total_time = time.time() - start_time
                    self.stats['total_time'] += total_time
                    self.logger.info(f"SUCCESS: {n} = {p} × {q} (time: {total_time:.3f}s)")
                    return (p, q)
                else:
                    raise ValueError(f"Invalid factorization: {p} × {q} ≠ {n}")
            else:
                raise ValueError("No valid factors found in polynomial roots")
                
        except Exception as e:
            self.logger.error(f"Factorization failed: {str(e)}")
            # Update success rate
            self.stats['success_rate'] = (self.stats['factorizations'] - 1) / self.stats['factorizations']
            raise
    
    def _is_prime(self, n: int) -> bool:
        """Quick primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Check small primes
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        # Miller-Rabin test for larger numbers
        if n < 2000:
            return True
        
        # Simplified Miller-Rabin
        d = n - 1
        r = 0
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Test with a few witnesses
        for a in [2, 3, 5, 7, 11]:
            if a >= n:
                continue
            
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = dict(self.stats)
        
        if stats['factorizations'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['factorizations']
            
            # Average phase times
            avg_phase_times = {}
            for phase, time in stats['phase_times'].items():
                avg_phase_times[phase] = time / stats['factorizations']
            stats['avg_phase_times'] = avg_phase_times
        
        return stats
    
    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.get_statistics()
        
        print("\nPPTS Performance Statistics")
        print("-" * 40)
        print(f"Total factorizations: {stats['factorizations']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        
        if stats['factorizations'] > 0:
            print(f"Average time: {stats['avg_time']:.4f}s")
            print("\nAverage phase times:")
            for phase, time in stats['avg_phase_times'].items():
                percentage = (time / stats['avg_time']) * 100 if stats['avg_time'] > 0 else 0
                print(f"  {phase}: {time:.4f}s ({percentage:.1f}%)")


def factor_polynomial_time(n: int) -> Tuple[int, int]:
    """
    Main polynomial-time factorization function
    
    Args:
        n: Integer to factor (must be composite)
        
    Returns:
        (p, q) where p <= q and p * q = n
    """
    solver = PPTS()
    return solver.factor(n)


# Example usage and verification
if __name__ == "__main__":
    # Test cases
    test_cases = [
        35,      # 5 × 7
        91,      # 7 × 13
        143,     # 11 × 13
        323,     # 17 × 19
        1073,    # 29 × 37
        10403,   # 101 × 103
    ]
    
    print("PPTS Verification Tests")
    print("=" * 50)
    
    solver = PPTS(log_level=logging.INFO)
    
    for n in test_cases:
        try:
            print(f"\nFactoring {n}...")
            p, q = solver.factor(n)
            print(f"✓ {n} = {p} × {q}")
            
            # Verify adelic balance
            balance = verify_adelic_balance(n, p)
            print(f"  Adelic balance score: {balance:.6f}")
            
        except Exception as e:
            print(f"✗ Failed to factor {n}: {e}")
    
    print("\n" + "=" * 50)
    solver.print_statistics()
