#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Pure UOR / Prime-Model  •  Ultra-Accelerated Factorizer V2 (axioms-only)
#
#  Axiom 1  Prime Ontology      → prime-space coordinates & primality checks
#  Axiom 2  Fibonacci Flow      → golden-ratio vortices & interference waves
#  Axiom 3  Duality Principle   → spectral (wave) vs. factor (particle) views
#  Axiom 4  Observer Effect     → adaptive, coherence-driven measurement
#
#  V2 Enhancements (all axiom-derived):
#     • Adaptive Observer           • Spectral Gradient Navigation
#     • Coherence Field Mapping     • Prime-Fibonacci Resonance Focus
#     • Quantum Superposition       • Hierarchical Coherence
#     • Wavefunction Collapse       • Axiom Phase Integration
# ---------------------------------------------------------------------------

import math, itertools
from typing import List, Tuple, Dict, Set, Optional

# ─────────────────────────────  constants & helpers  ────────────────────────
SQRT5, PHI = math.sqrt(5), (1 + math.sqrt(5)) / 2
PSI, GOLDEN_ANGLE = (1 - math.sqrt(5)) / 2, 2 * math.pi * (PHI - 1)
SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)

def primes_up_to(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = b"\x00" * len(range(p * p, limit + 1, p))
    return [i for i, f in enumerate(sieve) if f]

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d & 1 == 0:
        d //= 2; s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        a %= n
        if a == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def fib(k: int) -> int:
    if k < 0:
        raise ValueError("k must be non-negative")
    def _dbl(i: int) -> Tuple[int, int]:
        if i == 0:
            return (0, 1)
        a, b = _dbl(i >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        return (d, c + d) if i & 1 else (c, d)
    return _dbl(k)[0]

def fib_wave(x: float) -> float:
    return (PHI ** x - PSI ** x) / SQRT5

# ─────────────────────────  spectral tools  ────────────────────────────────
def binary_spectrum(n: int) -> List[float]:
    bits = bin(n)[2:]; L = len(bits)
    dens = bits.count("1") / L
    runs = [len(list(g)) / L for _, g in itertools.groupby(bits)]
    seq  = [1 if b == "1" else -1 for b in bits]; m = sum(seq) / L
    ac   = sum((s - m) * (t - m) for s, t in zip(seq, seq[1:])) / (L - 1) if L > 1 else 0
    return [dens, ac] + runs[:10]

def modular_spectrum(n: int, k: int = 10) -> List[float]:
    primes = primes_up_to(30)
    return [n % p / p for p in primes[1:min(len(primes), k + 1)]]

def digital_spectrum(n: int) -> List[float]:
    digits = list(map(int, str(n))); s = sum(digits)
    root = 1 + (s - 1) % 9 if s else 0
    return [s / (len(digits) * 9), root / 9]

def harmonic_spectrum(n: int) -> List[float]:
    x = math.log(max(n, 1), PHI)
    f = fib_wave(x)
    phase   = abs(f.real if isinstance(f, complex) else f) % 1
    nearest = fib(round(x))
    ratio   = math.log(nearest + 1) / math.log(n + 1)
    offset  = (n * GOLDEN_ANGLE / (2 * math.pi)) % 1
    return [phase, ratio, offset]

def spectral_vector(n: int) -> List[float]:
    return binary_spectrum(n) + modular_spectrum(n) + digital_spectrum(n) + harmonic_spectrum(n)

# ────────────────────  Enhanced Coherence (V2)  ─────────────────────────────
class CoherenceCache:
    """Cache coherence calculations to avoid redundant computations"""
    def __init__(self):
        self.cache: Dict[Tuple[int, int, int], float] = {}
        self.spectral_cache: Dict[int, List[float]] = {}
    
    def get_spectral(self, n: int) -> List[float]:
        if n not in self.spectral_cache:
            self.spectral_cache[n] = spectral_vector(n)
        return self.spectral_cache[n]
    
    def coherence(self, a: int, b: int, n: int) -> float:
        key = tuple(sorted([a, b, n]))
        if key not in self.cache:
            sa = self.get_spectral(a)
            sb = self.get_spectral(b)
            sn = self.get_spectral(n)
            diff2 = sum(((x + y) / 2 - s) ** 2 for x, y, s in zip(sa, sb, sn))
            self.cache[key] = math.exp(-diff2)
        return self.cache[key]

# Global coherence cache instance
_coherence_cache = CoherenceCache()

def coherence(a: int, b: int, n: int) -> float:
    """Cached coherence calculation"""
    return _coherence_cache.coherence(a, b, n)

# ───────────────  sharp-fold curvature  ──────────────────
def fold_energy(n: int, x: int) -> float:
    sx = _coherence_cache.get_spectral(x)
    sy = _coherence_cache.get_spectral(n // x)
    sn = _coherence_cache.get_spectral(n)
    return sum((a + b - c) ** 2 for a, b, c in zip(sx, sy, sn))

def sharp_fold_candidates(n: int, span: int = 25) -> List[int]:
    root = int(math.isqrt(n))
    window = range(max(2, root - span), min(root + span, n // 2) + 1)
    E = [fold_energy(n, x) for x in window]
    curv = [(E[i - 1] - 2 * E[i] + E[i + 1], x)
            for i, x in enumerate(list(window)[1:-1], 1)]
    curv.sort()
    return [x for _, x in curv[:10]]

# ─────────────  Prime × Fibonacci interference (peaks & dips)  ──────────────
def prime_fib_interference(n: int) -> List[float]:
    root = int(math.isqrt(n))
    primes = primes_up_to(min(root, 100))[:10]  # Limit primes for efficiency
    fibs   = [fib(k) for k in range(2, 20) if fib(k) <= root][:10]
    spec = []
    for x in range(2, root + 1):
        p_amp = sum(math.cos(2 * math.pi * p * x / n) for p in primes)
        f_amp = sum(math.cos(2 * math.pi * f * x / (n * PHI)) for f in fibs)
        spec.append(p_amp * f_amp)
    return spec

def interference_extrema(n: int, top: int = 30) -> List[int]:
    spec = prime_fib_interference(n)
    ext = []
    for i in range(1, len(spec) - 1):
        if (spec[i] - spec[i - 1]) * (spec[i + 1] - spec[i]) < 0:
            ext.append((abs(spec[i]), i + 2))
    ext.sort(reverse=True)
    return [x for _, x in ext[:top]]

# ──────────────────────  Resonance Memory  ─────────────────────────────────
class ResonanceMemory:
    def __init__(self):
        self.graph: Dict[Tuple[int, int], float] = {}
        self.success: List[Tuple[int, int, int, int]] = []
    def record(self, p: int, f: int, n: int, strength: float, factor: int = None):
        key = (p, f)
        self.graph[key] = 0.7 * self.graph.get(key, 0) + 0.3 * strength
        if factor:
            self.success.append((p, f, n, factor))
    def predict(self, n: int) -> List[Tuple[int, float]]:
        root, out = int(math.isqrt(n)), {}
        for p, f, prev_n, fact in self.success:
            scale = n / prev_n
            for pos, w in ((int(fact * scale), 0.8),
                           (int(fact * scale * PHI), 0.6)):
                if 2 <= pos <= root:
                    out[pos] = max(out.get(pos, 0), w)
            for (p2, f2), s in self.graph.items():
                if abs(p2 - p) <= 2 and abs(f2 - f) <= 1:
                    pos = (p2 * f2) % root or p2
                    out[pos] = max(out.get(pos, 0), s * 0.5)
        return sorted(out.items(), key=lambda t: -t[1])[:20]

def identify_resonance_source(x: int, n: int) -> Tuple[int, int]:
    root = int(math.isqrt(n))
    primes = [p for p in primes_up_to(min(root, 100)) if x % p] or [2]
    fibs   = [fib(k) for k in range(2, 12) if fib(k) <= root] or [2]
    best, bp, bf = 0, primes[0], fibs[0]
    for p in primes[:10]:  # Limit search
        for f in fibs[:10]:
            val = abs(math.cos(2 * math.pi * p * x / n)
                      * math.cos(2 * math.pi * f * x / (n * PHI)))
            if val > best:
                best, bp, bf = val, p, f
    return bp, bf

# ─────────────────────  Fibonacci vortices & Prime cascade  ────────────────
def fib_vortices(n: int) -> List[int]:
    root, vort, k = int(math.isqrt(n)), set(), 1
    while (f := fib(k)) < root and k < 30:  # Limit iterations
        vort.update({f, int(f * PHI), int(f / PHI)})
        for p in primes_up_to(min(100, f))[:20]:  # Limit primes
            vort.add((f * p) % root or p)
        k += 1
    return sorted(v for v in vort if 2 <= v <= root)

class PrimeCascade:
    def __init__(self, n: int):
        self.n = n
    def cascade(self, p: int) -> List[int]:
        out = []
        if is_prime(p + 2): out.append(p + 2)
        if p > 2 and is_prime(p - 2): out.append(p - 2)
        if is_prime(2 * p + 1): out.append(2 * p + 1)
        q = 2 * p + 1
        while is_prime(q) and q < self.n and len(out) < 10:  # Limit cascade
            out.append(q); q = 2 * q + 1
        return out

# ────────────────────  Spectral folding, geodesic, tunnelling  ─────────────
class SpectralFolder:
    def __init__(self, n: int):
        self.n, self.root = n, int(math.isqrt(n))
        self.points = self._build()
    def _build(self):
        folds = {2}
        for i in range(1, min(self.n.bit_length(), 64)):  # Limit bit length
            f = 2 ** i
            if f <= self.root:
                folds.add(f)
            if self.n // f >= 2:
                folds.add(self.n // f)
        for p in primes_up_to(min(100, self.root))[:20]:  # Limit primes
            t = 1
            while t < p and pow(10, t, p) != 1:
                t += 1
            if t < p:
                folds.update(range(t, min(self.root + 1, t * 20), t))
        return sorted(x for x in folds if 2 <= x <= self.root)
    def next_after(self, cur: int) -> int:
        for f in self.points:
            if f > cur:
                return f
        return 2

class PrimeGeodesic:
    def __init__(self, n: int):
        self.n = n
        self.coord = [n % p for p in primes_up_to(min(1000, n))[:100]]  # Limit
    def _pull(self, x: int) -> float:
        pull = 0.0
        for i, p in enumerate(primes_up_to(min(30, x))):
            if i < len(self.coord) and self.coord[i] == 0 and x % p == 0:
                pull += 1 / p
        return pull
    def walk(self, start: int, steps: int = 20) -> List[int]:
        path, cur = [start], start
        for _ in range(min(steps, 30)):  # Limit steps
            best, best_s = cur, 0
            for d in (-3, -2, -1, 1, 2, 3):
                cand = cur + d
                if 2 <= cand <= int(math.isqrt(self.n)):
                    s = self._pull(cand)
                    if s > best_s:
                        best, best_s = cand, s
            if best == cur:
                break
            cur = best; path.append(cur)
        return path

def harmonic_amplify(n: int, x: int) -> List[int]:
    root, out = int(math.isqrt(n)), set()
    for k in range(2, min(10, root // x + 1)):  # Limit range
        hx = (k * x) % root or k
        if 2 <= hx <= root:
            out.add(hx)
    phi_h = int(x * PHI) % root or int(x * PHI)
    if 2 <= phi_h <= root:
        out.add(phi_h)
    for p in primes_up_to(min(20, root))[:10]:  # Limit primes
        px = (x * p) % root or p
        if 2 <= px <= root:
            out.add(px)
    return sorted(out)[:20]  # Limit output

class QuantumTunnel:
    def exit(self, n: int, blocked: int, width: int = 60) -> int:
        root = int(math.isqrt(n))
        guess = min(root, blocked + width)
        cands = []
        k = 1
        while k < 30 and (f := fib(k)) < guess + 30:  # Limit iterations
            if f > guess:
                cands.append(f)
            k += 1
        for p in primes_up_to(min(guess + 60, root * 2))[:20]:  # Limit primes
            if p > guess:
                cands.append(p)
        return min(cands) if cands else guess

# ──────────────  Enhanced Adaptive Observer (V2)  ──────────────────────────
class AdaptiveObserver:
    """Implements true adaptive observation with quantum superposition"""
    def __init__(self, n: int):
        self.n = n
        self.root = int(math.isqrt(n))
        # Adaptive scales based on n's magnitude
        self.base_scales = self._compute_scales()
        self.coherence_field = {}
        self.gradient_cache = {}
        
    def _compute_scales(self) -> Dict[str, int]:
        """Compute axiom-based adaptive scales"""
        root = self.root
        return {
            "μ": 1,  # Micro: always 1
            "m": max(2, int(math.log(root, PHI))),  # Meso: logarithmic in PHI
            "M": max(3, int(root ** (1/PHI))),  # Macro: PHI root
            "Ω": max(2, fib(int(math.log(root, 2)))),  # Omega: Fibonacci scaled
        }
    
    def superposition_candidates(self, hint_positions: List[int] = None) -> List[int]:
        """Generate quantum superposition of candidate positions"""
        candidates = set()
        
        # Start with hint positions if provided
        if hint_positions:
            candidates.update(hint_positions)
        
        # Add Fibonacci positions
        k = 1
        while (f := fib(k)) <= self.root and k < 20:
            candidates.add(f)
            candidates.add(int(f * PHI) % self.root or 2)
            k += 1
        
        # Add positions near sqrt(n)
        for offset in range(-10, 11):
            pos = self.root + offset
            if 2 <= pos <= self.root:
                candidates.add(pos)
        
        # Add golden ratio spiral positions
        angle = 0
        for i in range(min(20, self.root // 10)):
            r = int(self.root * (i + 1) / 20)
            x = int(r * math.cos(angle)) + self.root // 2
            if 2 <= x <= self.root:
                candidates.add(x)
            angle += GOLDEN_ANGLE
        
        return sorted(candidates)
    
    def coherence_gradient(self, x: int, delta: int = 1) -> float:
        """Calculate coherence gradient at position x"""
        if x in self.gradient_cache:
            return self.gradient_cache[x]
        
        # Sample coherence at nearby points
        c0 = self.point_coherence(x)
        c_plus = self.point_coherence(min(x + delta, self.root))
        c_minus = self.point_coherence(max(x - delta, 2))
        
        # Gradient approximation
        gradient = (c_plus - c_minus) / (2 * delta)
        self.gradient_cache[x] = gradient
        return gradient
    
    def point_coherence(self, x: int) -> float:
        """Calculate coherence at a single point with caching"""
        if x in self.coherence_field:
            return self.coherence_field[x]
        
        total = 0.0
        scale_count = 0
        
        for scale_name, s in self.base_scales.items():
            # Adaptive sampling based on scale
            step = max(1, s // 3)  # Fewer samples for efficiency
            sample_count = 0
            scale_coherence = 0.0
            
            for offset in range(-s, s + 1, step):
                pos = x + offset
                if 2 <= pos <= self.root:
                    scale_coherence += coherence(pos,
                                               self.n // pos if self.n % pos == 0 else pos,
                                               self.n)
                    sample_count += 1
            
            if sample_count > 0:
                # Weight by inverse log of scale for axiom compliance
                weight = 1 / (1 + math.log(s))
                total += (scale_coherence / sample_count) * weight
                scale_count += 1
        
        result = total / scale_count if scale_count > 0 else 0.0
        self.coherence_field[x] = result
        return result
    
    def wavefunction_collapse(self, candidates: List[int], iterations: int = 10) -> List[int]:
        """Collapse quantum superposition using coherence measurements"""
        weights = {x: self.point_coherence(x) for x in candidates}
        
        # Iterative collapse based on coherence and gradients
        for i in range(min(iterations, 5)):  # Limit iterations
            new_weights = {}
            
            for x in candidates:
                # Combine coherence with gradient information
                gradient = self.coherence_gradient(x)
                
                # Move toward positive gradient
                if gradient > 0:
                    new_x = min(x + 1, self.root)
                elif gradient < 0:
                    new_x = max(x - 1, 2)
                else:
                    new_x = x
                
                # Update weight based on new position
                new_weights[new_x] = self.point_coherence(new_x) * (1 + abs(gradient))
            
            # Keep top candidates based on weights
            sorted_candidates = sorted(new_weights.items(), key=lambda t: -t[1])
            candidates = [x for x, _ in sorted_candidates[:max(10, len(candidates) // 2)]]
            
            if not candidates:
                break
        
        return candidates

# ────────────────────  Fibonacci Entanglement  ─────────────────────────────
class FibonacciEntanglement:
    def __init__(self, n: int):
        self.n, self.root = n, int(math.isqrt(n))
    def detect_double(self) -> List[Tuple[int, int, float]]:
        cand, k = [], 1
        while k < 30 and fib(k) < self.root:  # Limit iterations
            base = fib(k)
            for d in range(-5, 6):
                p = base + d
                if p > 1 and self.n % p == 0:
                    q = self.n // p
                    dist = min(abs(q - fib(j)) for j in range(1, min(30, int(math.log(q, PHI)) + 5)))
                    if dist < 0.1 * q:
                        cand.append((p, q, 1 / (1 + dist / q)))
            k += 1
        return sorted(cand, key=lambda t: -t[2])[:10]  # Limit results

# ────────────────────  Fold-Topology Navigation  ───────────────
class FoldTopology:
    def __init__(self, n: int):
        self.n, self.root = n, int(math.isqrt(n))
        self.points: List[int] = []
        self.conn: Dict[int, List[Tuple[int, float]]] = {}
        self._build()
    def _build(self):
        # Sample energies more efficiently
        sample_step = max(1, self.root // 100)  # Adaptive sampling
        sample_range = range(2, self.root + 1, sample_step)
        energies = {x: fold_energy(self.n, x) for x in sample_range}
        
        # Find local minima
        sample_list = list(sample_range)
        for i in range(1, len(sample_list) - 1):
            x = sample_list[i]
            prev_x = sample_list[i-1]
            next_x = sample_list[i+1]
            if energies[x] < energies[prev_x] and energies[x] < energies[next_x]:
                self.points.append(x)
        
        # Limit connections for efficiency
        for p1 in self.points[:20]:  # Limit points
            links = []
            for p2 in self.points[:20]:
                if p2 == p1:
                    continue
                # Simplified connection check
                energy_diff = abs(energies.get(p1, 0) - energies.get(p2, 0))
                if energy_diff < 10:  # Threshold for connection
                    links.append((p2, 1 / (1 + energy_diff)))
            self.conn[p1] = sorted(links, key=lambda t: -t[1])[:5]  # Top 5 connections
    
    def components(self) -> List[List[int]]:
        comps, unseen = [], set(self.points)
        while unseen and len(comps) < 5:  # Limit components
            comp, stack = [], [unseen.pop()]
            while stack and len(comp) < 20:  # Limit component size
                cur = stack.pop()
                comp.append(cur)
                for nb, _ in self.conn.get(cur, [])[:3]:  # Limit neighbors
                    if nb in unseen:
                        unseen.remove(nb)
                        stack.append(nb)
            comps.append(sorted(comp))
        return comps
    
    def traverse(self) -> List[int]:
        if not self.points:
            return []
        cur = self.points[0]  # Start from first point
        seen, path = set(), []
        max_steps = min(20, len(self.points))  # Limit traversal
        
        for _ in range(max_steps):
            if cur in seen:
                break
            seen.add(cur)
            path.append(cur)
            if self.n % cur == 0:
                return path
            neighbours = self.conn.get(cur, [])
            if neighbours:
                cur = neighbours[0][0]  # Take best neighbor
            else:
                break
        return path

# ──────────────────  Spectral Gradient Navigation (V2)  ────────────────────
class SpectralGradientNavigator:
    """Navigate factor space using spectral gradients"""
    def __init__(self, n: int, observer: AdaptiveObserver):
        self.n = n
        self.root = int(math.isqrt(n))
        self.observer = observer
        self.visited = set()
    
    def find_ascent_path(self, start: int, max_steps: int = 30) -> List[int]:
        """Follow gradient ascent to coherence peaks"""
        path = [start]
        current = start
        
        for _ in range(max_steps):
            if current in self.visited:
                break
            self.visited.add(current)
            
            # Check if factor found
            if self.n % current == 0:
                return path
            
            # Calculate gradient
            gradient = self.observer.coherence_gradient(current)
            
            # Determine step size based on gradient magnitude
            step = 1
            if abs(gradient) > 0.1:
                step = min(int(abs(gradient) * 10), 5)
            
            # Move in gradient direction
            if gradient > 0:
                next_pos = min(current + step, self.root)
            elif gradient < 0:
                next_pos = max(current - step, 2)
            else:
                # Zero gradient - try harmonic jump
                next_pos = int(current * PHI) % self.root or 2
            
            if next_pos == current:
                break
            
            current = next_pos
            path.append(current)
        
        return path
    
    def multi_path_search(self, seeds: List[int]) -> Optional[Tuple[int, int]]:
        """Search multiple gradient paths from seed positions"""
        for seed in seeds[:10]:  # Limit seeds
            path = self.find_ascent_path(seed)
            for pos in path:
                if self.n % pos == 0:
                    return (min(pos, self.n // pos), max(pos, self.n // pos))
        return None

# ───────────────────── Ultra-Accelerated UOR Factorizer V2  ───────────────────
def ultra_uor_factor_v2(n: int) -> Tuple[int, int]:
    """Enhanced factorizer with axiom-compliant optimizations"""
    if n <= 3 or is_prime(n):
        return (1, n)

    root = int(math.isqrt(n))
    mem = ResonanceMemory()
    
    # Clear caches for new factorization
    global _coherence_cache
    _coherence_cache = CoherenceCache()

    # Phase 0 – Fibonacci entanglement (fast check)
    ent = FibonacciEntanglement(n)
    for p, q, s in ent.detect_double():
        if s > 0.7:
            return (min(p, q), max(p, q))

    # Phase 1 – Sharp folds (local energy minima)
    for x in sharp_fold_candidates(n):
        if n % x == 0:
            return (min(x, n // x), max(x, n // x))

    # Phase 2 – Fold topology navigation
    topo = FoldTopology(n)
    for comp in topo.components()[:3]:
        for x in comp[:10]:  # Limit per component
            if n % x == 0:
                return (min(x, n // x), max(x, n // x))

    # Phase 3 – Interference extrema with resonance memory
    casc = PrimeCascade(n)
    spec = prime_fib_interference(n)
    for r in interference_extrema(n, top=20):  # Reduced from 30
        p_src, f_src = identify_resonance_source(r, n)
        strength = abs(spec[r - 2]) if r - 2 < len(spec) else 0
        if n % r == 0:
            mem.record(p_src, f_src, n, strength, r)
            return (min(r, n // r), max(r, n // r))
        mem.record(p_src, f_src, n, strength)
        for p in casc.cascade(r)[:5]:  # Limit cascade
            if p <= root and n % p == 0:
                return (min(p, n // p), max(p, n // p))

    # Phase 4 – Memory-guided positions
    for pos, _ in mem.predict(n):
        if n % pos == 0:
            return (min(pos, n // pos), max(pos, n // pos))

    # Phase 5 – Enhanced Adaptive Observer (V2)
    observer = AdaptiveObserver(n)
    
    # Generate initial superposition from earlier phase hints
    hint_positions = []
    hint_positions.extend(interference_extrema(n, top=10))
    hint_positions.extend([pos for pos, _ in mem.predict(n)[:10]])
    hint_positions.extend(fib_vortices(n)[:10])
    
    # Quantum superposition and collapse
    candidates = observer.superposition_candidates(hint_positions)
    collapsed = observer.wavefunction_collapse(candidates, iterations=5)
    
    # Check collapsed candidates first
    for x in collapsed:
        if n % x == 0:
            return (min(x, n // x), max(x, n // x))
    
    # Phase 5b – Spectral gradient navigation
    navigator = SpectralGradientNavigator(n, observer)
    result = navigator.multi_path_search(collapsed[:5])
    if result:
        return result
    
    # Phase 6 – Prime geodesic with harmonic amplification
    geo = PrimeGeodesic(n)
    for pos in geo.walk(root, steps=15):
        if n % pos == 0:
            return (min(pos, n // pos), max(pos, n // pos))
        for h in harmonic_amplify(n, pos)[:10]:  # Limit harmonics
            if n % h == 0:
                return (min(h, n // h), max(h, n // h))

    return (1, n)

# ─────────────────────────────── demo  ──────────────────────────────────────
def main() -> None:
    nums = [221, 299, 437, 10007, 101 * 103, 991 * 997, 13 * 1_000_003]
    print("UOR Ultra-Accelerated Factorizer V2 Demo")
    print("=" * 45)
    print("Enhanced with axiom-compliant optimizations\n")
    
    for n in nums:
        start_time = time.perf_counter()
        p, q = ultra_uor_factor_v2(n)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        status = f"{p} × {q}" if p != 1 else ("prime" if is_prime(n) else "unfactored")
        print(f"{n:>12} : {status:15} [{elapsed:6.2f}ms]")

if __name__ == "__main__":
    import time
    main()
