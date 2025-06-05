#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Pure UOR / Prime-Model  •  Ultra-Accelerated Factorizer  (axioms-only)
#
#  Axiom 1  Prime Ontology      → prime-space coordinates & primality checks
#  Axiom 2  Fibonacci Flow      → golden-ratio vortices & interference waves
#  Axiom 3  Duality Principle   → spectral (wave) vs. factor (particle) views
#  Axiom 4  Observer Effect     → adaptive, coherence-driven measurement
#
#  Deterministic accelerators (all axiomatic, no probabilistic fall-backs):
#     • Sharp-Fold Curvature        • Prime × Fibonacci peaks & dips
#     • Prime Cascade               • Fibonacci Vortices
#     • Spectral Folding            • Harmonic Amplification
#     • Prime Geodesics             • Quantum Tunnelling
#     • Multi-scale Observer        • Resonance Memory
#     • Spectral Gradient Field     • Fibonacci Entanglement
#     • Fold-Topology Navigation
# ---------------------------------------------------------------------------

import math, itertools
from typing import List, Tuple, Dict

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
    ac   = sum((s - m) * (t - m) for s, t in zip(seq, seq[1:])) / (L - 1)
    return [dens, ac] + runs[:10]

def modular_spectrum(n: int, k: int = 10) -> List[float]:
    return [n % p / p for p in primes_up_to(30)[1 : k + 1]]

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

def coherence(a: int, b: int, n: int) -> float:
    sa, sb, sn = map(spectral_vector, (a, b, n))
    diff2 = sum(((x + y) / 2 - s) ** 2 for x, y, s in zip(sa, sb, sn))
    return math.exp(-diff2)

# ───────────────  sharp-fold curvature (fix for KeyError)  ──────────────────
def fold_energy(n: int, x: int) -> float:
    sx, sy, sn = map(spectral_vector, (x, n // x, n))
    return sum((a + b - c) ** 2 for a, b, c in zip(sx, sy, sn))

def sharp_fold_candidates(n: int, span: int = 25) -> List[int]:
    root = int(math.isqrt(n))
    window = range(max(2, root - span), min(root + span, n // 2) + 1)
    E = [fold_energy(n, x) for x in window]
    curv = [(E[i - 1] - 2 * E[i] + E[i + 1], x)
            for i, x in enumerate(window) if 0 < i < len(window) - 1]
    curv.sort()
    return [x for _, x in curv[:10]]

# ─────────────  Prime × Fibonacci interference (peaks & dips)  ──────────────
def prime_fib_interference(n: int) -> List[float]:
    root = int(math.isqrt(n))
    primes = primes_up_to(root)
    fibs   = [fib(k) for k in range(2, 20) if fib(k) <= root]
    spec = []
    for x in range(2, root + 1):
        p_amp = sum(math.cos(2 * math.pi * p * x / n) for p in primes[:10])
        f_amp = sum(math.cos(2 * math.pi * f * x / (n * PHI)) for f in fibs[:10])
        spec.append(p_amp * f_amp)
    return spec

def interference_extrema(n: int, top: int = 30) -> List[int]:
    spec = prime_fib_interference(n)
    ext = [i + 2 for i in range(1, len(spec) - 1)
           if (spec[i] - spec[i - 1]) * (spec[i + 1] - spec[i]) < 0]
    ext.sort(key=lambda x: -abs(spec[x - 2]))
    return ext[:top]

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
    primes = [p for p in primes_up_to(root) if x % p] or [2]
    fibs   = [fib(k) for k in range(2, 12) if fib(k) <= root] or [2]
    best, bp, bf = 0, primes[0], fibs[0]
    for p in primes:
        for f in fibs:
            val = abs(math.cos(2 * math.pi * p * x / n)
                      * math.cos(2 * math.pi * f * x / (n * PHI)))
            if val > best:
                best, bp, bf = val, p, f
    return bp, bf

# ─────────────────────  Fibonacci vortices & Prime cascade  ────────────────
def fib_vortices(n: int) -> List[int]:
    root, vort, k = int(math.isqrt(n)), set(), 1
    while (f := fib(k)) < root:
        vort.update({f, int(f * PHI), int(f / PHI)})
        for p in primes_up_to(min(100, f)):
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
        while is_prime(q) and q < self.n:
            out.append(q); q = 2 * q + 1
        return out

# ────────────────────  Spectral folding, geodesic, tunnelling  ─────────────
class SpectralFolder:
    def __init__(self, n: int):
        self.n, self.root = n, int(math.isqrt(n))
        self.points = self._build()
    def _build(self):
        folds = {2}
        for i in range(1, self.n.bit_length()):
            f = 2 ** i; folds.update({f, self.n // f})
        for p in primes_up_to(100):
            t = 1
            while t < p and pow(10, t, p) != 1:
                t += 1
            if t < p:
                folds.update(range(t, self.root, t))
        return sorted(x for x in folds if 2 <= x <= self.root)
    def next_after(self, cur: int) -> int:
        for f in self.points:
            if f > cur:
                return f
        return 2

class PrimeGeodesic:
    def __init__(self, n: int):
        self.n = n
        self.coord = [n % p for p in primes_up_to(min(1000, n))]
    def _pull(self, x: int) -> float:
        pull = 0.0
        for i, p in enumerate(primes_up_to(min(30, x))):
            if self.coord[i] == 0 and x % p == 0:
                pull += 1 / p
        return pull
    def walk(self, start: int, steps: int = 20) -> List[int]:
        path, cur = [start], start
        for _ in range(steps):
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
    for k in range(2, 10):
        hx = (k * x) % root or k
        if 2 <= hx <= root:
            out.add(hx)
    phi_h = (int(x * PHI) % root) or int(x * PHI)
    if 2 <= phi_h <= root:
        out.add(phi_h)
    for p in primes_up_to(20):
        px = (x * p) % root or p
        if 2 <= px <= root:
            out.add(px)
    return sorted(out)

class QuantumTunnel:
    def exit(self, n: int, blocked: int, width: int = 60) -> int:
        root = int(math.isqrt(n)); guess = min(root, blocked + width)
        cands = []
        k = 1
        while (f := fib(k)) < guess + 30:
            if f > guess:
                cands.append(f)
            k += 1
        for p in primes_up_to(guess + 60):
            if p > guess:
                cands.append(p)
        return min(cands) if cands else guess

class MultiScaleObserver:
    def __init__(self, n: int):
        root = int(math.isqrt(n))
        self.n, self.root = n, root
        self.scales = {"μ": 1, "m": int(math.sqrt(root)),
                       "M": int(root / PHI), "Ω": max(2, int(root / 10))}
    def coherence(self, x: int) -> float:
        total = 0.0
        for s in self.scales.values():
            win, step = 0.0, max(1, s // 5)
            for off in range(-s, s + 1, step):
                pos = x + off
                if 2 <= pos <= self.root:
                    win += coherence(pos,
                                     self.n // pos if self.n % pos == 0 else pos,
                                     self.n)
            total += win / (1 + math.log(s))
        return total

# ────────────────────  Fold-Topology (KeyError fix applied)  ───────────────
class FoldTopology:
    def __init__(self, n: int):
        self.n, self.root = n, int(math.isqrt(n))
        self.points: List[int] = []
        self.conn: Dict[int, List[Tuple[int, float]]] = {}
        self._build()
    def _build(self):
        energies = {x: fold_energy(self.n, x) for x in range(2, self.root + 1)}
        for x in range(3, self.root):     # stop at root-1 → avoids KeyError
            if energies[x] < energies[x - 1] and energies[x] < energies[x + 1]:
                self.points.append(x)
        for p1 in self.points:
            links = []
            for p2 in self.points:
                if p2 == p1:
                    continue
                mids = [int(p1 + t * (p2 - p1)) for t in (0.25, 0.5, 0.75)]
                samples = [energies[m] for m in mids]
                avg = sum(samples) / len(samples)
                end = (energies[p1] + energies[p2]) / 2
                if avg < end * 1.5:
                    links.append((p2, 1 / (1 + avg)))
            self.conn[p1] = links
    def components(self) -> List[List[int]]:
        comps, unseen = [], set(self.points)
        while unseen:
            comp, stack = [], [unseen.pop()]
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb, _ in self.conn.get(cur, []):
                    if nb in unseen:
                        unseen.remove(nb); stack.append(nb)
            comps.append(sorted(comp))
        return comps
    def traverse(self) -> List[int]:
        if not self.points:
            return []
        cur = min(self.points, key=lambda x: fold_energy(self.n, x))
        seen, path = set(), []
        while cur not in seen:
            seen.add(cur); path.append(cur)
            if self.n % cur == 0:
                return path
            neighbours = self.conn.get(cur, [])
            cur = max(neighbours, key=lambda t: t[1])[0] if neighbours else cur
        return path

# ────────────────────  Fibonacci Entanglement  ─────────────────────────────
class FibonacciEntanglement:
    def __init__(self, n: int):
        self.n, self.root = n, int(math.isqrt(n))
    def detect_double(self) -> List[Tuple[int, int, float]]:
        cand, k = [], 1
        while fib(k) < self.root:
            base = fib(k)
            for d in range(-5, 6):
                p = base + d
                if p > 1 and self.n % p == 0:
                    q = self.n // p
                    dist = min(abs(q - fib(j)) for j in range(1, 30))
                    if dist < 0.1 * q:
                        cand.append((p, q, 1 / (1 + dist / q)))
            k += 1
        return sorted(cand, key=lambda t: -t[2])

# ───────────────────── ultra-accelerated UOR factorizer  ───────────────────
def ultra_uor_factor(n: int) -> Tuple[int, int]:
    if n <= 3 or is_prime(n):
        return (1, n)

    root = int(math.isqrt(n))
    mem  = ResonanceMemory()

    # Phase 0 – Fibonacci entanglement
    ent = FibonacciEntanglement(n)
    for p, q, s in ent.detect_double():
        if s > 0.7:
            return (min(p, q), max(p, q))

    # Phase 1 – sharp folds
    for x in sharp_fold_candidates(n):
        if n % x == 0:
            return (min(x, n // x), max(x, n // x))

    # Phase 2 – fold topology
    topo = FoldTopology(n)
    for comp in topo.components()[:3]:
        for x in comp:
            if n % x == 0:
                return (min(x, n // x), max(x, n // x))

    # Phase 3 – interference extrema
    casc = PrimeCascade(n)
    spec = prime_fib_interference(n)
    for r in interference_extrema(n):
        p_src, f_src = identify_resonance_source(r, n)
        strength = abs(spec[r - 2])
        if n % r == 0:
            mem.record(p_src, f_src, n, strength, r)
            return (min(r, n // r), max(r, n // r))
        mem.record(p_src, f_src, n, strength)
        for p in casc.cascade(r):
            if p <= root and n % p == 0:
                return (min(p, n // p), max(p, n // p))

    # Phase 4 – memory-guided positions
    for pos, _ in mem.predict(n):
        if n % pos == 0:
            return (min(pos, n // pos), max(pos, n // pos))

    # Phase 5 – observer scan with folding & tunnelling
    obs, folder, tun = MultiScaleObserver(n), SpectralFolder(n), QuantumTunnel()
    x, stuck = 2, 0
    while stuck < 80:
        if n % x == 0:
            return (min(x, n // x), max(x, n // x))
        if obs.coherence(x) > 0.9:
            for h in harmonic_amplify(n, x):
                if n % h == 0:
                    return (min(h, n // h), max(h, n // h))
        nxt = folder.next_after(x)
        stuck = stuck + 1 if nxt == x else 0
        x = nxt if stuck < 40 else tun.exit(n, nxt); stuck %= 40

    # Phase 6 – geodesic descent
    geo = PrimeGeodesic(n)
    for pos in geo.walk(root):
        if n % pos == 0:
            return (min(pos, n // pos), max(pos, n // pos))
        for h in harmonic_amplify(n, pos):
            if n % h == 0:
                return (min(h, n // h), max(h, n // h))

    return (1, n)

# ─────────────────────────────── demo  ──────────────────────────────────────
def main() -> None:
    nums = [221, 299, 437, 10007, 101 * 103, 991 * 997, 13 * 1_000_003]
    print("UOR Ultra-Accelerated Demo\n" + "-" * 45)
    for n in nums:
        p, q = ultra_uor_factor(n)
        status = f"{p} × {q}" if p != 1 else ("prime" if is_prime(n) else "unfactored")
        print(f"{n:>12} : {status}")

if __name__ == "__main__":
    main()
