"""
Hierarchical Search Controller - Coarse-to-fine resonance field exploration
"""

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .multi_scale_resonance import MultiScaleResonance


class HierarchicalSearch:
    """
    Implements coarse-to-fine resonance field exploration.
    Uses multi-resolution sampling to efficiently find high-resonance regions.
    """

    def __init__(self, n: int, analyzer: "MultiScaleResonance"):
        self.n = n
        self.sqrt_n = int(math.sqrt(n))
        self.analyzer = analyzer
        self.levels = self._compute_hierarchy_levels()
        self.log_n = math.log(n)

    def _compute_hierarchy_levels(self) -> List[int]:
        """Compute sampling resolutions for each level"""
        levels = []
        points = self.sqrt_n

        # Adaptive level computation based on n's size
        if self.n.bit_length() > 100:
            # For very large numbers, use fewer levels
            reduction_factor = 0.7
        else:
            reduction_factor = 0.5

        while points > 100:
            levels.append(int(points))
            points = int(points**reduction_factor)

        levels.append(100)  # Minimum level
        return levels[::-1]  # Reverse to go coarse to fine

    def search(self, max_time: float = 5.0) -> List[Tuple[int, float]]:
        """
        Perform hierarchical search.
        Returns list of (position, resonance) tuples sorted by resonance.
        """
        import time

        start_time = time.time()

        # Level 1: Coarse sampling
        coarse_peaks = self._coarse_sample(max_time * 0.3)

        if time.time() - start_time > max_time * 0.5:
            return self._finalize_candidates(coarse_peaks)

        # Level 2: Refine around peaks
        refined_regions = []
        for peak, resonance in coarse_peaks[:10]:  # Top 10 peaks
            if time.time() - start_time > max_time * 0.7:
                break
            refined = self._refine_peak(peak, resonance)
            refined_regions.extend(refined)

        # Level 3: Binary search on gradients
        candidates = []
        for region_center, region_resonance in refined_regions[:20]:
            if time.time() - start_time > max_time * 0.9:
                break
            binary_results = self._binary_search_peak(region_center)
            candidates.extend(binary_results)

        # Combine all candidates
        all_candidates = coarse_peaks + refined_regions + candidates
        return self._finalize_candidates(all_candidates)

    def _coarse_sample(self, timeout: float) -> List[Tuple[int, float]]:
        """Coarse sampling of resonance field"""
        import time

        start_time = time.time()

        if not self.levels:
            return []

        # Adaptive sampling based on n's size
        if self.n.bit_length() > 80:
            sample_points = min(self.levels[0], 500)
        else:
            sample_points = min(self.levels[0], 1000)

        step = max(1, self.sqrt_n // sample_points)

        peaks = []

        # Special handling for balanced semiprimes - CRITICAL
        # Always check near sqrt(n) first
        sqrt_region_size = min(100, int(self.sqrt_n**0.05))
        for offset in range(-sqrt_region_size, sqrt_region_size + 1):
            x = self.sqrt_n + offset
            if 2 <= x <= self.sqrt_n:
                resonance = self.analyzer.compute_coarse_resonance(x, self.n)
                if resonance > 0.1:
                    peaks.append((x, resonance))

        # Regular coarse sampling
        for i in range(2, self.sqrt_n, step):
            if time.time() - start_time > timeout:
                break

            # Skip if already checked in sqrt region
            if abs(i - self.sqrt_n) <= sqrt_region_size:
                continue

            resonance = self.analyzer.compute_coarse_resonance(i, self.n)
            if resonance > 0.1:  # Basic threshold
                peaks.append((i, resonance))

        # Sort by resonance
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:50]  # Keep top 50

    def _refine_peak(self, peak: int, peak_resonance: float) -> List[Tuple[int, float]]:
        """Refine search around a peak"""
        # Adaptive window size based on peak position and n's size
        if abs(peak - self.sqrt_n) < self.sqrt_n * 0.01:
            # Near sqrt(n) - use larger window for balanced factors
            window = int(self.sqrt_n**0.05 * (1 + peak_resonance))
        else:
            window = int(self.sqrt_n**0.1 * (1 + peak_resonance))

        window = min(window, 10000)  # Cap window size

        refined = []

        # Finer sampling around peak
        step = max(1, window // 20)
        for offset in range(-window, window + 1, step):
            x = peak + offset
            if 2 <= x <= self.sqrt_n:
                # Use full resonance computation for refinement
                resonance = self.analyzer.compute_resonance(x, self.n)
                if resonance > peak_resonance * 0.5:
                    refined.append((x, resonance))

        return refined

    def _binary_search_peak(self, center: int) -> List[Tuple[int, float]]:
        """Binary search for local maxima using golden section search"""
        candidates = []

        # Search window
        window_size = int(self.sqrt_n**0.05)
        left = max(2, center - window_size)
        right = min(self.sqrt_n, center + window_size)

        # Golden section search
        phi = (1 + math.sqrt(5)) / 2
        resphi = 2 - phi

        tol = max(1, int((right - left) * 0.001))

        x1 = left + int(resphi * (right - left))
        x2 = right - int(resphi * (right - left))
        f1 = self.analyzer.compute_resonance(x1, self.n)
        f2 = self.analyzer.compute_resonance(x2, self.n)

        iterations = 0
        max_iterations = 20  # Limit iterations

        while abs(right - left) > tol and iterations < max_iterations:
            iterations += 1

            if f1 > f2:
                right = x2
                x2 = x1
                f2 = f1
                x1 = left + int(resphi * (right - left))
                f1 = self.analyzer.compute_resonance(x1, self.n)
            else:
                left = x1
                x1 = x2
                f1 = f2
                x2 = right - int(resphi * (right - left))
                f2 = self.analyzer.compute_resonance(x2, self.n)

        # Check the peak
        peak_x = (left + right) // 2
        peak_resonance = self.analyzer.compute_resonance(peak_x, self.n)

        if peak_resonance > 0.1:
            candidates.append((peak_x, peak_resonance))

            # Also check immediate neighbors
            for offset in [-1, 1]:
                x = peak_x + offset
                if 2 <= x <= self.sqrt_n:
                    res = self.analyzer.compute_resonance(x, self.n)
                    if res > 0.1:
                        candidates.append((x, res))

        return candidates

    def _finalize_candidates(
        self, candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Remove duplicates and sort candidates"""
        seen = {}
        for x, res in candidates:
            if x not in seen or res > seen[x]:
                seen[x] = res

        final = [(x, res) for x, res in seen.items()]
        final.sort(key=lambda x: x[1], reverse=True)
        return final[:100]  # Return top 100
