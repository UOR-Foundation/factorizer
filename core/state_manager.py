"""
State Manager - Manages computation state with minimal memory footprint
"""

import json
import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class StateManager:
    """
    Manages computation state with minimal memory footprint.
    Provides checkpointing, statistics tracking, and resumption capabilities.
    """

    def __init__(self, checkpoint_interval: int = 10000):
        self.checkpoint_interval = checkpoint_interval
        self.sliding_window = deque(maxlen=1000)
        self.checkpoints: List[Dict[str, Any]] = []
        self.iteration_count = 0
        self.best_resonance = 0.0
        self.best_position = 0
        self.explored_regions: List[Tuple[int, int, float]] = []
        self.start_time = time.time()

    def update(self, x: int, resonance: float):
        """Update state with new evaluation"""
        self.iteration_count += 1
        self.sliding_window.append((x, resonance))

        if resonance > self.best_resonance:
            self.best_resonance = resonance
            self.best_position = x

        # Checkpoint if needed
        if self.iteration_count % self.checkpoint_interval == 0:
            self.checkpoint()

    def checkpoint(self):
        """Save current state for resumption"""
        # Summarize explored regions
        if self.sliding_window:
            positions = [x for x, _ in self.sliding_window]
            resonances = [r for _, r in self.sliding_window]
            region_summary = {
                "min_pos": min(positions),
                "max_pos": max(positions),
                "mean_resonance": sum(resonances) / len(resonances),
                "max_resonance": max(resonances),
                "std_resonance": self._compute_std(resonances),
            }
            self.explored_regions.append(
                (
                    region_summary["min_pos"],
                    region_summary["max_pos"],
                    region_summary["max_resonance"],
                )
            )

        state = {
            "iteration": self.iteration_count,
            "best_resonance": self.best_resonance,
            "best_position": self.best_position,
            "explored_regions": self.explored_regions[-10:],  # Keep last 10
            "elapsed_time": time.time() - self.start_time,
            "timestamp": time.time(),
        }

        self.checkpoints.append(state)

        # Keep only recent checkpoints to save memory
        if len(self.checkpoints) > 10:
            self.checkpoints = self.checkpoints[-10:]

    def save_to_file(self, filename: str):
        """Save state to file"""
        with open(filename, "w") as f:
            json.dump(
                {
                    "checkpoints": self.checkpoints,
                    "final_state": {
                        "iteration": self.iteration_count,
                        "best_resonance": self.best_resonance,
                        "best_position": self.best_position,
                        "total_time": time.time() - self.start_time,
                    },
                },
                f,
                indent=2,
            )

    def resume_from_file(self, filename: str):
        """Resume from saved state"""
        with open(filename, "r") as f:
            data = json.load(f)

        self.checkpoints = data["checkpoints"]
        final_state = data["final_state"]
        self.iteration_count = final_state["iteration"]
        self.best_resonance = final_state["best_resonance"]
        self.best_position = final_state["best_position"]

        # Rebuild explored regions
        self.explored_regions = []
        for checkpoint in self.checkpoints:
            self.explored_regions.extend(checkpoint.get("explored_regions", []))

        # Reset start time
        self.start_time = time.time()

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        if self.sliding_window:
            recent_resonances = [r for _, r in self.sliding_window]
            mean_recent = sum(recent_resonances) / len(recent_resonances)
            std_recent = self._compute_std(recent_resonances)

            # Compute exploration efficiency
            unique_positions = len(set(x for x, _ in self.sliding_window))
            exploration_efficiency = unique_positions / len(self.sliding_window)
        else:
            mean_recent = 0
            std_recent = 0
            exploration_efficiency = 0

        return {
            "iterations": self.iteration_count,
            "best_resonance": self.best_resonance,
            "best_position": self.best_position,
            "mean_recent_resonance": mean_recent,
            "std_recent_resonance": std_recent,
            "regions_explored": len(self.explored_regions),
            "exploration_efficiency": exploration_efficiency,
            "elapsed_time": time.time() - self.start_time,
        }

    def get_convergence_metrics(self) -> Dict[str, float]:
        """Compute convergence metrics"""
        if len(self.checkpoints) < 2:
            return {
                "resonance_improvement_rate": 0.0,
                "exploration_decay_rate": 0.0,
                "convergence_score": 0.0,
            }

        # Resonance improvement over time
        recent_checkpoints = self.checkpoints[-5:]
        if len(recent_checkpoints) >= 2:
            resonance_improvements = [
                recent_checkpoints[i]["best_resonance"]
                - recent_checkpoints[i - 1]["best_resonance"]
                for i in range(1, len(recent_checkpoints))
            ]
            avg_improvement = sum(resonance_improvements) / len(resonance_improvements)

            # Time deltas
            time_deltas = [
                recent_checkpoints[i]["timestamp"]
                - recent_checkpoints[i - 1]["timestamp"]
                for i in range(1, len(recent_checkpoints))
            ]
            avg_time_delta = sum(time_deltas) / len(time_deltas)

            improvement_rate = (
                avg_improvement / avg_time_delta if avg_time_delta > 0 else 0
            )
        else:
            improvement_rate = 0

        # Exploration decay (are we still finding new regions?)
        if len(self.explored_regions) >= 10:
            recent_regions = self.explored_regions[-10:]
            region_overlaps = 0
            for i in range(1, len(recent_regions)):
                r1_start, r1_end, _ = recent_regions[i - 1]
                r2_start, r2_end, _ = recent_regions[i]

                # Check overlap
                if r1_start <= r2_end and r2_start <= r1_end:
                    region_overlaps += 1

            exploration_decay = region_overlaps / (len(recent_regions) - 1)
        else:
            exploration_decay = 0

        # Convergence score (higher = more converged)
        convergence_score = (1 - exploration_decay) * (1 / (1 + abs(improvement_rate)))

        return {
            "resonance_improvement_rate": improvement_rate,
            "exploration_decay_rate": exploration_decay,
            "convergence_score": convergence_score,
        }

    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def reset(self):
        """Reset state for new factorization"""
        self.sliding_window.clear()
        self.checkpoints.clear()
        self.iteration_count = 0
        self.best_resonance = 0.0
        self.best_position = 0
        self.explored_regions.clear()
        self.start_time = time.time()
