"""
Universal Basis - The mathematical foundation of The Pattern

The Universal Basis provides the coordinate system and operations
for expressing any number in terms of universal constants.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class UniversalBasis:
    """
    The Universal Basis defines the mathematical space where The Pattern operates.
    
    Key principles:
    - Every number has a unique representation in universal coordinates
    - Universal constants (φ, π, e, 1) form the basis vectors
    - Operations in this space reveal hidden structures
    """
    
    def __init__(self):
        # Load universal constants from previous discoveries
        self.constants = self._load_universal_constants()
        
        # Basis vectors
        self.PHI = (1 + np.sqrt(5)) / 2
        self.PI = np.pi
        self.E = np.e
        self.UNITY = 1.0
        
        # Derived constants
        self.ALPHA = self.constants.get('alpha', 1.618033988749895)  # φ
        self.BETA = self.constants.get('beta', 0.3819660112501051)   # 2 - φ
        self.GAMMA = self.constants.get('gamma', 2.718281828459045)  # e
        self.DELTA = self.constants.get('delta', 0.5772156649015329) # Euler-Mascheroni
        
        # Pattern-specific constants
        self.RESONANCE_BASE = self.PHI / self.PI
        self.HARMONIC_SCALE = self.E / self.PHI
        self.UNITY_FIELD = 2 * self.PI
        
    def _load_universal_constants(self) -> Dict[str, float]:
        """Load universal constants from previous discoveries"""
        try:
            with open('/workspaces/factorizer/poly_solver/universal_constants.json', 'r') as f:
                return json.load(f)
        except:
            # Default constants if file not found
            return {
                'alpha': 1.618033988749895,
                'beta': 0.3819660112501051,
                'gamma': 2.718281828459045,
                'delta': 0.5772156649015329,
                'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            }
    
    def project(self, n: int) -> np.ndarray:
        """Project a number onto the universal basis"""
        coordinates = np.zeros(4)
        
        # φ-coordinate: Golden ratio projection
        coordinates[0] = np.log(n) / np.log(self.PHI)
        
        # π-coordinate: Circular projection
        coordinates[1] = (n * self.PHI) % self.PI
        
        # e-coordinate: Exponential projection
        coordinates[2] = np.log(n + 1) / self.E
        
        # Unity coordinate: Normalized projection
        coordinates[3] = n / (n + self.PHI + self.PI + self.E)
        
        return coordinates
    
    def reconstruct(self, coordinates: np.ndarray) -> int:
        """Reconstruct a number from universal coordinates"""
        # Inverse projection
        phi_component = self.PHI ** coordinates[0]
        pi_component = coordinates[1] * self.PI / self.PHI
        e_component = self.E * coordinates[2]
        unity_component = coordinates[3] / (1 - coordinates[3])
        
        # Weighted reconstruction
        n_estimate = (phi_component * 0.4 + 
                     pi_component * 0.2 + 
                     e_component * 0.3 + 
                     unity_component * 0.1)
        
        return int(round(n_estimate))
    
    def transform(self, coordinates: np.ndarray, operation: str) -> np.ndarray:
        """Apply universal transformations"""
        if operation == 'rotate':
            # Rotation in universal space
            theta = self.PHI / self.PI
            rotation = np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            return rotation @ coordinates
        
        elif operation == 'scale':
            # Scaling by universal constants
            scaling = np.diag([self.PHI, self.PI, self.E, self.UNITY])
            return scaling @ coordinates
        
        elif operation == 'reflect':
            # Reflection through universal hyperplane
            normal = np.array([1, self.BETA, self.GAMMA, self.DELTA])
            normal = normal / np.linalg.norm(normal)
            reflection = np.eye(4) - 2 * np.outer(normal, normal)
            return reflection @ coordinates
        
        else:
            return coordinates
    
    def measure_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Measure distance in universal space"""
        # Weighted distance incorporating universal constants
        weights = np.array([self.PHI, self.PI, self.E, self.UNITY])
        diff = coord1 - coord2
        return np.sqrt(np.sum(weights * diff**2))
    
    def find_resonance_points(self, n: int) -> List[int]:
        """Find resonance points in universal space near n"""
        n_coords = self.project(n)
        resonance_points = []
        
        # Search radius based on n's magnitude
        radius = int(np.log(n) * self.PHI)
        
        for offset in range(-radius, radius + 1):
            candidate = n + offset
            if candidate > 1:
                candidate_coords = self.project(candidate)
                
                # Check for resonance conditions
                if self._is_resonant(n_coords, candidate_coords):
                    resonance_points.append(candidate)
        
        return resonance_points
    
    def _is_resonant(self, coord1: np.ndarray, coord2: np.ndarray) -> bool:
        """Check if two points are in resonance"""
        # Resonance occurs when coordinates have special relationships
        
        # Condition 1: Golden ratio relationship
        phi_ratio = coord1[0] / (coord2[0] + 1e-10)
        if abs(phi_ratio - self.PHI) < 0.1 or abs(phi_ratio - self.BETA) < 0.1:
            return True
        
        # Condition 2: Harmonic relationship
        freq_ratio = coord1[1] / (coord2[1] + 1e-10)
        if abs(freq_ratio - round(freq_ratio)) < 0.1:
            return True
        
        # Condition 3: Exponential relationship
        exp_diff = abs(coord1[2] - coord2[2])
        if exp_diff < 0.1 or abs(exp_diff - 1) < 0.1:
            return True
        
        return False
    
    def decompose_in_basis(self, n: int) -> Dict[str, float]:
        """Decompose n in terms of universal basis"""
        coords = self.project(n)
        
        decomposition = {
            'phi_component': coords[0] * self.PHI,
            'pi_component': coords[1] * self.PI,
            'e_component': coords[2] * self.E,
            'unity_component': coords[3] * self.UNITY,
            'resonance_field': self._compute_resonance_field_strength(coords),
            'harmonic_signature': self._compute_harmonic_signature(coords),
            'universal_phase': self._compute_universal_phase(coords)
        }
        
        return decomposition
    
    def _compute_resonance_field_strength(self, coords: np.ndarray) -> float:
        """Compute the resonance field strength at given coordinates"""
        # Field strength is combination of all components
        phi_field = np.sin(coords[0] * self.PHI)
        pi_field = np.cos(coords[1] * self.PI)
        e_field = np.exp(-coords[2] / self.E)
        unity_field = coords[3]
        
        return (phi_field + pi_field + e_field + unity_field) / 4
    
    def _compute_harmonic_signature(self, coords: np.ndarray) -> List[float]:
        """Compute harmonic signature in universal space"""
        harmonics = []
        
        for k in range(1, 8):  # First 7 harmonics
            harmonic = (coords[0] * k * self.PHI + 
                       coords[1] * k * self.PI + 
                       coords[2] * k * self.E + 
                       coords[3] * k * self.UNITY)
            harmonics.append(harmonic % self.UNITY_FIELD)
        
        return harmonics
    
    def _compute_universal_phase(self, coords: np.ndarray) -> float:
        """Compute the universal phase of the coordinates"""
        # Phase is the angle in universal space
        magnitude = np.linalg.norm(coords)
        if magnitude > 0:
            normalized = coords / magnitude
            # Phase relative to φ-axis
            phase = np.arccos(normalized[0])
        else:
            phase = 0
        
        return phase
    
    def get_factor_relationship_matrix(self, n: int) -> np.ndarray:
        """
        Compute the factor relationship matrix in universal space.
        This matrix encodes how factors relate through universal constants.
        """
        size = 5  # 5x5 matrix for comprehensive relationships
        matrix = np.zeros((size, size))
        
        # Compute universal properties
        n_coords = self.project(n)
        decomp = self.decompose_in_basis(n)
        
        # Row 0: Direct universal components
        matrix[0, :4] = n_coords
        matrix[0, 4] = decomp['resonance_field']
        
        # Row 1: Harmonic relationships
        harmonics = decomp['harmonic_signature']
        matrix[1, :len(harmonics[:5])] = harmonics[:5]
        
        # Row 2: Transformed coordinates
        rotated = self.transform(n_coords, 'rotate')
        matrix[2, :4] = rotated
        matrix[2, 4] = decomp['universal_phase']
        
        # Row 3: Resonance relationships
        resonance_points = self.find_resonance_points(n)[:5]
        for i, point in enumerate(resonance_points):
            if i < 5:
                matrix[3, i] = point / n
        
        # Row 4: Cross-component relationships
        matrix[4, 0] = n_coords[0] * n_coords[1]  # φ-π coupling
        matrix[4, 1] = n_coords[1] * n_coords[2]  # π-e coupling
        matrix[4, 2] = n_coords[2] * n_coords[3]  # e-unity coupling
        matrix[4, 3] = n_coords[3] * n_coords[0]  # unity-φ coupling
        matrix[4, 4] = np.prod(n_coords)  # Total coupling
        
        return matrix