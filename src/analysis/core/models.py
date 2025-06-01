from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from .fib_utils import FibonacciRatios

class WaveDegree(Enum):
    SUPERCYCLE = "Supercycle"
    CYCLE = "Cycle"
    PRIMARY = "Primary"
    INTERMEDIATE = "Intermediate"
    MINOR = "Minor"
    MINUTE = "Minute"
    MINUETTE = "Minuette"
    SUBMINUETTE = "Subminuette"

class WaveType(Enum):
    IMPULSE = "impulse"
    DIAGONAL_LEADING = "diagonal_leading"
    DIAGONAL_ENDING = "diagonal_ending"
    ZIGZAG = "zigzag"
    FLAT_REGULAR = "flat_regular"
    FLAT_IRREGULAR = "flat_irregular"
    FLAT_RUNNING = "flat_running"
    TRIANGLE_CONTRACTING = "triangle_contracting"
    TRIANGLE_EXPANDING = "triangle_expanding"
    DOUBLE_CORRECTION = "double_correction"
    TRIPLE_CORRECTION = "triple_correction"

class CorrectiveType(Enum):
    SHARP = "sharp"
    FLAT = "flat"
    SIMPLE = "simple"
    COMPLEX = "complex"

@dataclass
class WaveProperties:
    """Properties of an Elliott Wave"""
    start_idx: int
    end_idx: int
    price_start: float
    price_end: float
    duration: int
    volume_avg: float
    internal_structure: List[int]
    fibonacci_relationships: Dict[str, float]

@dataclass
class ElliottWave:
    wave_points: np.ndarray
    wave_type: WaveType
    degree: WaveDegree
    confidence: float
    properties: WaveProperties
    sub_waves: list = None
    def __post_init__(self):
        if self.sub_waves is None:
            self.sub_waves = [] 