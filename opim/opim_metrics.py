"""Paper-defined OPIM metrics.

The paper defines three batch-averaged quantities and combines them with
weights 0.4, 0.3, and 0.3.  The agent models use the following explicit
mapping from their internal names to the paper symbols:

``wave_amplitude -> A``; ``wavelength_purity -> L``;
``photon_density -> rho``; ``coherence -> C``;
``fringe_clarity -> S``; ``diffraction_resolution -> R``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


OPIM_WEIGHTS = {
    "perceptual_energy": 0.40,
    "perceptual_stability": 0.30,
    "perceptual_coverage": 0.30,
}

# Theoretical ranges of Eqs. (1)-(3) for inputs in [0, 1]: E_p <= 2 (A=C=S=1),
# S_p <= 1, C_p <= 1 + ln 2.  The weighted score therefore lies in [0, ~1.608]
# and is not bounded by 1.
OPIM_COMPONENT_MAX = 2.0
OPIM_SCORE_MAX = (
    OPIM_WEIGHTS["perceptual_energy"] * 2.0
    + OPIM_WEIGHTS["perceptual_stability"] * 1.0
    + OPIM_WEIGHTS["perceptual_coverage"] * (1.0 + np.log(2.0))
)


def compute_opim_metrics(
    *,
    amplitude: Iterable[float],
    clarity: Iterable[float],
    density: Iterable[float],
    consistency: Iterable[float],
    salience: Iterable[float],
    recognizability: Iterable[float],
    epsilon: float = 1e-8,
    delta: float = 1e-8,
) -> Mapping[str, float]:
    """Compute the exact equations in the paper over a temporal batch.

    Inputs are expected to be normalized to [0, 1] by the upstream scorer.
    The function intentionally does not clip values silently: an out-of-range
    value indicates a malformed scorer response and should be visible to the
    caller.
    """

    def _as_vector(name: str, values: Iterable[float]) -> np.ndarray:
        """Convert one temporal input to a finite, one-dimensional vector.

        ``np.asarray`` otherwise accepts nested arrays and silently changes the
        meaning of the temporal average.  Rejecting those inputs keeps the
        public helper aligned with the paper's one value per time step
        notation and gives callers a useful error for malformed API output.
        """
        try:
            vector = np.asarray(list(values), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("OPIM input %s must be an iterable of numbers" % name) from exc
        if vector.ndim != 1:
            raise ValueError("OPIM input %s must be one-dimensional" % name)
        if not np.all(np.isfinite(vector)):
            raise ValueError("OPIM input %s contains NaN or infinite values" % name)
        return vector

    arrays = {
        "A": _as_vector("amplitude", amplitude),
        "L": _as_vector("clarity", clarity),
        "rho": _as_vector("density", density),
        "C": _as_vector("consistency", consistency),
        "S": _as_vector("salience", salience),
        "R": _as_vector("recognizability", recognizability),
    }
    if not arrays["A"].size:
        raise ValueError("OPIM requires at least one temporal observation")
    size = arrays["A"].size
    if any(value.size != size for value in arrays.values()):
        raise ValueError("OPIM metric inputs must have the same length")
    try:
        epsilon = float(epsilon)
        delta = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon and delta must be finite positive numbers") from exc
    if not np.isfinite(epsilon) or not np.isfinite(delta) or epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be positive")
    if any(np.any((value < 0) | (value > 1)) for value in arrays.values()):
        raise ValueError("OPIM inputs must be normalized to [0,1]")

    energy = np.mean(arrays["A"] ** 2 * (1 + arrays["C"] * arrays["S"]))
    stability = np.mean(
        (arrays["L"] / (arrays["L"] + epsilon)) * arrays["C"]
        + (arrays["R"] / (arrays["R"] + delta)) * (1 - arrays["C"])
    )
    coverage = np.mean(arrays["rho"] * (1 + np.log1p(arrays["R"])))
    score = (
        OPIM_WEIGHTS["perceptual_energy"] * energy
        + OPIM_WEIGHTS["perceptual_stability"] * stability
        + OPIM_WEIGHTS["perceptual_coverage"] * coverage
    )
    output = {
        "perceptual_energy": float(energy),
        "perceptual_stability": float(stability),
        "perceptual_coverage": float(coverage),
        "score": float(score),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise FloatingPointError("OPIM metric computation produced NaN or infinite values")
    return output
