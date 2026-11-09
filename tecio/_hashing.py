"""Integer hash functions that act on whole numpy arrays.

Both functions are bijections, so distinct inputs always give distinct outputs, and both
have good avalanche behavior (flipping one input bit flips about half of the output
bits). They are not cryptographic.

References:
    * :func:`splitmix64` is the output function of the SplitMix64 generator, written in
      2015 by Sebastiano Vigna and dedicated to the public domain
      (https://prng.di.unimi.it/splitmix64.c). It is a fixed increment version of the
      SplittableRandom generator of G. L. Steele Jr., D. Lea and C. H. Flood, "Fast
      splittable pseudorandom number generators", OOPSLA 2014,
      https://doi.org/10.1145/2714064.2660195.
    * :func:`lowbias32` is a two round xorshift-multiply hash from the hash prospector
      of Chris Wellons, "Prospecting for Hash Functions", 2018,
      https://nullprogram.com/blog/2018/07/31/ (public domain, exact bias 0.1735).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["lowbias32", "splitmix64"]

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MUL64 = (np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB))
_MUL32 = (np.uint32(0x7FEB352D), np.uint32(0x846CA68B))


def splitmix64(x: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Return the SplitMix64 output for each generator state in *x*.

    This is one step of the generator: add the golden ratio increment to the state, then
    mix. Output ``k`` of a generator seeded with ``s`` is ``splitmix64(s + k * gamma)``
    with ``gamma = 0x9E3779B97F4A7C15``, so ``splitmix64(0)`` is its first output for
    seed 0. Arithmetic wraps modulo 2**64 and *x* is not modified.

    Args:
        x: Array of ``uint64`` states.

    Returns:
        New ``uint64`` array of the same shape.
    """
    z = x + _GAMMA
    z ^= z >> np.uint64(30)
    z *= _MUL64[0]
    z ^= z >> np.uint64(27)
    z *= _MUL64[1]
    z ^= z >> np.uint64(31)
    return z


def lowbias32(x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """Return the ``lowbias32`` hash of each value in *x*.

    Zero maps to zero, so offset the input when that matters. Arithmetic wraps modulo
    2**32 and *x* is not modified.

    Args:
        x: Array of ``uint32`` values.

    Returns:
        New ``uint32`` array of the same shape.
    """
    z = x ^ (x >> np.uint32(16))
    z *= _MUL32[0]
    z ^= z >> np.uint32(15)
    z *= _MUL32[1]
    z ^= z >> np.uint32(16)
    return z
