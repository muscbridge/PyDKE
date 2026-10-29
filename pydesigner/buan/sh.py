"""Convert the current pyDKE odf.shbasis('tournier') to MRtrix3 SH.

This changes basis functions only. It does not establish the physical frame
of a historical export or correct an independently applied coordinate flip.
"""
import numpy as np


def pydke_to_mrtrix(coefficients):
    """Convert even-degree coefficients from the current pyDKE implementation."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim < 1:
        raise ValueError('Expected an array with a coefficient axis')
    count = coefficients.shape[-1]
    modes = [(l, m) for l in range(0, 14, 2) for m in range(-l, l + 1)]
    ends = [(l + 1) * (l + 2) // 2 for l in range(0, 14, 2)]
    if count not in ends:
        raise ValueError('Expected a complete symmetric even-order SH expansion')
    signs = np.array([-1 if (m < 0 and m % 2 == 0) or (m > 0 and m % 2 != 0) else 1
                      for _, m in modes[:count]])
    return coefficients * signs
