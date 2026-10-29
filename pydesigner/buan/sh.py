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


def gradient_frame(bvec, mrtrix_grad):
    """Find row-vector mapping from pyDKE FSL gradients to MRtrix scanner gradients."""
    bvec = np.asarray(bvec, float)
    grad = np.asarray(mrtrix_grad, float)
    if bvec.shape != (len(grad), 3) or grad.shape[1] != 4:
        raise ValueError('Gradient tables have incompatible shapes')
    use = (grad[:, 3] > 50) & (np.linalg.norm(bvec, axis=1) > .5)
    if np.count_nonzero(use) < 6 or np.linalg.matrix_rank(bvec[use]) < 3:
        raise ValueError('Insufficient gradients to infer a 3D coordinate frame')
    mapping = np.linalg.lstsq(bvec[use], grad[use, :3], rcond=None)[0]
    error = np.max(np.linalg.norm(bvec[use] @ mapping - grad[use, :3], axis=1))
    if error > 1e-3 or not np.allclose(mapping.T @ mapping, np.eye(3), atol=1e-3):
        raise ValueError(f'Gradient tables do not share one orthogonal coordinate frame (error {error})')
    return mapping, float(error)


def rotate_mrtrix_sh(coefficients, source_to_world):
    """Express scanner-basis SH in a different orthogonal physical frame.

    source_to_world maps row direction vectors in the source frame to world.
    Both even-order rotations and reflections are handled through evaluation.
    """
    from dipy.reconst.shm import real_sh_tournier
    data = np.asarray(coefficients)
    mapping = np.asarray(source_to_world, float)
    if mapping.shape != (3, 3) or not np.allclose(mapping.T @ mapping, np.eye(3), atol=1e-3):
        raise ValueError('Expected orthogonal 3x3 direction transform')
    count = data.shape[-1]
    possible = {((l + 1) * (l + 2) // 2): l for l in range(0, 14, 2)}
    if count not in possible:
        raise ValueError('Invalid SH coefficient count')
    lmax = possible[count]
    n = max(4 * count, 256)
    z = 1 - 2 * (np.arange(n) + 0.5) / n
    phi = np.arange(n) * (np.pi * (3 - np.sqrt(5)))
    dirs = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
    def basis(d):
        return real_sh_tournier(lmax, np.arccos(np.clip(d[:, 2], -1, 1)), np.arctan2(d[:, 1], d[:, 0]), legacy=False)[0]
    world_basis = basis(dirs)
    source_basis = basis(dirs @ mapping.T)
    transform = np.linalg.lstsq(world_basis, source_basis, rcond=None)[0]
    return data @ transform.T
