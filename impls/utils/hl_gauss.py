"""HL-Gauss (histogram + Gaussian) utilities for distributional RL.

Shapes
------
* ``z_centers``: ``[num_atoms]`` — fixed support locations in scalar return space.
* ``transform_to_probs(scalar)`` with ``scalar`` of shape ``[B]`` → ``[B, num_atoms]``.
* ``transform_from_probs(probs)`` with ``probs`` ``[..., num_atoms]`` → ``[...,]`` scalar expectation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def hl_gauss_atom_centers(v_min: float, v_max: float, num_atoms: int) -> jnp.ndarray:
    """Uniformly spaced atom centers in ``[v_min, v_max]`` inclusive."""
    if num_atoms < 2:
        raise ValueError('num_atoms must be >= 2.')
    return jnp.linspace(v_min, v_max, num_atoms, dtype=jnp.float32)


def hl_gauss_transform(v_min: float, v_max: float, num_atoms: int, sigma: float) -> tuple[jnp.ndarray, float]:
    """Return atom centers and sigma (HL-Gauss parameters).

    ``hl_gauss_transform`` is kept for API symmetry with the reference design; the
    returned centers are the only persistent transform state used downstream.
    """
    z = hl_gauss_atom_centers(v_min, v_max, num_atoms)
    return z, float(sigma)


def transform_to_probs(
    scalar: jnp.ndarray,
    *,
    z_centers: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Map scalar targets to a smoothed histogram over atoms (HL-Gauss).

    Args:
        scalar: ``[B]`` (or broadcastable) scalar returns / values.
        z_centers: ``[num_atoms]`` support locations.
        sigma: Gaussian smoothing width (must be > 0).

    Returns:
        ``[..., num_atoms]`` non-negative probabilities summing to 1 along the last axis.
    """
    if sigma <= 0:
        raise ValueError('sigma must be > 0 for HL-Gauss.')
    # [B, 1] - [1, A] -> [B, A]
    d = scalar[..., None] - z_centers
    logits = -(d**2) / (2.0 * (sigma**2))
    return jax.nn.softmax(logits, axis=-1)


def transform_from_probs(probs: jnp.ndarray, *, z_centers: jnp.ndarray) -> jnp.ndarray:
    """Expectation ``sum_i p_i z_i`` (distributional scalar)."""
    return jnp.sum(probs * z_centers, axis=-1)


def cross_entropy_loss_on_scalar(
    logits: jnp.ndarray,
    scalar_target: jnp.ndarray,
    *,
    z_centers: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Per-example cross-entropy ``-sum_k p*_k log pi_k`` with HL-Gauss target ``p*``.

    Args:
        logits: ``[..., num_atoms]`` (any leading batch dims).
        scalar_target: ``[B]`` or broadcastable to logits' leading dims without the atom axis.
        z_centers: ``[num_atoms]``.
        sigma: HL-Gauss smoothing.

    Returns:
        Per-example CE with the same leading shape as ``logits[..., 0]``.
    """
    target_probs = jax.lax.stop_gradient(transform_to_probs(scalar_target, z_centers=z_centers, sigma=sigma))
    log_pi = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(target_probs * log_pi, axis=-1)
