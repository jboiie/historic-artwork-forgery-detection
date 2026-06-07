"""
latent_utils.py
---------------
Utility functions for generating latent vectors for GAN training.
Provides outlier-biased sampling to push the generator toward
anomalous/unusual artwork regions — useful for forgery detection.
"""

import numpy as np
import tensorflow as tf


def generate_random_latent(batch_size: int, latent_dim: int) -> tf.Tensor:
    """Standard normal latent vector."""
    return tf.random.normal([batch_size, latent_dim])


def generate_outlier_latent(batch_size: int, latent_dim: int, outlier_scale: float = 2.5) -> tf.Tensor:
    """
    Generates latent vectors biased toward the tails of a normal distribution.
    This encourages the generator to explore 'unusual' image regions,
    which is useful when training for forgery/anomaly detection.

    Args:
        batch_size:     Number of latent vectors to generate.
        latent_dim:     Dimensionality of the latent space.
        outlier_scale:  Multiplier for the standard deviation (default 2.5).
                        Higher values = more extreme / outlier-like samples.

    Returns:
        A tf.Tensor of shape (batch_size, latent_dim).
    """
    # Half normal samples, half outlier-biased
    half = batch_size // 2
    remainder = batch_size - half

    normal_samples = tf.random.normal([half, latent_dim])

    # Outlier samples: random sign * absolute value drawn from a scaled normal
    outlier_raw = tf.abs(tf.random.normal([remainder, latent_dim])) * outlier_scale
    signs = tf.cast(tf.random.uniform([remainder, latent_dim], minval=0, maxval=2, dtype=tf.int32) * 2 - 1, tf.float32)
    outlier_samples = outlier_raw * signs

    combined = tf.concat([normal_samples, outlier_samples], axis=0)

    # Shuffle so normal and outlier samples are interleaved
    indices = tf.random.shuffle(tf.range(batch_size))
    return tf.gather(combined, indices)


def generate_interpolated_latent(z1: tf.Tensor, z2: tf.Tensor, steps: int = 10) -> tf.Tensor:
    """
    Linearly interpolates between two latent vectors.
    Useful for visualising the GAN's learned latent space.

    Args:
        z1:    Start latent vector, shape (latent_dim,).
        z2:    End latent vector, shape (latent_dim,).
        steps: Number of interpolation steps.

    Returns:
        Tensor of shape (steps, latent_dim).
    """
    alphas = tf.linspace(0.0, 1.0, steps)
    return tf.stack([alpha * z2 + (1 - alpha) * z1 for alpha in alphas])
