"""Laboratory to compare per-direction embeddings and synchronizations.

An experiment has two stages, each described by a plain dictionary:

* stage A groups the particles by projection direction and embeds every group
  on the GPU. Its result is cached on disk, keyed by its configuration.
* stage B aligns the per-direction embeddings with each other and merges
  them into one embedding per particle, which is then scored.
"""
from functools import partial
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import sklearn.decomposition
import starfile

from factorem import analysis, ctf, geometry, synchronization
from factorem.analysis.preprocessor import _align_single, _in_plane_rotation, _matmul

import datasets
import metrics
import sync
import commonlines

CACHE = f'{datasets.BENCH}/cache'

STAGE_A_DEFAULTS = dict(
    resolution=16.0,        # Band limit [A].
    aperture_index=1.0,     # Cone half-angle, in Shannon angles at `resolution`.
    direction_index=1.0,    # Spacing of the directions, in Shannon angles.
    min_particles=100,
    padding=1.0,            # FFT box over image box.
    mask=None,              # Mask diameter over particle diameter, or None.
    support=1.0,            # Kept Fourier radius over the band limit.
    butterworth=False,      # Shape the band with the Butterworth of factorem.
    whiten=True,            # Divide by the radial noise amplitude.
    noise='residual',       # 'residual' or 'background' (needs a mask).
    tilt_order=0,           # Order of the in-cone view polynomial removed.
    method='ppca',
    band_width=2,           # Shells per band of the two halves of a split.
    components=6,
    options={},             # Options of the method.
)

STAGE_B_DEFAULTS = dict(
    solver='burer_monteiro',  # or 'multigrid', 'common_lines'
    partners=None,          # Random partners of each direction (None: all).
    split=False,            # Similarities across independent halves.
    smoothing=16,
    refine=0,               # Power iterations after the multigrid.
    normalize='max',        # How to scale the similarities.
    diagonal=True,          # Keep the similarity of each direction to itself.
    rank=None,              # Burer-Monteiro rank, 2k+1 if None.
    max_iter=1024,
    tol=1e-8,
    seed=0,
    gain_iterations=0,
    components=None,        # Output components, same as stage A if None.
)


# ---------------------------------------------------------------- input ----

def load(name: str) -> SimpleNamespace:
    dataset = datasets.DATASETS[name]
    star = starfile.read(dataset.star, always_dict=True)
    particles = star['particles']
    optics = star['optics'].iloc[0]
    pixel_size = float(optics['rlnImagePixelSize'])
    rotations = geometry.euler_zyz_to_matrix(
        np.deg2rad(particles['rlnAngleRot'].to_numpy()),
        np.deg2rad(particles['rlnAngleTilt'].to_numpy()),
        np.deg2rad(particles['rlnAnglePsi'].to_numpy()),
    )
    origin = particles[['rlnOriginXAngst', 'rlnOriginYAngst']].to_numpy()
    defocus = 0.5 * (
        particles['rlnDefocusU'].to_numpy() + particles['rlnDefocusV'].to_numpy()
    )
    return SimpleNamespace(
        dataset=dataset,
        count=len(particles),
        image_names=particles['rlnImageName'],
        rotations=rotations,
        shifts=origin / pixel_size,
        defocus=defocus,
        pixel_size=pixel_size,
        box_size=int(optics['rlnImageSize']),
        ctf_context=ctf.CtfContext(
            pixel_size_a=pixel_size,
            spherical_aberration_mm=float(optics['rlnSphericalAberration']),
            voltage_kv=float(optics['rlnVoltage']),
            q0=float(optics['rlnAmplitudeContrast']),
        ),
    )


def make_jobs(particles: SimpleNamespace, config: dict) -> List[analysis.Job]:
    shannon_angle = config['resolution'] / particles.dataset.diameter
    spacing = config['direction_index'] * shannon_angle
    aperture = config['aperture_index'] * shannon_angle

    count = geometry.estimate_projection_direction_count(spacing)
    directions = geometry.sample_projection_directions(count)
    matrices = geometry.euler_zyz_to_matrix(
        directions[:, 0], directions[:, 1], np.array(0)
    )
    groups = geometry.group_projection_directions(
        particles.rotations[:, 2, :], matrices[:, 2, :], aperture
    )
    bad = np.array(particles.dataset.bad, dtype=int)

    groups = [np.setdiff1d(group, bad) for group in groups]  # Also sorts.
    kept = [i for i, group in enumerate(groups) if len(group) >= config['min_particles']]
    if config.get('cover', False):
        # Particles outside every populated cone join the closest one.
        covered = np.zeros(particles.count, dtype=bool)
        for i in kept:
            covered[groups[i]] = True
        covered[bad] = True
        orphans = np.flatnonzero(~covered)
        if len(orphans):
            cosine = np.abs(particles.rotations[orphans, 2, :] @ matrices[kept, 2, :].T)
            closest = np.argmax(cosine, axis=1)
            for j, i in enumerate(kept):
                groups[i] = np.union1d(groups[i], orphans[closest == j])
    return [analysis.Job(i, groups[i], matrices[i]) for i in kept]


# -------------------------------------------------------- preprocessing ----

def _soft_disc(box_size: int, radius: float, edge: float = 6.0) -> np.ndarray:
    y, x = np.mgrid[0:box_size, 0:box_size] - box_size / 2
    t = np.clip((np.hypot(x, y) - radius) / edge, 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)


class Grid:
    """The Fourier coefficients kept from every image."""

    def __init__(self, particles: SimpleNamespace, config: dict):
        self.box_size = round(config['padding'] * particles.box_size)
        max_freq = particles.pixel_size / config['resolution']  # [1/px]
        ky = np.fft.fftfreq(self.box_size)[:, None]
        kx = np.fft.rfftfreq(self.box_size)[None, :]
        radius = np.hypot(ky, kx)
        keep = radius <= min(config['support'] * max_freq, 0.5)

        self.selection = np.flatnonzero(keep)
        self.frequency2 = np.square(radius[keep] / particles.pixel_size)  # [1/A2]
        self.shell = np.round(radius[keep] * self.box_size).astype(np.int32)
        self.shell_count = int(self.shell.max()) + 1
        multiplicity = np.full(radius.shape, 2.0)
        multiplicity[:, 0] = 1.0
        if self.box_size % 2 == 0:
            multiplicity[:, -1] = 1.0
        self.multiplicity = multiplicity[keep].astype(np.float32)
        self.envelope = None
        if config['butterworth']:
            self.envelope = (
                1.0 / (1.0 + np.square(np.square(radius[keep] / max_freq)))
            ).astype(np.float32)

        self.mask = None
        self.background = None
        if config['mask'] is not None:
            mask_radius = 0.5 * config['mask'] * particles.dataset.diameter
            mask_radius = mask_radius / particles.pixel_size
            self.mask = _soft_disc(particles.box_size, mask_radius)
            self.background = 1.0 - self.mask
            # Noise power of a masked image over that of its background.
            self.background_gain = float(
                np.square(self.mask).sum() / np.square(self.background).sum()
            )


def _to_fourier(images: jax.Array, box_size: int, selection: jax.Array) -> jax.Array:
    """Selected Fourier coefficients, with the image centre as origin."""
    h, w = images.shape[-2:]
    if box_size >= h:
        images = jnp.pad(images, ((0, 0), (0, box_size - h), (0, box_size - w)))
        images = jnp.roll(images, (-(h // 2), -(w // 2)), axis=(-2, -1))
    else:
        top = h // 2 - box_size // 2
        left = w // 2 - box_size // 2
        images = images[:, top:top + box_size, left:left + box_size]
        images = jnp.roll(images, (-(box_size // 2), -(box_size // 2)), axis=(-2, -1))
    images_ft = jnp.fft.rfft2(images) / box_size
    return images_ft.reshape(len(images_ft), -1)[:, selection]


@partial(jax.jit, static_argnames=('box_size', 'ctf_context'))
def _preprocess(
    images, rotations, shifts, defocus, direction_matrix,
    mask, background, selection, frequency2,
    box_size: int, ctf_context: ctf.CtfContext,
):
    in_plane = _in_plane_rotation(direction_matrix, rotations)
    aligned = jax.vmap(_align_single)(images, in_plane, shifts)
    aligned = jnp.nan_to_num(aligned)

    background_ft = None
    if mask is not None:
        background_ft = _to_fourier(aligned * background, box_size, selection)
        aligned = aligned * mask
    images_ft = _to_fourier(aligned, box_size, selection)

    wavelength = ctf_context.wavelength_a
    angle = jnp.pi * wavelength * frequency2 * (
        0.5 * ctf_context.spherical_aberration_a * wavelength**2 * frequency2
        - defocus[:, None]
    )
    ctfs = -jnp.sin(angle - ctf_context.amplitude_contrast_phase_shift)

    # View direction of each particle in the frame of the reference direction.
    # Opposite views are mirror images, equivalent to the negated direction.
    delta = _matmul(direction_matrix, rotations.swapaxes(-2, -1))
    offsets = jnp.sign(delta[:, 2, 2])[:, None] * delta[:, :2, 2]
    return images_ft, ctfs, background_ft, offsets


# ------------------------------------------------- mean and noise model ----

def _regressors(offsets: jax.Array, order: int) -> jax.Array:
    """Polynomial of the in-cone view offsets, scaled to unit magnitude."""
    scale = jnp.sqrt(jnp.mean(jnp.square(offsets))) + 1e-12
    x, y = (offsets / scale).T
    columns = [jnp.ones_like(x)]
    if order >= 1:
        columns += [x, y]
    if order >= 2:
        columns += [x * x - 1.0, y * y - 1.0, x * y]
    return jnp.stack(columns, axis=1)


def _remove_mean(images_ft, ctfs, regressors, valid, ridge: float = 1e-3):
    """Residual of the CTF-aware least squares fit of a mean that depends on
    the regressors: image = ctf * (regressors @ maps) + noise."""
    weight = valid[:, None]
    gram = jnp.einsum('nd,nr,ns->drs', jnp.square(ctfs) * weight, regressors, regressors)
    rhs = jnp.einsum('nd,nr->dr', ctfs * images_ft * weight, regressors)
    scale = jnp.mean(jnp.trace(gram, axis1=-2, axis2=-1)) / gram.shape[-1]
    gram = gram + ridge * scale * jnp.eye(gram.shape[-1])
    maps = jnp.linalg.solve(gram.astype(rhs.dtype), rhs[..., None])[..., 0]
    return images_ft - ctfs * (regressors.astype(maps.dtype) @ maps.T), maps


def _loadings(x, c, y, valid, ridge: float = 1e-3, excess=None):
    """Eigenimages of an embedding: CTF-aware least squares fit of
    x_i = c_i * (W y_i), feature by feature. ``excess`` is the covariance of
    the error of y, added to its second moments as in an M step."""
    k = y.shape[1]
    y = y * valid[:, None]
    outer = y[:, :, None] * y[:, None, :]
    if excess is not None:
        outer = outer + excess * valid[:, None, None]
    outer = outer.reshape(len(y), k * k)
    lhs = (jnp.square(c).T @ outer).reshape(-1, k, k)
    rhs = (c * x).T @ y
    lhs = lhs + ridge * jnp.sum(valid) * jnp.eye(k, dtype=x.dtype)
    return jnp.linalg.solve(lhs, rhs[..., None])[..., 0]


def _posterior_means(x, c, loadings, valid):
    """E[z | x] for x = c * (W z) + unit noise, z ~ N(0, I)."""
    k = loadings.shape[1]
    outer = (loadings[:, :, None] * loadings[:, None, :]).reshape(len(loadings), k * k)
    precision = (jnp.square(c) @ outer).reshape(-1, k, k) + jnp.eye(k, dtype=x.dtype)
    rhs = (c * x) @ loadings
    return jnp.linalg.solve(precision, rhs[..., None])[..., 0] * valid[:, None]


def _shell_power(x_ft, valid, multiplicity, shell, shell_count: int):
    """Mean power of the valid rows in each Fourier shell."""
    power = jnp.sum(
        (jnp.square(x_ft.real) + jnp.square(x_ft.imag)) * valid[:, None], axis=0
    ) / jnp.sum(valid)
    total = jnp.bincount(shell, weights=multiplicity * power, length=shell_count)
    norm = jnp.bincount(shell, weights=multiplicity, length=shell_count)
    return total / jnp.maximum(norm, 1.0)


def _as_real(x_ft: jax.Array, scale: jax.Array) -> jax.Array:
    """Real and imaginary parts as separate features."""
    return jnp.concatenate((x_ft.real * scale, x_ft.imag * scale), axis=1)


# ------------------------------------------------------------- methods ----

def _order_components(y: jax.Array, valid: jax.Array) -> jax.Array:
    """Rotate so that components are uncorrelated and sorted by power."""
    covariance = (y * valid[:, None]).T @ y / jnp.sum(valid)
    _, vectors = jnp.linalg.eigh(covariance)
    return y @ vectors[:, ::-1]


def _top_singular(x: jax.Array, k: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Leading singular triplets of a (rows, features) matrix through the
    smaller of its two Gram matrices."""
    n, d = x.shape
    if n <= d:
        values, vectors = jnp.linalg.eigh(x @ x.T)
        s = jnp.sqrt(jnp.maximum(values[::-1][:k], 1e-20))
        u = vectors[:, ::-1][:, :k]
        return u, s, (x.T @ u) / s
    values, vectors = jnp.linalg.eigh(x.T @ x)
    s = jnp.sqrt(jnp.maximum(values[::-1][:k], 1e-20))
    v = vectors[:, ::-1][:, :k]
    return (x @ v) / s, s, v


def _wiener(x, c, valid, inv_snr: float):
    """Wiener-like CTF correction with a constant regularizer, as factorem."""
    c2 = jnp.square(c)
    return (x * c) / (c2 + inv_snr * jnp.mean(c2, axis=1, keepdims=True)) * valid[:, None]


@partial(jax.jit, static_argnames=('k', 'scaling'))
def wiener_pca(x, c, valid, k: int, scaling: str = 'singular', inv_snr: float = 0.1):
    u, s, _ = _top_singular(_wiener(x, c, valid, inv_snr), k)
    if scaling == 'singular':
        return u * s
    if scaling == 'white':
        return u * jnp.sqrt(jnp.sum(valid))
    raise ValueError(scaling)


@partial(jax.jit, static_argnames=('k', 'iterations', 'split', 'shrink', 'model'))
def ppca(x, c, valid, half, k: int, iterations: int = 30, inv_snr: float = 0.1,
         ridge: float = 1e-3, split: bool = False, shrink: bool = True,
         model: bool = False):
    """Probabilistic PCA with the CTF in the forward model, fitted by
    expectation-maximization: x_i = c_i * (V z_i) + noise, z_i ~ N(0, I),
    with unit-variance white noise. Returns the posterior means of z.

    With ``split``, the posterior means given only the features where
    ``half`` is 0 and only those where it is 1 are appended: two estimates
    of z with independent noise.
    """
    count = jnp.sum(valid)
    weight = valid[:, None]
    c2 = jnp.square(c) * weight
    cx = c * x * weight
    eye = jnp.eye(k, dtype=x.dtype)

    _, s, v = _top_singular(_wiener(x, c, valid, inv_snr), k)
    loadings = v * (s / jnp.sqrt(count))

    def posterior(loadings, select=None):
        outer = (loadings[:, :, None] * loadings[:, None, :]).reshape(len(loadings), k * k)
        if select is not None:
            outer = outer * select[:, None]
            loadings = loadings * select[:, None]
        information = (c2 @ outer).reshape(-1, k, k)
        precision = information + (eye if shrink else 1e-6 * eye)
        covariance = jnp.linalg.inv(precision)
        mean = jnp.einsum('nkl,nl->nk', covariance, cx @ loadings)
        return mean, covariance

    def step(_, loadings):
        mean, covariance = posterior(loadings)
        second = (mean[:, :, None] * mean[:, None, :] + covariance) * valid[:, None, None]
        lhs = (c2.T @ second.reshape(-1, k * k)).reshape(-1, k, k)
        rhs = cx.T @ mean
        return jnp.linalg.solve(lhs + ridge * count * eye, rhs[..., None])[..., 0]

    loadings = jax.lax.fori_loop(0, iterations, step, loadings)

    # Fix the rotation of the latent space: uncorrelated, sorted by power.
    mean, _ = posterior(loadings)
    _, rotation = jnp.linalg.eigh((mean * weight).T @ mean / count)
    rotation = rotation[:, ::-1]
    out = [mean @ rotation]
    if split:
        for value in (0.0, 1.0):
            mean, _ = posterior(loadings, (half == value).astype(x.dtype))
            out.append(mean @ rotation)
    if model:
        return jnp.concatenate(out, axis=1), loadings @ rotation
    return jnp.concatenate(out, axis=1)


def _double_filter_distance2(x, c):
    """|c_j x_i - c_i x_j|^2 as in ManifoldEM."""
    c2 = jnp.square(c)
    term1 = jnp.square(x) @ c2.T
    cx = c * x
    return jnp.maximum(term1 + term1.T - 2.0 * (cx @ cx.T), 0.0)


def _euclidean_distance2(x):
    norm2 = jnp.sum(jnp.square(x), axis=1)
    return jnp.maximum(norm2[:, None] + norm2[None, :] - 2.0 * (x @ x.T), 0.0)


def _rowwise(function, rows: int, chunk: int = 32):
    """Evaluate ``function(i)`` for every row, a few rows at a time."""
    return jax.lax.map(function, jnp.arange(rows), batch_size=chunk)


def _normalized_double_filter_distance2(x, c, epsilon: float = 1e-3):
    """Double-filter distance with every coefficient divided by its noise
    variance c_i^2 + c_j^2: a Mahalanobis distance without signal prior."""
    c2 = jnp.square(c)

    def row(i):
        difference = c * x[i] - c[i] * x
        return jnp.sum(jnp.square(difference) / (c2 + c2[i] + epsilon), axis=1)

    return _rowwise(row, len(x))


def _signal_prior(x, c, valid, shell_of_feature, shell_count: int, floor: float = 1e-3):
    """Radial power of the signal behind unit-noise features, by least
    squares on E[x^2] = c^2 tau^2 + 1."""
    c2 = jnp.square(c) * valid[:, None]
    num = jnp.sum(c2 * (jnp.square(x) - 1.0), axis=0)
    den = jnp.sum(jnp.square(c2), axis=0)
    num = jnp.bincount(shell_of_feature, weights=num, length=shell_count)
    den = jnp.bincount(shell_of_feature, weights=den, length=shell_count)
    tau2 = jnp.maximum(num / jnp.maximum(den, 1e-12), floor)
    return tau2[shell_of_feature]


def _mahalanobis_distance2(x, c, tau2):
    """Distance between posterior means over the sum of the posterior
    variances (Bhamre, Zhao & Singer 2017), with a diagonal prior tau2."""
    variance = tau2 / (jnp.square(c) * tau2 + 1.0)
    mean = variance * c * x

    def row(i):
        return jnp.sum(jnp.square(mean - mean[i]) / (variance + variance[i]), axis=1)

    return _rowwise(row, len(x))


def _posterior_mean(x, c, tau2):
    """Wiener filter with the estimated radial signal power."""
    return tau2 * c * x / (jnp.square(c) * tau2 + 1.0)


def _classical_mds(distance2, valid, k: int):
    weight = valid / jnp.sum(valid)
    pair = valid[:, None] * valid[None, :]
    row_mean = distance2 @ weight
    total_mean = row_mean @ weight
    gram = -0.5 * (distance2 - row_mean[:, None] - row_mean[None, :] + total_mean) * pair
    values, vectors = jnp.linalg.eigh(gram)
    values = jnp.maximum(values[::-1][:k], 0.0)
    return vectors[:, ::-1][:, :k] * jnp.sqrt(values)


def _diffusion_map(distance2, valid, k: int, kernel: str, alpha: float, time_: float,
                   neighbours: int, scale: float):
    from factorem.analysis import spectral_embedding as se
    valid = valid > 0
    if kernel == 'median':
        pair = valid[:, None] & valid[None, :] & ~jnp.eye(len(valid), dtype=bool)
        sigma2 = jnp.nanmedian(jnp.where(pair, distance2, jnp.nan))
        affinity = jnp.exp(-distance2 / (scale * sigma2))
    elif kernel == 'excess':
        # Bandwidth from the spread of the distances rather than their size,
        # as the noise adds a large constant to all of them.
        pair = valid[:, None] & valid[None, :] & ~jnp.eye(len(valid), dtype=bool)
        d = jnp.where(pair, distance2, jnp.nan)
        low = jnp.nanpercentile(d, 1.0)
        sigma2 = jnp.nanmedian(d) - low
        affinity = jnp.exp(-jnp.maximum(distance2 - low, 0.0) / (scale * sigma2))
    elif kernel == 'local':
        masked = jnp.where(valid[None, :], distance2, jnp.inf)
        sigma = jnp.sqrt(jnp.partition(masked, neighbours, axis=1)[:, neighbours])
        affinity = jnp.exp(-distance2 / (scale * jnp.outer(sigma, sigma)))
    else:
        raise ValueError(kernel)
    return se._spectral_embedding(affinity, valid, k, alpha=alpha, diffusion_time=time_)


@partial(jax.jit, static_argnames=(
    'k', 'distance', 'reduction', 'kernel', 'shell_count', 'scaling'))
def distance_embedding(
    x, c, valid, shell_of_feature, k: int, shell_count: int,
    distance: str = 'double', reduction: str = 'diffusion',
    kernel: str = 'median', alpha: float = 1.0, time_: float = 1.0,
    neighbours: int = 7, scale: float = 1.0, inv_snr: float = 0.1,
    scaling: str = 'native',
):
    if distance == 'double':
        distance2 = _double_filter_distance2(x, c)
    elif distance == 'ndouble':
        distance2 = _normalized_double_filter_distance2(x, c)
    elif distance == 'wiener':
        distance2 = _euclidean_distance2(_wiener(x, c, valid, inv_snr))
    elif distance == 'posterior':
        tau2 = _signal_prior(x, c, valid, shell_of_feature, shell_count)
        distance2 = _euclidean_distance2(_posterior_mean(x, c, tau2))
    elif distance == 'mahalanobis':
        tau2 = _signal_prior(x, c, valid, shell_of_feature, shell_count)
        distance2 = _mahalanobis_distance2(x, c, tau2)
    else:
        raise ValueError(distance)

    if reduction == 'diffusion':
        y = _diffusion_map(distance2, valid, k, kernel, alpha, time_, neighbours, scale)
    elif reduction == 'mds':
        y = _classical_mds(distance2, valid, k)
    else:
        raise ValueError(reduction)
    if scaling == 'white':
        y = y - jnp.sum(y * valid[:, None], axis=0) / jnp.sum(valid)
        y = y / jnp.sqrt(jnp.sum(jnp.square(y) * valid[:, None], axis=0) / jnp.sum(valid))
    return y


# -------------------------------------------------------------- stage A ----

def _embed_group(batch, job, count: int, particles, grid: Grid, config: dict,
                 consensus=None, excess=None):
    images_ft, ctfs, background_ft, offsets = _preprocess(
        batch['image'], batch['rotation'], batch['shift'], batch['defocus'],
        jnp.asarray(job.direction_matrix, dtype=jnp.float32),
        grid.mask, grid.background, grid.selection, grid.frequency2,
        box_size=grid.box_size, ctf_context=particles.ctf_context,
    )
    return _embed_fourier(
        images_ft, ctfs, background_ft, offsets, count,
        grid.multiplicity, grid.shell, grid.envelope,
        getattr(grid, 'background_gain', 1.0),
        shell_count=grid.shell_count,
        config=_freeze(config), consensus=consensus, excess=excess,
    )


def _freeze(config: dict) -> tuple:
    return tuple(sorted(
        (k, _freeze(v) if isinstance(v, dict) else v) for k, v in config.items()
    ))


@partial(jax.jit, static_argnames=('shell_count', 'config'))
def _embed_fourier(
    images_ft, ctfs, background_ft, offsets, count,
    multiplicity, shell, envelope, background_gain,
    shell_count: int, config: tuple, consensus=None, excess=None,
):
    config = dict(config)
    options = dict(config['options'])
    valid = (jnp.arange(len(images_ft)) < count).astype(jnp.float32)

    regressors = _regressors(offsets, config['tilt_order'])
    residual, maps = _remove_mean(images_ft, ctfs, regressors, valid)

    scale = jnp.sqrt(multiplicity)
    if envelope is not None:
        scale = scale * envelope
    if config['whiten']:
        if config['noise'] == 'background':
            noise = background_gain * _shell_power(
                background_ft, valid, multiplicity, shell, shell_count)
        else:
            noise = _shell_power(residual, valid, multiplicity, shell, shell_count)
        # Unit variance for the real and imaginary parts of plain coefficients.
        scale = scale / jnp.sqrt(jnp.maximum(noise[shell], 1e-20))

    x = _as_real(residual, scale) * valid[:, None]
    c = jnp.concatenate((ctfs, ctfs), axis=1)
    shell_of_feature = jnp.concatenate((shell, shell))
    k = config['components']

    if consensus is not None:
        # Refinement: eigenimages from the consensus coordinates of the
        # particles, then their posterior means, already in the common frame.
        w = _loadings(x, c, consensus, valid, excess=excess)
        return _posterior_means(x, c, w, valid)

    method = config['method']
    if method == 'ppca':
        half = (shell_of_feature // config['band_width']) % 2
        y = ppca(x, c, valid, half, k, **options)
        if options.get('model', False):
            y, model_loadings = y
    elif method == 'wiener_pca':
        y = wiener_pca(x, c, valid, k, **options)
    elif method == 'distance':
        y = distance_embedding(
            x, c, valid, shell_of_feature, k, shell_count, **options)
    else:
        raise ValueError(method)
    if not config.get('loadings', False):
        return y

    # Eigenimages and mean as complex Fourier coefficients, in noise units.
    w = _loadings(x, c, y[:, :k], valid)
    if method == 'ppca' and options.get('model', False):
        w = model_loadings  # Those of the fitted model instead.
    d = len(shell)
    unit = scale / jnp.sqrt(multiplicity)
    w = (w[:d] + 1j * w[d:]) / jnp.sqrt(multiplicity)[:, None]
    return y, w, maps[:, 0] * unit


def stage_a(name: str, config: dict, particles=None, cache: bool = True, log=print):
    """Per-direction embeddings as (jobs, list of arrays, seconds)."""
    config = {**STAGE_A_DEFAULTS, **config}
    key = hashlib.sha1(json.dumps([name, config], sort_keys=True).encode()).hexdigest()[:16]
    path = f'{CACHE}/{name}_{key}.npz'
    particles = particles or load(name)
    jobs = make_jobs(particles, config)

    if cache and os.path.exists(path):
        stored = np.load(path)
        results = np.split(stored['values'], stored['indptr'][1:-1])
        return jobs, results, float(stored['seconds'])
    export = config.get('loadings', False)
    loadings, means = [], []

    import vitrio as vio
    grid = Grid(particles, config)
    dataset = analysis.make_dataset(
        particles.image_names, particles.rotations, particles.shifts,
        particles.defocus, particles.dataset.prefix,
    )
    device = jax.devices()[0]
    grain = 256
    batches = vio.data.DataLoader(
        dataset,
        batch_sampler=[
            np.resize(job.indices, -(-len(job.indices) // grain) * grain)
            for job in jobs
        ],
        output='jax', device=device, prefetch=4,
    )
    start = time.time()
    results = []
    pending = []
    for i, (job, batch) in enumerate(zip(jobs, batches)):
        y = _embed_group(batch, job, len(job.indices), particles, grid, config)
        if export:
            y, w, mean = y
            loadings.append(w)
            means.append(mean)
        pending.append((len(job.indices), y))
        if len(pending) > 2:
            n, y = pending.pop(0)
            results.append(np.asarray(y)[:n])
        if log and i % 200 == 0:
            log(f'  {name} {i}/{len(jobs)} {time.time() - start:.0f}s', flush=True)
    for n, y in pending:
        results.append(np.asarray(y)[:n])
    seconds = time.time() - start

    os.makedirs(CACHE, exist_ok=True)
    np.savez(
        path,
        values=np.concatenate(results),
        indptr=np.cumsum([0] + [len(r) for r in results]),
        seconds=seconds,
        config=json.dumps(config),
        **(dict(loadings=np.stack([np.asarray(w) for w in loadings]),
                means=np.stack([np.asarray(m) for m in means])) if export else {}),
    )
    return jobs, results, seconds


def stage_c(name: str, config: dict, particles, jobs, z: np.ndarray, log=None):
    """Refine a consensus embedding ``z`` (count, k): every direction fits
    its eigenimages to the consensus coordinates of its particles and embeds
    them again. Returns the average over directions and the seconds taken."""
    import vitrio as vio
    config = {**STAGE_A_DEFAULTS, **config}
    config.pop('loadings', None)
    z = np.nan_to_num(z)
    z = z - z.mean(axis=0)
    covariance = z.T @ z / len(z)
    # Law of total variance under a unit prior: what the means lack.
    values, vectors = np.linalg.eigh(np.eye(z.shape[1]) - covariance)
    excess = jnp.asarray((vectors * np.maximum(values, 0.0)) @ vectors.T, dtype=jnp.float32)
    z = z.astype(np.float32)

    grid = Grid(particles, config)
    dataset = analysis.make_dataset(
        particles.image_names, particles.rotations, particles.shifts,
        particles.defocus, particles.dataset.prefix,
    )
    device = jax.devices()[0]
    grain = 256
    padded = [np.resize(job.indices, -(-len(job.indices) // grain) * grain) for job in jobs]
    batches = vio.data.DataLoader(
        dataset, batch_sampler=padded, output='jax', device=device, prefetch=4)
    start = time.time()
    total = np.zeros(z.shape)
    counts = np.zeros(len(z))
    pending = []

    def collect():
        job, y = pending.pop(0)
        total[job.indices] += np.asarray(y)[:len(job.indices)]
        counts[job.indices] += 1

    for i, (job, indices, batch) in enumerate(zip(jobs, padded, batches)):
        y = _embed_group(batch, job, len(job.indices), particles, grid, config,
                         consensus=jnp.asarray(z[indices]), excess=excess)
        pending.append((job, y))
        if len(pending) > 2:
            collect()
        if log and i % 200 == 0:
            log(f'  refine {name} {i}/{len(jobs)} {time.time() - start:.0f}s', flush=True)
    while pending:
        collect()
    return total / np.maximum(counts, 1)[:, None], time.time() - start


def eigenimages(name: str, config: dict, particles=None):
    """Eigenimages (directions, features, components) and means (directions,
    features) of a stage A run with ``loadings=True``, with the integer
    (kx, ky) frequency of every feature."""
    config = {**STAGE_A_DEFAULTS, **config}
    key = hashlib.sha1(json.dumps([name, config], sort_keys=True).encode()).hexdigest()[:16]
    stored = np.load(f'{CACHE}/{name}_{key}.npz')
    grid = Grid(particles or load(name), config)
    ky, kx = np.unravel_index(grid.selection, (grid.box_size, grid.box_size // 2 + 1))
    ky = np.where(ky > grid.box_size // 2, ky - grid.box_size, ky)
    return stored['loadings'], stored['means'], np.stack((kx, ky), axis=1)


# -------------------------------------------------------------- stage B ----

def stack(jobs, results, count: int) -> scipy.sparse.bsr_array:
    k = results[0].shape[1]
    return scipy.sparse.bsr_array(
        (
            np.concatenate(results).astype(np.float64)[:, :, None],
            np.concatenate([job.indices for job in jobs]),
            np.cumsum([0] + [len(job.indices) for job in jobs]),
        ),
        shape=(len(jobs) * k, count),
    )


def _similarities(left, right, k: int):
    out = left @ right.T
    return (0.5 * (out + out.T)).tobsr(blocksize=(k, k))


def _whiten(e: np.ndarray) -> np.ndarray:
    e = e - e.mean(axis=0)
    values, vectors = np.linalg.eigh(e.T @ e / len(e))
    return (e @ (vectors / np.sqrt(values))) @ vectors.T


def _common_line_sync(jobs, results, count: int, config: dict, eigen):
    """Orthogonal transforms from the common lines of the eigenimages."""
    loadings, _, frequencies = eigen
    k = results[0].shape[1]
    matrices = np.stack([job.direction_matrix for job in jobs])
    rng = np.random.default_rng(config['seed'])
    similarities = commonlines.similarities(
        loadings[..., :k], frequencies, matrices, config['partners'], rng=rng)
    similarities /= abs(similarities).max()
    rank = config['rank'] or k
    start = sync.polar(rng.normal(size=(len(jobs), k, rank)))
    transforms, iterations = sync.power_iterations(
        similarities, start, config['max_iter'], 1e-6)
    return similarities, transforms, iterations


def reduce(unified: np.ndarray, components: int) -> np.ndarray:
    """PCA of the covered particles."""
    covered = np.isfinite(unified).all(axis=1)
    out = np.full((len(unified), components), np.nan)
    pca = sklearn.decomposition.PCA(n_components=components)
    out[covered] = pca.fit_transform(unified[covered])
    return out


def stage_b(jobs, results, count: int, config: dict, eigen=None) -> Tuple[np.ndarray, dict]:
    config = {**STAGE_B_DEFAULTS, **config}
    if config['solver'] == 'common_lines':
        t0 = time.time()
        k = results[0].shape[1]
        similarities, transforms, iterations = _common_line_sync(
            jobs, results, count, config, eigen)
        info = {'sync_seconds': time.time() - t0, 'sync_iterations': iterations,
                'objective': sync.objective(similarities, transforms)}
        embeddings = stack(jobs, results, count)
        unified = synchronization.average_embeddings(
            synchronization.correct_embeddings(embeddings, transforms), max_iter=0)
        info['unified'] = unified
        return reduce(unified, config['components'] or k), info
    k = config['k'] if config.get('k') else results[0].shape[1]
    if config['split']:
        k = results[0].shape[1] // 3
        halves = [stack(jobs, [r[:, i * k:(i + 1) * k] for r in results], count) for i in (1, 2)]
    results = [r[:, :k] for r in results]
    m = len(jobs)
    embeddings = stack(jobs, results, count)
    info = {}

    if config['split']:
        similarities = _similarities(halves[0], halves[1], k)
    else:
        similarities = _similarities(embeddings, embeddings, k)
    if config['normalize'] == 'overlap':
        # Mean instead of sum over the particles two directions share.
        member = scipy.sparse.csr_array(
            (np.ones(len(embeddings.indices)), embeddings.indices, embeddings.indptr),
            shape=(m, count),
        )
        overlap = (member @ member.T).tocsr()
        overlap.data = 1.0 / overlap.data
        scale = scipy.sparse.kron(overlap, np.ones((k, k))).tobsr(blocksize=(k, k))
        similarities = similarities.multiply(scale).tobsr(blocksize=(k, k))
    if not config['diagonal']:
        blocks = scipy.sparse.kron(
            scipy.sparse.eye(m), np.ones((k, k))).tobsr(blocksize=(k, k))
        similarities = (similarities - similarities.multiply(blocks)).tobsr(blocksize=(k, k))
    similarities.sort_indices()
    similarities /= abs(similarities).max()

    t0 = time.time()
    if config['solver'] == 'burer_monteiro':
        rank = config['rank'] or 2 * k + 1
        start = synchronization.burer_monteiro_random_start(
            n=m, k=k, p=rank, rng=np.random.default_rng(config['seed']))
        transforms, eigenvalues = synchronization.burer_monteiro_ortho_group_synchronization(
            similarities, start, tol=config['tol'], max_iter=config['max_iter'])
    elif config['solver'] == 'multigrid':
        transforms = sync.multigrid(
            similarities, config['smoothing'], np.random.default_rng(config['seed']))
        if config['refine']:
            transforms, info['refine_iterations'] = sync.power_iterations(
                similarities, transforms, config['refine'])
    else:
        raise ValueError(config['solver'])
    info['sync_seconds'] = time.time() - t0
    info['objective'] = sync.objective(similarities, transforms)

    corrected = synchronization.correct_embeddings(embeddings, transforms)
    unified = synchronization.average_embeddings(
        corrected, max_iter=config['gain_iterations'])
    covered = np.isfinite(unified).all(axis=1)
    out = np.full((count, config['components'] or k), np.nan)
    pca = sklearn.decomposition.PCA(n_components=out.shape[1])
    out[covered] = pca.fit_transform(unified[covered])
    return out, info


# -------------------------------------------------------------- scoring ----

_REFERENCES = {}


def references(name: str) -> dict:
    if name not in _REFERENCES:
        path = datasets.DATASETS[name].recovar
        _REFERENCES[name] = {}
        if os.path.exists(path):
            zs = np.load(path)
            _REFERENCES[name] = {'recovar4': zs['z4'], 'recovar10': zs['z10']}
    return _REFERENCES[name]


def nuisances(particles: SimpleNamespace) -> dict:
    return {
        'defocus': particles.defocus,
        'view': particles.rotations[:, 2, :] * np.sign(particles.rotations[:, 2, 2:3]),
    }


def local_scores(name: str, jobs, results, count: int, stride: int = 8, k: int = 10) -> dict:
    """Quality of the per-direction embeddings before any synchronization,
    averaged over a subset of the directions."""
    truth = datasets.ground_truth(name, count)
    values = []
    for job, y in list(zip(jobs, results))[::stride]:
        if len(y) <= 4 * k:
            continue
        nn = metrics.neighbours(y, k)
        if name == 'igg':
            values.append(metrics.circular_error(nn, truth['state'][job.indices], 100))
        elif name == 'e10076':
            labels = truth['major'][job.indices]
            keep = labels >= 0
            nn = metrics.neighbours(y[keep], k)
            values.append(metrics.label_accuracy(nn, labels[keep])['acc'])
        else:
            labels = truth['kept'][job.indices]
            values.append(metrics.label_accuracy(nn, labels)['bal_acc'])
    return {'local': float(np.mean(values))}


def run(name: str, config_a: dict, config_b: Optional[dict] = None, particles=None,
        tag: str = '', log=print, local: bool = True, save: Optional[str] = None,
        refine: int = 0) -> dict:
    particles = particles or load(name)
    jobs, results, seconds = stage_a(name, config_a, particles, log=None)
    out = {'dataset': name, 'tag': tag, 'directions': len(jobs), 'a_seconds': round(seconds, 1)}
    if local:
        out.update(local_scores(name, jobs, results, particles.count))
    t0 = time.time()
    eigen = None
    if config_a.get('loadings'):
        eigen = eigenimages(name, config_a, particles)
    z, info = stage_b(jobs, results, particles.count, config_b or {}, eigen)
    out['sync_seconds'] = round(info['sync_seconds'], 1)
    out['b_seconds'] = round(time.time() - t0, 1)
    if refine:
        unified = info['unified']
        out['c_seconds'] = 0.0
        for _ in range(refine):
            unified, seconds = stage_c(name, config_a, particles, jobs, unified)
            out['c_seconds'] += round(seconds, 1)
        z = reduce(unified, z.shape[1])
    out.update(metrics.evaluate(
        name, z, datasets.ground_truth(name, particles.count),
        references(name), nuisances(particles),
    ))
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    if save:
        np.save(save, z)
    if log:
        log(json.dumps(out), flush=True)
    with open(f'{datasets.BENCH}/results.jsonl', 'a') as f:
        f.write(json.dumps({**out, 'a': config_a, 'b': config_b or {}, 'refine': refine}) + '\n')
    return out
