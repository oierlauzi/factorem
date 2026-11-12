# factorem

Continuous heterogeneity analysis of cryo-EM particles, one projection
direction at a time.

factorem takes the particles of a consensus refinement and gives each one a
few coordinates that describe its conformation and composition. It never
reconstructs a volume: particles that view the molecule from about the same
direction are compared as 2D images, and the results of all the directions
are then brought to a common frame. That makes it fast (minutes for
hundreds of thousands of particles on one GPU) and leaves it with only
physical inputs: the diameter of the particle and the resolution to look at.

## Installation

```sh
pip install git+https://github.com/oierlauzi/factorem
```

For an NVIDIA GPU, ask for the `cuda13` or the `cuda12` extra, which bring
the matching build of JAX:

```sh
pip install 'factorem[cuda13] @ git+https://github.com/oierlauzi/factorem'
```

It needs Python 3.10 or later. Images are read with
[vitrio](https://pypi.org/project/vitrio/) and everything is computed with
[JAX](https://docs.jax.dev).

## Command line

```sh
factorem particles.star embedded.star --diameter 180
```

`particles.star` is a RELION STAR file of the particles, with their
alignment and CTF. Both the layout with optics groups (RELION 3.1 and later)
and the older one are read. The output is the same file with two columns
more:

- `factoremEmbedding`, the coordinates of the particle, as a list such as
  `[1.2e-01, -3.4e-01, ...]`, by decreasing variance.
- `factoremGroupCount`, how many projection directions the particle was
  analysed with.

| Option | Default | |
|---|---|---|
| `--diameter` | required | Diameter of the particle [Å]. |
| `--resolution` | 16 | Resolution of the analysis [Å]. |
| `--components` | 6 | Coordinates to give each particle. |
| `--root` | working directory | Directory that the images named in the STAR file are relative to. |
| `--device` | first GPU, else CPU | Such as `cpu` or `gpu:1`. |
| `--scratch` | none | Directory to keep the working images in, instead of in memory. |
| `-v` | | Report the progress. |

Particles whose image is corrupt (not finite, constant, or of values beyond
what single precision can compute with) are reported, left out, and given
`nan` coordinates.

## Python

```python
import factorem

dataset = factorem.StarDataset.read('particles.star')
embedding = factorem.analyze(dataset.particles, diameter=180.0)

embedding.coordinates  # (particles, components)
embedding.variances  # along every axis found, the ones kept first
dataset.write('embedded.star', embedding.coordinates, embedding.group_count)
```

`analyze` chains stages that are public as well, and each can be replaced.
[examples/stages.py](examples/stages.py) calls them one by one:

| Stage | Function | Replace it with |
|---|---|---|
| Describe the particles | `StarDataset.read` | any `Particles` |
| Read the images, once, at a coarse sampling | `load_images` | |
| Group the particles by projection direction | `sample_directions`, `group_by_direction` | any `DirectionGroups` |
| Embed the particles of each direction | `embed_directions` | any `Embedder` (default `ProbabilisticPCA`) |
| Bring the directions to a common frame | `CommonLineSynchronizer` | any `Synchronizer` |
| Average them and keep the principal axes | `merge_directions`, `principal_components` | |

An `Embedder` is a callable that takes the images of a direction (whitened,
with their CTFs) and returns coordinates and eigenimages. A `Synchronizer`
takes the embeddings of all the directions and returns an orthogonal
transform for each. Both are protocols, so nothing has to be subclassed.

## How it works

1. **Images.** Every image is read once, centred with the translation of
   the refinement and Fourier cropped to four pixels per resolution
   element. What is kept fits in memory (24 GB for the 327,000
   particles of EMPIAR-10180), or goes to `--scratch`.
2. **Directions.** Projection directions are spread over half of the
   sphere, one *Shannon angle* apart: the ratio of the resolution to the
   diameter, which is how far the view can turn before the projection
   changes at that resolution. Each direction is analysed with the
   particles within two Shannon angles of it, the ones that view from the
   opposite side included, mirrored.
3. **Embedding of a direction.** Its particles are rotated in-plane to a
   common frame, masked to 1.25 diameters, and whitened with the noise
   spectrum measured outside of the mask. Their mean is fitted through
   their CTFs, as a quadratic function of the view within the cone, so
   that what changes with the view is not mistaken for heterogeneity. The
   rest is modelled by probabilistic PCA with the CTF in it,
   `image = CTF · (eigenimages · z) + noise` with `z ~ N(0, I)`, and the
   coordinates of a particle are the mean of `z` given its image. No
   image is ever CTF corrected or compared to another one, so there are
   no kernel widths or neighbour counts to choose.
4. **Synchronization.** Each direction finds its coordinates up to an
   orthogonal transform. The eigenimages of a direction are projections of
   the same *eigenvolumes*, so those of two directions agree along the
   line where their central slices meet, once transformed alike.
   Correlating them along these common lines, for 64 random partners of
   each direction, gives the transforms between pairs; a generalized
   power method finds the transform of each direction that agrees best
   with all of them.
5. **Merging.** The coordinates of each particle, which about a dozen
   directions have estimated, are averaged in the common frame and turned
   to their principal axes.

### Constants

The inputs are the diameter, the resolution and the number of components.
Everything else is a constant of `factorem.Design`, stated in units that
follow the particle (Shannon angles, diameters, pixels per resolution
element), and the same values are used for every dataset below. They can be
changed from Python, for experimenting, but none is meant to be tuned.

| Constant | Value | |
|---|---|---|
| `oversampling` | 4 | Pixels of the working images per resolution element. |
| `direction_spacing` | 1 | Angle between directions [Shannon angles]. |
| `aperture` | 2 | Half-angle of the cone of each direction [Shannon angles]. |
| `view_order` | 2 | Order of the polynomial of the mean image over the view. |
| `mask_scale` | 1.25 | Diameter of the mask over that of the particle. |
| `min_components` | 10 | Fewest components per direction, however few are asked for. |
| `images_per_component` | 10 | Fewest particles of a direction per component. |

## Benchmarks

Scored on the neighbours of each particle in the embedding, against what is
known of it: the state it was simulated in (IgG-1D of CryoBench, as the mean
distance to the states of its neighbours, of 100), its published class
(EMPIAR-10076) or whether cryoDRGN kept it (EMPIAR-10180), as the accuracy
of the vote of its neighbours. The same command for the three, with the
diameter as its only argument, and
[RECOVAR](https://github.com/ma-gilles/recovar) 0.4.5 for reference:

| | | factorem, 6 | factorem, 10 | RECOVAR, 4 | RECOVAR, 6 | RECOVAR, 10 |
|---|---|---|---|---|---|---|
| IgG-1D | `knn_err`, lower is better | 1.34 | 1.10 | 1.89 | 1.28 | 0.85 |
| EMPIAR-10076 | `major_acc` | 0.818 | 0.820 | 0.838 | | 0.843 |
| | `major_bal_acc` | 0.637 | 0.641 | 0.614 | | 0.632 |
| | `minor_acc` | 0.472 | 0.485 | 0.452 | | 0.552 |
| | `minor_bal_acc` | 0.407 | 0.423 | 0.373 | | 0.486 |
| EMPIAR-10180 | `kept_acc` | 0.641 | 0.647 | 0.629 | | 0.658 |
| | `kept_bal_acc` | 0.617 | 0.623 | 0.607 | | 0.642 |

| | Particles | Time | Peak memory |
|---|---|---|---|
| IgG-1D | 100,000 of 128 px | 55 s | 7 GB |
| EMPIAR-10076 | 131,899 of 320 px | 2.3 min | 11 GB |
| EMPIAR-10180 | 327,490 of 320 px | 5.3 min | 31 GB |

On one NVIDIA RTX PRO 4500, with the files in the cache of the system:
reading them from disk for the first time can take longer than the
analysis. RECOVAR took 42 min on IgG-1D.

[benchmarks](benchmarks/README.md) has the details, and what else was
tried: other image metrics (Wiener corrected, ManifoldEM, Mahalanobis),
diffusion maps and MDS, and the synchronization of the particles that
directions share.

## Development

```sh
pip install -e '.[dev]'
pytest                 # -m 'not slow' leaves out the end-to-end runs
ruff check . && ruff format --check .
mypy
```

The tests simulate their own particles and run on the CPU in under a
minute. The same checks run for every push and pull request, on every
supported Python and with the oldest supported dependencies, in
[GitHub Actions](.github/workflows/ci.yml).

```
src/factorem/
    particles.py        What the analysis needs to know of the particles
    geometry.py         Orientations, projection directions and their groups
    io/                 STAR files, and reading the images once
    imaging/            CTF, Fourier band, in-plane alignment
    embedding/          Embedding of each direction (Embedder, ProbabilisticPCA)
    synchronization/    Common frame (Synchronizer, common lines, power method)
    merging.py          Average over directions and principal axes
    pipeline.py         analyze, Design
    cli.py              Command line
benchmarks/             Scores against published labels and RECOVAR
```

## License

GPL-3.0. See [LICENSE](LICENSE).
