# factorem

Conformational and compositional heterogeneity of cryo-EM single particles,
from independent analyses of each projection direction that are then
synchronized.

factorem takes the particles of a consensus refinement and gives each one a
few coordinates that describe its state. It never reconstructs a volume.
Particles that view the molecule from about the same direction differ in
their state and in little else, so each direction is analysed on its own,
as a set of 2D images. Every direction ends up with coordinates in a frame
of its own, and a synchronization finds the orthogonal transform of each
that brings them all to a common one: by default, the transforms that make
the eigenimages of the directions agree along their common lines.

Analysing each projection direction by itself and relating the directions
afterwards is the idea of [ManifoldEM][manifoldem] (Dashti et al.,
[2014][dashti2014], [2020][dashti2020] and Ojha et al. [2025][ojha2025]), 
which factorem follows. It differs in how: each direction is embedded 
linearly, by a probabilistic PCA with the CTF in its model, and the 
directions are related all at once and in all their coordinates, by an 
orthogonal synchronization.

## Installation

```sh
pip install factorem
```

For an NVIDIA GPU, ask for the `cuda13` or the `cuda12` extra, which bring
the matching build of JAX:

```sh
pip install 'factorem[cuda13]'
```

It needs Python 3.10 or later. Images are read with
[vitrio](https://pypi.org/project/vitrio/) and everything is computed with
[JAX](https://docs.jax.dev).

## Usage

```sh
factorem analyze particles.star embedded.star --diameter 180
```

`particles.star` is a RELION STAR file of the particles, with their
alignment and CTF. The output is the same file with two columns more:
`factoremEmbedding`, the coordinates of the particle as a list such as
`[1.2e-01, -3.4e-01, ...]`, and `factoremGroupCount`, how many projection
directions it was analysed with.

| Option | Default | |
|---|---|---|
| `--diameter` | required | Diameter of the particle [Å]. |
| `--resolution` | 16 | Resolution of the analysis [Å]. |
| `--components` | 6 | Coordinates to give each particle. |
| `--embedding` | `ppca` | How the particles of each direction are embedded: `ppca`, by a probabilistic PCA with the CTF in its model, or `mahalanobis-mds`, by the multidimensional scaling of the Mahalanobis distances between the images. |
| `--synchronization` | `common-lines` | How the directions are brought to a common frame: `common-lines`, by the common lines of their eigenimages, or `burer-monteiro`, by the particles they share. |
| `--root` | working directory | Directory that the images named in the STAR file are relative to. |
| `--device` | first GPU, else CPU | Such as `cpu` or `gpu:1`. |
| `--scratch` | none | Directory to keep the working images in, instead of in memory. |
| `-v` | | Report the progress. |

The same from Python:

```python
import factorem

dataset = factorem.StarDataset.read('particles.star')
embedding = factorem.analyze(
    dataset.particles,
    diameter=180.0,
    embedder=factorem.ProbabilisticPCA(),  # The default.
    synchronizer=factorem.CommonLineSynchronizer(),  # The default.
)
dataset.write('embedded.star', embedding.coordinates, embedding.group_count)
```

`factorem.MahalanobisMDS()` is the other embedder, with the distances of
[Bhamre, Zhao and Singer (2017)][bhamre2017], and
`factorem.BurerMonteiroSynchronizer()` the other synchronization. The
stages that `analyze` chains are public as well, and
[examples/stages.py][stages] calls them one by one.

## Annotation

```sh
pip install 'factorem[annotate]'
factorem annotate embedded.star annotated.star
```

`factorem annotate` opens a window with the density of the embedded
particles on a plane, to sort them into classes there. The output is the
same file with one column more, `factoremClass`: the class of each
particle, from one on, and zero for the particles that the analysis left
out. With `--label rlnClassNumber` the classes go where RELION looks for
them.

The classes are the Gaussians of a mixture, and each particle is of the one
that most likely gave it.

| Control | |
|---|---|
| `x`, `y` | The coordinate along each axis of the plane. |
| Lay out by UMAP | Show the plane that [UMAP][umap] lays all the coordinates out on instead. It needs the `umap` extra. |
| `full`, `tied`, `diag`, `spherical` | What the covariances of the fitted Gaussians may be: any, one for all of them, aligned with the axes, or round. |
| Fit K classes | Fit a mixture of `K` Gaussians. |
| Fit as many as the BIC asks | Fit the mixture of least Bayesian information criterion, among those of up to 20 Gaussians. |
| Fit from the seeds | Fit a mixture of a Gaussian for each seed, starting from them. The classes keep the order of the seeds. |
| Take the seeds as they are | Make a round Gaussian of each seed, without fitting anything. |
| Save and close | Write the output. Nothing is written if the window is closed in any other way. |

A click on the plane places a seed, dragging a seed moves it, the wheel
over it changes its size and a right click removes it. The zoom and the pan
of the toolbar take the clicks while they are on, and the density is
counted again for what they leave in sight. The classes and the seeds
belong to the plane they were made on, so changing it drops them.

## Development

```sh
pip install -e '.[dev]'
pytest
ruff check . && ruff format --check .
mypy
```

## License

GPL-3.0. See [LICENSE][license].

[manifoldem]: https://github.com/flatironinstitute/ManifoldEM
[dashti2014]: https://doi.org/10.1073/pnas.1419276111
[dashti2020]: https://doi.org/10.1038/s41467-020-18403-x
[ojha2025]: https://journals.iucr.org/paper?sor5002
[bhamre2017]: https://doi.org/10.1109/ISBI.2017.7950605
[umap]: https://umap-learn.readthedocs.io
[stages]: https://github.com/oierlauzi/factorem/blob/main/examples/stages.py
[license]: https://github.com/oierlauzi/factorem/blob/main/LICENSE
