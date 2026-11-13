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
neighbouring directions agree on the particles they share.

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
| `--synchronization` | `burer-monteiro` | How the directions are brought to a common frame: `burer-monteiro`, by the particles they share, or `common-lines`, by the common lines of their eigenimages. |
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
    synchronizer=factorem.BurerMonteiroSynchronizer(),  # The default.
)
dataset.write('embedded.star', embedding.coordinates, embedding.group_count)
```

`factorem.CommonLineSynchronizer()` is the other synchronization. The
stages that `analyze` chains are public as well, and
[examples/stages.py][stages] calls them one by one.

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
[stages]: https://github.com/oierlauzi/factorem/blob/main/examples/stages.py
[license]: https://github.com/oierlauzi/factorem/blob/main/LICENSE
