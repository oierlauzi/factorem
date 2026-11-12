# Benchmarks

How factorem is scored, how it compares to
[RECOVAR](https://github.com/ma-gilles/recovar), and what was tried before
settling on the method it implements.

## Datasets

| | Particles | Box | Diameter given | Known about each particle |
|---|---|---|---|---|
| IgG-1D (CryoBench, SNR 0.01) | 100,000 | 128 px, 3.0 Å | 180 Å | Which of 100 states of a circular motion it was simulated in. |
| EMPIAR-10076, ribosome assembly | 131,899 | 320 px, 1.31 Å | 280 Å | The class Davis et al. (2016) put it in: 6 major, 14 minor. |
| EMPIAR-10180, spliceosome | 327,490 | 320 px, 1.70 Å | 400 Å | Whether cryoDRGN kept it after filtering its latent space. |

One image of EMPIAR-10180 (the 308,903rd) is corrupt. factorem reports it,
leaves it out, and gives it `nan` coordinates.

## Scores

All of them look at the 10 nearest neighbours of each particle in the
embedding, so they do not change with rotations or scalings of it and
assume no shape for it ([metrics.py](metrics.py)).

- **IgG-1D**, `knn_err`: mean circular distance between the state of a
  particle and those of its neighbours. 0 is perfect and 25 is chance.
- **EMPIAR-10076**, `major` and `minor`: how often the vote of the
  neighbours gives the published class of the particle, over all the
  particles (`acc`) and averaged over the classes (`bal_acc`).
- **EMPIAR-10180**, `kept`: the same, for whether cryoDRGN kept it.
- `r2_recovar`: how much of the variance of the RECOVAR coordinates the
  neighbours predict; whether both methods see the same.
- `nuis_defocus`, `nuis_view`: the same for the defocus and for the
  projection direction, which an embedding should know nothing about.
  Used in the sweep only.

## Results

factorem 0.1.0 at its default of 16 Å, with 6 (its default) and 10
components and nothing else but the diameter given, against RECOVAR 0.4.5
with 4 and 10 dimensions (and 6 on IgG-1D). One NVIDIA RTX PRO 4500 (32 GB)
and 72 cores.

| | | factorem, 6 | factorem, 10 | RECOVAR, 4 | RECOVAR, 6 | RECOVAR, 10 |
|---|---|---|---|---|---|---|
| IgG-1D | `knn_err`, lower is better | 1.34 | 1.10 | 1.89 | 1.28 | 0.85 |
| EMPIAR-10076 | `major_acc` | 0.818 | 0.820 | 0.838 | | 0.843 |
| | `major_bal_acc` | 0.637 | 0.641 | 0.614 | | 0.632 |
| | `minor_acc` | 0.472 | 0.485 | 0.452 | | 0.552 |
| | `minor_bal_acc` | 0.407 | 0.423 | 0.373 | | 0.486 |
| EMPIAR-10180 | `kept_acc` | 0.641 | 0.647 | 0.629 | | 0.658 |
| | `kept_bal_acc` | 0.617 | 0.623 | 0.607 | | 0.642 |

RECOVAR was given the EMPIAR particles Fourier cropped to 128 px, and
factorem the 320 px ones.

factorem is about where RECOVAR is with as many dimensions or a few fewer:
behind it on IgG-1D and on the minor classes of EMPIAR-10076 at 10
dimensions, level with it on the major classes and on EMPIAR-10180. The
neighbours in its embedding predict 96 %, 74 % and 57 % of the variance of
the 4 coordinates of RECOVAR on the three datasets (`r2_recovar4`).

| | Directions | Working images | Time | Peak memory |
|---|---|---|---|---|
| IgG-1D | 795 | 96 px, 4.0 Å | 55 s | 7 GB |
| EMPIAR-10076 | 1,924 | 106 px, 4.0 Å | 2.3 min | 11 GB |
| EMPIAR-10180 | 3,927 | 136 px, 4.0 Å | 5.3 min | 31 GB |

The times are with the files in the cache of the system. The first read of
EMPIAR-10076 (54 GB) made it 3.6 min, and that of EMPIAR-10180 (134 GB, on a
hard disk) 31 min. RECOVAR took 42 min on IgG-1D; its runs on the other two
were not timed.

The resolution is an input. IgG-1D gives a `knn_err` of 1.28 at 12 Å, 1.34
at 16 Å and 1.50 at 20 Å.

## Reproducing

```sh
pip install -e '..[dev]' scikit-learn
factorem particles.star embedded.star --diameter 180 -v
python score_star.py igg embedded.star
```

[datasets.py](datasets.py) says where the labels and the exported RECOVAR
coordinates of each dataset are, on the machine these were run on.
`export_recovar.py` writes the latter from a RECOVAR run, with the Python of
its environment, and `score_recovar.py` scores them the same way.

## What was tried

The method is what was left of a sweep of image metrics, dimensionality
reductions and synchronizations. Its harness is in the history of this
repository, at commit `ac987d4` (`benchmarks/lab.py`); the numbers below are
its own, and differ slightly from those of the package above. The rule was
one configuration for all the datasets, with no input that is not a
physical property of the particle.

### Image metric and dimensionality reduction

Each row embeds the particles of every direction of IgG-1D in a different
way, to 10 components, with everything else alike (common-line
synchronization, 6 components at the end).

| Embedding of a direction | Reduction | `knn_err` | `nuis_defocus` |
|---|---|---|---|
| **Probabilistic PCA with the CTF in the model** | | **1.41** | **-0.06** |
| PCA of Wiener corrected images | whitened coordinates | 1.68 | 0.29 |
| | coordinates scaled by the singular values | 1.86 | 0.29 |
| Distances between Wiener corrected images | MDS | 1.67 | 0.29 |
| | diffusion map, median width | 1.60 | 0.37 |
| | diffusion map, width of the excess | 1.44 | 0.41 |
| | diffusion map, local widths | 1.54 | 0.17 |
| Distances between images filtered by the CTF of the other (ManifoldEM) | MDS | 1.81 | 0.40 |
| | diffusion map, median width | 1.74 | 0.36 |
| | diffusion map, width of the excess | 1.61 | 0.35 |
| | diffusion map, local widths | 1.71 | 0.51 |
| The same, each coefficient over its noise variance | MDS | 1.45 | 0.33 |
| | diffusion map, median width | 1.39 | 0.36 |
| | diffusion map, width of the excess | 1.27 | 0.39 |
| | diffusion map, local widths | 1.37 | 0.36 |
| Mahalanobis distances (Bhamre, Zhao and Singer, 2017), radial prior | MDS | 1.44 | 0.61 |
| | diffusion map, median width | 1.36 | 0.64 |
| | diffusion map, width of the excess | 1.25 | 0.65 |
| | diffusion map, local widths | 1.32 | 0.66 |
| Distances between Wiener filtered images, radial prior | MDS | 3.61 | 0.19 |
| | diffusion map, median width | 3.19 | 0.23 |
| | diffusion map, width of the excess | 2.73 | 0.26 |
| | diffusion map, local widths | 2.93 | 0.26 |

The width of the Gaussian kernel of the diffusion maps is the median of the
squared distances, that median less their first percentile (the noise adds
about the same to all of them) and, for the local widths, the distance of
each image to its 7th neighbour.

Whatever is compared, the images come through different CTFs, and what is
left of that in their distances is the defocus: every one of these
embeddings knows it (`nuis_defocus` of 0.17 to 0.66), while the
probabilistic PCA, which never compares two images, does not. The best of
them are 0.15 states better than it on IgG-1D, but which is best depends on
how the kernel width is chosen, by more than that. That is the kind of
choice that ends up tuned for each dataset, so they were left out. They
were only run on IgG-1D.

### Synchronization

Six components per direction and at the end, probabilistic PCA. The three
columns are cones of 1, 1.5 and 2 Shannon angles (the first two with a
linear model of the mean over the view, the last with a quadratic one).

| | Synchronization | 1 | 1.5 | 2 |
|---|---|---|---|---|
| EMPIAR-10076, `major_acc` | common lines | 0.764 | 0.804 | 0.817 |
| | Burer-Monteiro | 0.750 | 0.777 | 0.721 |
| EMPIAR-10076, `nuis_view` | common lines | -0.03 | 0.00 | 0.02 |
| | Burer-Monteiro | 0.14 | 0.19 | 0.18 |
| EMPIAR-10180, `kept_acc` | common lines | 0.573 | 0.612 | 0.623 |
| | Burer-Monteiro | 0.557 | 0.582 | 0.590 |
| EMPIAR-10180, `nuis_view` | common lines | -0.08 | -0.06 | -0.04 |
| | Burer-Monteiro | 0.04 | 0.05 | 0.04 |

On IgG-1D, with the first of these configurations, common lines give a
`knn_err` of 1.93 with a `nuis_view` of 0.09, and Burer-Monteiro 4.66 with
0.65 (3.78 with 0.66 at its best, the last one).

The synchronization that factorem had, a Burer-Monteiro solver on the
coordinates of the particles that two directions share, leaves the view in
the embedding (`nuis_view`): neighbouring directions agree, but the frame
drifts over the sphere. Common lines relate any two directions, however
far apart, and gave the better scores on the three datasets.

### Everything else

On IgG-1D (`knn_err`), unless stated otherwise. The choice kept is in bold.

| | | |
|---|---|---|
| Components per direction, of which 6 are kept | 6, 8, **10**, 12, 16 | 1.93, 1.55, 1.41, 1.43, 1.46 |
| Partners of each direction in the synchronization (6 components) | 8, 32, 128, all | 1.96, 1.94, 1.93, 1.93 |
| Cone [Shannon angles], quadratic mean | 1, 1.5, **2** | 1.44, 1.32, 1.34 |
| ... EMPIAR-10076, `major_acc` | 1.5, **2** | 0.811, 0.821 |
| ... EMPIAR-10180, `kept_acc` | 1.5, **2** | 0.621, 0.633 |
| Spacing of the directions [Shannon angles], cone of 2 | **1**, 2 | 1.34, 1.51 |
| Model of the mean over the view, cone of 1 | constant, linear, **quadratic** | 1.54, 1.41, 1.44 |
| ... EMPIAR-10076, cone of 2, `major_acc` | linear, **quadratic** | 0.822, 0.821 |
| Mask [diameters] | none, 1, **1.25** | 1.67, 1.38, 1.41 |
| Iterations of expectation maximization | 15, **30**, 60 | 1.413, 1.409, 1.416 |
| A second pass: eigenimages refitted to the merged coordinates | **no**, yes | 1.41, 1.61 |
| ... EMPIAR-10076, `major_acc` | **no**, yes | 0.821, 0.320 |

The package gives each direction 64 partners, among values that score
alike. Linear and quadratic models of the mean score alike too, and the
quadratic one was kept for the wider cone. The mask of one diameter scores
a little better on IgG-1D, a simulation whose particles are centred
exactly; 1.25 leaves room for those that are not.
