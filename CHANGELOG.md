# Changelog

## 0.1.0

The first release: a rewrite from scratch of what was in this repository,
after a sweep of image metrics, dimensionality reductions and
synchronizations.

- Each projection direction is embedded by probabilistic PCA with the CTF
  in its model, on images that are masked, whitened with the noise of
  their background and freed from the change of the mean with the view.
  It replaces the PCA of Wiener corrected images and the diffusion maps.
- `--embedding mahalanobis-mds` embeds each direction by the
  multidimensional scaling of the Mahalanobis distances between its
  images (Bhamre, Zhao and Singer, 2017) instead.
- Directions are brought to a common frame through the common lines of
  their eigenimages.
- `--synchronization burer-monteiro` relates the directions by the
  particles they share instead, with the relaxation of Burer and Monteiro,
  as before. Each coordinate now counts by the strength of its eigenimage.
- The command line is made of commands, of which `factorem analyze` is
  the first.
- The inputs are the diameter of the particle, the resolution and the
  number of components. The aperture and kernel options are gone.
- Images are read from their files as each projection direction needs
  them, and resampled for the resolution, so that few are in memory at a
  time. `--preload` copies them to memory instead, and `--scratch` to a
  file or a directory on a fast disk.
- Corrupt images are reported and left out. STAR files of RELION before
  3.1 are read.
- The package is typed and tested, and its stages are public: `Embedder`
  and `Synchronizer` are protocols to implement alternatives with.
- Only JAX, vitrio, NumPy, pandas and starfile are needed.
