# Changelog

## 0.2.0

A rewrite from scratch, around the method that a sweep of image metrics,
dimensionality reductions and synchronizations settled on (see
[benchmarks](benchmarks/README.md)).

- Each projection direction is embedded by probabilistic PCA with the CTF
  in its model, on images that are masked, whitened with the noise of
  their background and freed from the change of the mean with the view.
  It replaces the PCA of Wiener corrected images and the diffusion maps.
- Directions are brought to a common frame through the common lines of
  their eigenimages, with a generalized power method. It replaces the
  Burer-Monteiro synchronization of the particles that directions share.
- The inputs are the diameter of the particle, the resolution and the
  number of components. The aperture, kernel and synchronization options
  are gone.
- Images are read once and resampled for the resolution. Corrupt images
  are reported and left out. STAR files of RELION before 3.1 are read.
- The package is typed and tested, and its stages are public: `Embedder`
  and `Synchronizer` are protocols to implement alternatives with.
- Only JAX, vitrio, NumPy, pandas and starfile are needed.
