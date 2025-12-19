[![PyPi version](https://badge.fury.io/py/fast-plscan.svg)](https://badge.fury.io/py/fast-plscan)
![Conda version](https://anaconda.org/conda-forge/fast-plscan/badges/version.svg)
[![DOI](https://zenodo.org/badge/1022168364.svg)](https://doi.org/10.5281/zenodo.17964285)


# Persistent Leaves Spatial Clustering for Applications with Noise

This library provides a new clustering algorithm based on HDBSCAN*. The primary
advantages of PLSCAN over the
[``hdbscan``](https://github.com/scikit-learn-contrib/hdbscan) and
[``fast_hdbscan``](https://github.com/TutteInstitute/fast_hdbscan) libraries are:

 - PLSCAN automatically finds the optimal minimum cluster size.
 - PLSCAN can easily use all available cores to speed up computation.
 - PLSCAN has much faster implementations of tree condensing and cluster extraction.
 - PLSCAN does not rely on JIT compilation.

To use PLSCAN, you only need to set the ``min_samples`` parameter. This
parameter controls how many neighbors are considered when measuring distances
between points. Setting a higher value for ``min_samples`` makes the algorithm
group points into larger, smoother clusters, and usually results in fewer, more
stable clusters.

```python
import numpy as np
import matplotlib.pyplot as plt

from fast_plscan import PLSCAN

data = np.load("docs/data/data.npy")

clusterer = PLSCAN(
  min_samples = 5, # same as in HDBSCAN
).fit(data)

plt.figure()
plt.scatter(
  *data.T, c=clusterer.labels_ % 10, s=5, alpha=0.5, 
  edgecolor="none", cmap="tab10", vmin=0, vmax=9
)
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
```

![scatterplot](./docs/_static/readme.png)

The algorithm creates a hierarchy of leaf-clusters by changing the minimum
cluster size. As this parameter varies, clusters appear or disappear. For each
minimum cluster size, the algorithm measures how long these leaf-clusters
persist. It then selects the minimum cluster size where the total persistence is
highest, giving the most stable clustering. You can visualize this hierarchy
using the ``leaf_tree_`` attribute, which provides an alternative to HDBSCAN*'s
condensed cluster tree.

```python
clusterer.leaf_tree_.plot(leaf_separation=0.1)
plt.show()
```

![leaf tree](./docs/_static/leaf_tree.png)

You can also explore how the clustering changes for other important values of
the minimum cluster size. The ``cluster_layers`` method automatically finds the
most persistent clusterings and returns their cluster labels and membership
strengths.

```python
layers = clusterer.cluster_layers(max_peaks=4)
for i, (size, labels, probs) in enumerate(layers):
  plt.subplot(2, 2, i + 1)
  plt.scatter(
    *data.T,
    c=labels % 10,
    alpha=np.maximum(0.1, probs),
    s=1,
    linewidth=0,
    cmap="tab10",
  )
  plt.title(f"min_cluster_size={int(size)}")
  plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
```

![layers](./docs/_static/layers.png)

## Installation instructions

Pre-build binaries are available on [pypi](https://badge.fury.io/py/fast-plscan), so the package can be installed with `pip` and similar package managers on most systems:

```bash
pip install fast_plscan
```

Conda forge builds are in progress. See [our
documentation](https://fast-plscan.readthedocs.io/en/latest/local_development.html)
for instructions on compiling the package locally.

## Citing

When using this work, please cite [our preprint](https://arxiv.org/abs/2512.16558):

```bibtex
@misc{bot2025plscan,
  title         = {Persistent Multiscale Density-based Clustering},
  author        = {Dani{\"{e}}l Bot and Leland McInnes and Jan Aerts},
  year          = {2025},
  eprint        = {2512.16558},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2512.16558}
}
```

## Licensing

The ``fast-plscan`` package has a 3-Clause BSD license.