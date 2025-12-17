[![PyPi version](https://badge.fury.io/py/fast-plscan.svg)](https://badge.fury.io/py/plscan)
![Conda version](https://anaconda.org/conda-forge/fast-plscan/badges/version.svg)
[![Repository DOI](https://zenodo.org/badge/xxx.svg)](https://zenodo.org/doi/xxx/zenodo.yyy)

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


## Local development

The development workflow works best by pre-installing python dependencies with
`pip` (or alternatives):

```bash
pip install numpy scipy matplotlib scikit-learn scikit-build-core nanobind setuptools_scm
```

Building the package requires cmake and a C++ 23 compiler with OpenMP support.
The OpenMP version must support user-defined reductions. Selecting the proper
OpenMP version requires some additional configuration, see below. Assuming the
compiler and OpenMP are present, the package can be compiled and installed with:

```bash
pip install --no-deps --no-build-isolation -ve .
```

To change the build type, add `-C cmake.build-type=Debug` or `-C
cmake.build-type=Release` to the command.

`scikit-build-core` also experimentally supports editable installs (see [their
documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs)):

```bash
pip install --no-deps --no-build-isolation -C editable.rebuild=true -ve .
```

### Linux

It may be necessary to tell cmake which compiler it should use. For example,
using `g++-14` when that is not the system default can be done by adding a `-C
cmake.args="-DCMAKE_CXX_COMPILER=g++-14"` option. The `-C cmake.args=...` option
does not have to be repeated on rebuilds.

### MacOS

MacOS requires installing OpenMP using homebrew:

```bash
brew install libomp
```

Also update the `~/.zshrc` config file with:

```bash
export OpenMP_ROOT=$(brew --prefix)/opt/libomp
```

or pass `OpenMP_ROOT` as cmake argument:

```bash
pip install --no-deps --no-build-isolation \
  -C cmake.args="-DOpenMP_ROOT=$(brew --prefix)/opt/libomp" \
  -ve .
```

### Windows

The default MSVC C++ compiler on windows does not support
a recent enough OpenMP. In addition, the default powershell terminal on windows
is not configured for cmake to find the correct OpenMP version. Instead, use a
developer powershell configured for a 64-bit target architecture. To open such a
terminal, run the following code in a normal Powershell terminal:

```powershell
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$iloc = & $vswhere -products * -latest -property installationpath
$devddl = "$iloc/Common7/Tools/Microsoft.VisualStudio.DevShell.dll"
Import-Module $devddl; Enter-VsDevShell -Arch amd64 -VsInstallPath $iloc -SkipAutomaticLocation
```

In addition, select the MSVC Clang compiler using `-C cmake.args="-T
ClangCL"` the first time the package is installed:

```powershell
pip install --no-deps --no-build-isolation -C cmake.args="-T ClangCL" -ve . 
```

The `-C cmake.args=...` option does not have to be repeated on rebuilds.

You may need to install the [visual studio build
tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
with the optional Clang compiler support enabled.

## Citing

When using this work, please cite our (upcoming) preprint:

```bibtex
@article{bot2025plscan,
  title         = {Persistent Multiscale Density-based Clustering},
  author        = {Dani{\"{e}}l M. Bot and Leland McInnes and Jan Aerts},
  year          = {2025},
  month         = {12},
  archiveprefix = {arXiv},
  eprint        = {TODO},
  primaryclass  = {cs.CL}
}
```

## Licensing

The ``fast-plscan`` package has a 3-Clause BSD license.