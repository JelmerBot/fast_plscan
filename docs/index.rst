.. toctree::
   :caption: Features
   :maxdepth: 1
   :hidden:

   using_basic_api
   using_exploration_plots
   using_other_distances
   using_multiple_components
   using_sample_weights

.. toctree::
   :caption: API reference
   :maxdepth: 1
   :hidden:
   
   _autosummary_public/plscan
   _autosummary_plots/plscan.plots

.. toctree::
   :caption: Development
   :maxdepth: 1
   :hidden:
   
   local_development
   reference_lib

|PyPI version|

Persistent Leaves Spatial Clustering of Applications with Noise
===============================================================

This library provides a new clustering algorithm based on HDBSCAN. The primary
advantages of PLSCAN over the standard ``hdbscan`` library are:

 - PLSCAN automatically finds the optimal minimum cluster size.
 - PLSCAN can easily use all available cores to speed up computation;
 - PLSCAN has much faster implementations of tree condensing and cluster extraction;
 - PLSCAN does not rely on JIT compilation.

When using PLSCAN, only the `min_samples` parameter has to be given, which
specifies the number of neighbors used for mutual reachability distances. Higher
values produce smoother density profiles with fewer leaf clusters.

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from plscan import PLSCAN

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

.. figure:: _static/readme.png
   :alt: scatterplot

The algorithm builds a hierarchy of leaf-clusters, showing which clusters are
leaves as the minimum cluster size varies (filtration). Then, it computes the
total leaf-cluster persistence per minimum cluster size, and picks the minimum
cluster size that maximizes that score. The leaf-cluster hierarchy in
`leaf_tree_` can be plotted as an alternative to HDBSCAN\*'s condensed cluster
tree.

.. code:: python

    clusterer.leaf_tree_.plot(leaf_separation=0.1)
    plt.show()

.. figure:: _static/leaf_tree.png
   :alt: leaf cluster tree

Cluster segmentations for other high-persistence minimum cluster sizes can
be computed using the `cluster_layers` method. This method finds the
persistence peaks and returns their cluster labels and memberships.

.. code:: python

    layers = clusterer.cluster_layers(n_peaks=4)
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

.. figure:: _static/layers.png
   :alt: layers scatterplot


Citing
------

TODO

Licensing
---------

The ``plscan`` package has a 3-Clause BSD license.

.. |PyPI version| image:: https://badge.fury.io/py/plscan.svg
   :target: https://badge.fury.io/py/plscan

