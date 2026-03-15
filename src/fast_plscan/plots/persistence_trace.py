"""Public API for plotting and exporting condensed trees, leaf trees, and
persistence traces."""

import numpy as np
import matplotlib.pyplot as plt

from typing import Any

from .._api import PersistenceTrace as PersistenceTraceTuple


class PersistenceTrace(object):
    """
    A trace of total leaf cluster persistence across size thresholds.

    For each tested minimum cluster size value, this object stores the total
    persistence contribution of all leaf clusters alive at that threshold.
    It can be exported with ``to_numpy`` and ``to_pandas`` or visualized with
    ``plot``.
    """

    def __init__(self, trace: PersistenceTraceTuple):
        """
        Parameters
        ----------
        trace
            The total persistence trace as produced internally.
        """
        self._trace = trace

    def to_numpy(self) -> np.ndarray:
        """Returns a numpy array of the persistence trace.

        The total persistence is computed over the leaf clusters' left-open
        (birth, death] intervals. `min_size` contains all unique birth minimum
        cluster size thresholds. It should not be confused with the
        `minimum_cluster_size` threshold, as `min_size` refers to the last value
        before a cluster becomes a leaf.
        """
        dtype = [
            ("min_size", np.float32),
            ("persistence", np.float32),
        ]
        result = np.empty(self._trace.min_size.shape[0], dtype=dtype)
        result["min_size"] = self._trace.min_size
        result["persistence"] = self._trace.persistence
        return result

    def to_pandas(self):
        """Returns a pandas dataframe representation of the persistence trace.

        The total persistence is computed over the leaf clusters' left-open
        (birth, death] intervals. `min_size` contains all unique birth minimum
        cluster size thresholds. It should not be confused with the
        `minimum_cluster_size` threshold, as `min_size` refers to the last value
        before a cluster becomes a leaf.
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError(
                "You must have pandas installed to export pandas DataFrames"
            )

        return DataFrame(
            dict(min_size=self._trace.min_size, persistence=self._trace.persistence)
        )

    def plot(self, line_kws: dict[str, Any] | None = None):
        """
        Plots the total persistence trace.

        The x-axis shows the last minimum cluster size value before a cluster
        becomes a leaf! This matches the left-open (birth, death] interval used
        in the leaf tree and is needed to support weighted samples.

        Parameters
        ----------

        line_kws
            Additional keyword arguments for the line plot.
        """
        if line_kws is None:
            line_kws = dict()

        plt.plot(
            np.column_stack(
                (self._trace.min_size[:-1], self._trace.min_size[1:])
            ).reshape(-1),
            np.repeat(self._trace.persistence[:-1], 2),
            **line_kws,
        )
        plt.ylim([0, plt.ylim()[1]])
        plt.xlabel("Min cluster size in (birth, death]")
        plt.ylabel("Total persistence")
