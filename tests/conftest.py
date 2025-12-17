import pytest
import numpy as np
from pathlib import Path
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors._kd_tree import KDTree32
from sklearn.neighbors._ball_tree import BallTree32

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

from fast_plscan import PLSCAN
from fast_plscan._helpers import distance_matrix_to_csr, knn_to_csr
from fast_plscan._api import set_num_threads, get_max_threads

# Ensure random data does not change on different OSes
datapath = Path(__file__).parent / "data"

# used to select which input the algorithm should use (X or X_bool)
boolean_metrics = {
    "hamming",
    "dice",
    "jaccard",
    "russellrao",
    "rogerstanimoto",
    "sokalsneath",
}
# used to avoid duplicate tests where possible
duplicate_metrics = {"p", "infinity", "manhattan", "l1", "l2"}
# used to select which input the algorithm should use (X or X_bool)
numerical_balltree_metrics = set(PLSCAN.VALID_BALLTREE_METRICS) - boolean_metrics


def pytest_sessionstart(session):
    set_num_threads(1)


def pytest_sessionfinish(session, exitstatus):
    set_num_threads(get_max_threads())


@pytest.fixture(scope="session")
def X():
    X, y = make_blobs(n_samples=200, random_state=10)
    X, y = shuffle(X, y, random_state=7)
    return StandardScaler().fit_transform(X).astype(np.float32)


@pytest.fixture(scope="session")
def X_bool():
    p = 0.25
    rng = np.random.Generator(np.random.PCG64(10))
    return rng.choice(a=[True, False], size=(200, 100), p=[p, 1 - p]).astype(np.float32)


@pytest.fixture(scope="session")
def con_dists(X):
    return pdist(X).astype(np.float32)


@pytest.fixture(scope="session")
def dists(con_dists):
    return squareform(con_dists)


@pytest.fixture(scope="session")
def knn(X):
    knn = NearestNeighbors(n_neighbors=8).fit(X).kneighbors(X, return_distance=True)
    knn[0][0:5, -1] = np.inf
    knn[1][0:5, -1] = -1
    return knn


@pytest.fixture(scope="session")
def knn_no_loops(X):
    knn = NearestNeighbors(n_neighbors=8).fit(X).kneighbors()
    knn[0][0:5, -1] = np.inf
    knn[1][0:5, -1] = -1
    return knn


@pytest.fixture(scope="session")
def g_knn(knn):
    return knn_to_csr(*knn)


@pytest.fixture(scope="session")
def g_dists(dists):
    return distance_matrix_to_csr(dists)


@pytest.fixture(scope="session")
def mst(g_dists):
    mst = sp.csgraph.minimum_spanning_tree(g_dists, overwrite=True).tocoo()
    out = np.empty((mst.row.size, 3), dtype=np.float64)
    order = np.argsort(mst.data)
    out[:, 0] = mst.row[order]
    out[:, 1] = mst.col[order]
    out[:, 2] = mst.data[order]
    return out


@pytest.fixture(scope="session")
def kdtree(X):
    return KDTree32(X)


@pytest.fixture(scope="session")
def balltree(X):
    return BallTree32(X)
