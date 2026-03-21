import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors._kd_tree import KDTree32
from sklearn.neighbors._ball_tree import BallTree32

from fast_plscan import PLSCAN
from fast_plscan._helpers import distance_matrix_to_csr, knn_to_csr
from fast_plscan._api import set_num_threads, get_max_threads

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
    # See tests/generate_fixture.py for how this was generated.
    return np.load("tests/data/X.npy")


@pytest.fixture(scope="session")
def X_bool():
    # See tests/generate_fixture.py for how this was generated.
    return np.load("tests/data/X_bool.npy")


@pytest.fixture(scope="session")
def con_dists(X):
    return pdist(X).astype(np.float32)


@pytest.fixture(scope="session")
def dists(con_dists):
    return squareform(con_dists)


@pytest.fixture(scope="session")
def knn():
    # See generate_fixture.py for how this was generated.
    data = np.load("tests/data/fixture_knn.npz")
    return data["distances"], data["indices"]


@pytest.fixture(scope="session")
def knn_no_loops():
    # See tests/generate_fixture.py for how this was generated.
    data = np.load("tests/data/knn_no_loops.npz")
    return data["distances"], data["indices"]


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
