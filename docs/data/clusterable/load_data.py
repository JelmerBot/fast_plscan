import requests
import numpy as np
from io import BytesIO
from hdbscan import HDBSCAN

data = np.load(
    BytesIO(
        requests.get(
            "https://github.com/scikit-learn-contrib/hdbscan/blob/master/notebooks/clusterable_data.npy?raw=true"
        ).content
    )
)
mst = (
    HDBSCAN(min_samples=5, gen_min_span_tree=True)
    .fit(data)
    .minimum_spanning_tree_.to_numpy()
)
np.save("docs/data/clusterable/sources/clusterable_data.npy", data)
np.save("docs/data/clusterable/generated/clusterable_mst.npy", mst)
