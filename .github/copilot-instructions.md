# fast-plscan — Project Guidelines

fast-plscan implements a density-based clustering algorithm based on HDBSCAN*.

## Architecture

The library is split into a Python layer and a C++ core:

- `src/fast_plscan/_api/` — C++ extension compiled with **nanobind** and OpenMP (C++23).
- `src/fast_plscan/api.py` — Functional Python API of the main algorithm.
- `src/fast_plscan/sklearn.py` — Scikit-learn-compatible `PLSCAN` estimator (`BaseEstimator`, `ClusterMixin`).
- `src/fast_plscan/prediction.py` — Additional functional Python API for specialized prediction tasks.
- `src/fast_plscan/plots.py` — Visualization and conversion classes for the main PLSCAN data structures.
- `src/fast_plscan/_helpers.py` — Internal utilities for the Python layer.
- `src/fast_plscan/__init__.py` — Public API; all exported symbols listed in `__all__`.

Public entry-point: `from fast_plscan import PLSCAN` (sklearn interface) or the functional API functions.

### Algorithm overview

PLSCAN computes cluster labels through a fixed conversion pipeline:

1. Mutual-reachability MST/forest
2. Linkage tree
3. Condensed tree
4. Leaf tree
5. Persistence trace
6. Selected clusters and point labels/probabilities

A Python layer orchestrates these stages (`compute_mutual_spanning_tree` or `extract_mutual_spanning_forest`, then `clusters_from_spanning_forest`). A C++ layer performs the core conversions.

#### 1) MST/forest -> Linkage tree

- Input: sorted spanning-tree edges `(parent_point, child_point, distance)`.
- Output: linkage rows `(parent_node, child_node, child_count, child_size)` in merge order.
- Main structure: single forward sequential loop over MST edges with a disjoint-set/union-find state (`find` + path compression + `link`).
- Relation to indices:
	- Point indices are `0..num_points-1`.
	- Merge nodes are introduced as `num_points + row_index`.
	- Each linkage row corresponds to one merge event at the same edge order as MST distance order.

#### 2) Linkage tree -> Condensed tree

- Input: full linkage hierarchy plus MST distances and `min_cluster_size`.
- Output: condensed rows `(parent, child, distance, child_size)` and `cluster_rows`.
- Main structure: reverse sequential scan over linkage rows (bottom-up), with branch pruning and delayed row placement for pruned subtrees.
- Relation to indices:
	- Condensed parents use cluster labels starting at `num_points`.
	- `num_points` acts as a phantom root label for forest support.
	- `child < num_points` means a point row.
	- `child >= num_points` means a cluster-segment row.
	- `cluster_rows` stores row indices of cluster-segment rows (used by later stages).

#### 3) Condensed tree -> Leaf tree

- Input: condensed tree rows and `cluster_rows`.
- Output: per-segment arrays: `parent`, `[min_distance, max_distance]`, `[min_size, max_size]`.
- Main structure: a few linear passes (no recursive traversal):
	- Pass over all condensed rows to fill segment birth/min distance.
	- Pass over `cluster_rows` to fill parent and max distance.
	- Reverse paired pass over `cluster_rows` to propagate size intervals.
- Relation to indices:
	- Leaf segment index is the condensed cluster label offset: `segment_idx = condensed_label - num_points`.
	- Therefore `condensed_tree.parent[row] - num_points` and `condensed_tree.child[row] - num_points` index leaf-tree segments for cluster rows.
	- Segment `0` is reserved for the phantom root.

#### 4) Leaf tree -> Persistence trace

- Input: leaf segment size/distance lifetimes (and condensed point rows for distance/density variants).
- Output: trace arrays `min_size[]`, `persistence[]`.
- Main structure:
	- Build candidate size thresholds from leaf intervals, sort + unique.
	- Accumulate each segment's persistence over its active size interval via interval fill on the trace.
	- For distance/density and bi-persistence, use a reverse scan of condensed point rows and walk ancestors via `leaf_tree.parent` to collect contributions.
- Result semantics:
	- Standard traces sum one persistence value per active segment.
	- Bi-persistence traces integrate distance/density persistence contributions over size.

#### 5) Persistence trace -> selected clusters -> point labelling

- Input: persistence trace + leaf/condensed trees.
- Output: selected segment ids, per-point labels, and probabilities.
- Main structure:
	- Select the best cut size by maximizing total trace persistence (bounded by `max_cluster_size`).
	- Compute selected segments for that cut with a linear pass over leaf segments.
	- Build segment labels in one forward pass over leaf segments (inherit parent label unless selected).
	- Label points in one pass over condensed point rows (`child < num_points`).
- Relation to indices:
	- Selected cluster ids are leaf segment indices (`uint32`) compatible with leaf-tree indexing.
	- Point-row parent mapping uses `parent_segment = condensed_tree.parent[row] - num_points`.
	- Probability is normalized within selected segment persistence (distance to segment death / segment persistence span).

### Additional functionality

PLSCAN also exposes additional functionality beyond the core clustering pipeline:

#### Cluster layers (`PLSCAN.cluster_layers`)

- Purpose: return multiple valid clusterings by cutting at persistence-trace peaks instead of only the global best peak used by `fit()`.
- Input: fitted trace arrays `(min_size, persistence)` and optional peak filters (`min_size`, `max_size`, `height`, `threshold`, `max_peaks`).
- Output: list of `(peak_min_cluster_size, labels, probabilities)` tuples.
- Main structure:
	- Build a padded 1D signal and run `scipy.signal.find_peaks` (single pass over the sampled trace + peak extraction).
	- Filter peak indices by size range and optionally keep only top-`k` by persistence using `np.partition`.
	- For each surviving peak, compute labels/probabilities at that `cut_size`.
- Relation to existing structures:
	- `cut_size` is a sampled `trace.min_size` value.
	- Returned cluster ids are dense labels (`0..k-1`) for each layer, while selected regions correspond to leaf-tree segments.

#### Cluster centroids (`PLSCAN.compute_centroids`)

- Purpose: return one representative center per cluster as a probability-weighted feature mean.
- Input: fitted feature vectors and cluster labels (default: fitted labels).
- Output: centroid matrix `(n_clusters, n_features)`.
- Main structure:
	- Aggregate points by cluster label with membership probabilities as weights.
	- Compute weighted means with vectorized linear algebra.
- Relation to structures:
	- Uses final labels for membership and fitted probabilities for weighting.
	- Available for feature-input models.

#### Cluster medoids (`PLSCAN.compute_medoid_indices`)

- Purpose: return one representative point index per cluster minimizing weighted within-cluster dissimilarity.
- Input: fitted data representation and cluster labels (default: fitted labels).
- Output: index array `(n_clusters,)` into the original fitted samples.
- Main structure:
	- Evaluate per-cluster candidate points by weighted total distance to other members.
	- Select the minimum-cost member per cluster.
	- Use feature-space distances for feature input and graph distances for precomputed graph input.
- Relation to structures:
	- Labels define cluster membership and probabilities provide member weights.
	- Returned indices use the original fitted-point indexing.

#### Cluster exemplars (`PLSCAN.compute_exemplar_indices`)

- Purpose: return dense-core exemplar point sets for each cluster.
- Input: fitted leaf/condensed trees and cluster labels (default: fitted labels).
- Output: list of exemplar index arrays, one array per cluster.
- Main structure:
	- Map points to leaf segments from condensed point rows.
	- Keep points that hit the minimum distance level of their leaf segment.
	- Group kept points by cluster label.
- Relation to structures:
	- Uses condensed-tree point rows (`child < num_points`) and leaf-tree minimum distances.
	- Output indices are in the same coordinate system as fitted sample indices.

#### Soft membership vectors (`fast_plscan.prediction.all_points_membership_vectors`)

- Purpose: compute soft membership vectors for the fitted points.
- Input: fitted PLSCAN object, optional alternative labels.
- Output: membership matrix `(n_points, n_clusters)` where row sums equal probability of belonging to some cluster.
- Main structure:
	- Determine exemplar sets per cluster.
	- Compute distance weights: dense pairwise distance path (feature input) or sparse graph extraction path (precomputed graph input).
	- Compute topology weights by mapping each point to a leaf segment from condensed point rows, then computing merge distances to each selected cluster lineage.
	- Blend distance and topology weights multiplicatively, row-normalize, then scale by per-point `probability_in_some_cluster`.
- Relation to structures:
	- Point-to-segment map uses condensed point rows: `child_to_leaf_tree[point] = parent - num_points`.
	- Selected clusters are leaf segment ids; if custom labels are passed, selected segments are re-derived from label-to-segment membership.

#### Cross-cluster membership vectors for unseen points (`fast_plscan.prediction.membership_vectors`)

- Purpose: approximate soft membership vectors for unseen feature vectors.
- Input: unseen feature vectors `X` plus a fitted feature-space `clusterer`.
- Output: approximate membership matrix `(n_new_points, n_clusters)`.
- Main structure:
	- Query fitted spatial tree for nearest neighbors; compute approximate new-point core distances.
	- Use nearest mutual-reachability neighbor indices to anchor each new point into fitted topology.
	- Compute distance weights to fitted exemplars.
	- Reuse topology-weight computation with the neighbor anchors and new core distances.
	- Blend, normalize, and scale using the same distance/topology weighting scheme as fitted-point memberships.
- Relation to structures:
	- New points do not create new tree nodes; they inherit topology via nearest fitted neighbors.
	- Selected clusters remain fitted leaf segment ids; output columns align with fitted cluster label order.

#### Cluster labelling of unseen points (`fast_plscan.prediction.approximate_predict`)

- Purpose: assign hard labels and probabilities to unseen feature vectors.
- Input: unseen feature vectors `X` plus a fitted PLSCAN `clusterer`.
- Output: hard labels `(n_new_points,)` and probabilities `(n_new_points,)`.
- Main structure:
	- Query nearest mutual-reachability neighbor for each new point.
	- Assign hard label directly from the neighbor's fitted label.
	- For assigned points, compute probability from selected-cluster distance span:
		- cluster span = `max_distance - min_distance` of the selected leaf segment,
		- point persistence proxy = `max_distance - new_point_core_distance`,
		- probability = clipped ratio.
- Relation to structures:
	- Label transfer uses fitted point labels indexed by nearest-neighbor indices.
	- Probability uses selected leaf segment ids (`selected_clusters_[label]`) to read leaf-tree min/max distances.
	- Points inheriting noise neighbor labels remain `-1` with probability `0`.

### Build and Test

**Install for development** (requires CMake ≥ 3.18, a C++23 compiler, and the clang-cl toolchain on Windows):
```sh
pip install nanobind scikit-build-core[pyproject]
pip install --no-build-isolation -ve .
```
Add `--config-settings editable.rebuild=true` to auto-rebuild C++ on import.

**Run tests:**
```sh
pytest .
```
Test dependencies: `pytest`, `networkx`, `pandas`.

**Windows notes:** The build uses `-T ClangCL`; wheels are repaired with `delvewheel`.

### Key Dependencies

| Package | Version |
|---------|---------|
| numpy | ≥2, <3 |
| scipy | ≥1, <2 |
| scikit-learn | ≥1.6, <2 |
| matplotlib | ≥3, <4 |

Python 3.10–3.14 supported (ABI3 stable wheel tagged `cp312-abi3`).

## Agent Instructions

You are a senior developer on the fast-plscan project. Judge all proposed code changes against the conventions in this file and the more detailed instructions in the linked instruction files (see below). When analyzing or creating a change, critique it for any convention violations. Give pushback on proposed changes that violate conventions, introduce bugs, or are simply a bad idea for maintainability, clarity, or user experience reasons.

### Cross-cutting conventions

- Prefer extending existing flows over adding near-duplicate paths. Extract small helpers for shared logic and keep entry points focused on orchestration.
- Keep concerns separated: validation and normalization at boundaries, compute in dedicated kernels/helpers, and formatting/plotting in presentation layers.
- Preserve user-facing Python API behavior by default. Internal C++ interfaces may evolve as needed with coordinated Python updates.
- Keep changes cohesive across code, docs, and tests. Behavior changes should be reflected in docstrings/docs and covered by tests in the same change.
- Prefer describing durable patterns and invariants in instructions, not transient implementation details (exact helper names, temporary data layouts, or current fixture wiring).

### Where to find detailed rules

- **C++ extension** (`src/fast_plscan/_api/`): see [instructions/cpp-api.instructions.md](instructions/cpp-api.instructions.md)
- **Python API** (`src/fast_plscan/*.py`): see [instructions/python-api.instructions.md](instructions/python-api.instructions.md)
- **Documentation** (`docs/`, docstrings, API reference): see [instructions/documentation.instructions.md](instructions/documentation.instructions.md)
- **Tests** (`tests/`): see [instructions/tests.instructions.md](instructions/tests.instructions.md)

### Reply formats

Use the following formats for different types of requests:

#### Simple questions

If the user asks a simple question about the project, reply with a direct answer. Use the instructions to figure out where in the project to look for relevant information.

#### Small code change requests (default)

If the user requests a small code change (e.g., "move these lines to a helper function"), review the request against the instructions and directly make the change if you judge it to be a good idea. If the change violates conventions or is a bad idea, push back with an explanation and suggest an alternative when appropriate. If the requested changes affects the public API, behavior, documentation, or the information in these instructions, treat it as a larger design change and follow the process outlined below.

#### Larger design or implementation requests

If the user requests a larger design or implementation change (e.g., "add a new feature"), follow this multi-stage process. The user may send follow-up messages, treat these as simple questions, small code change requests, or refinements of the active larger design process. Don't start a new larger design process when one is already active unless explicitly asked to.

##### 1. API surface design

Make a concise proposal for the public API surface changes needed to support the feature. This includes new parameters, new public functions, new fitted attributes, and any new types returned to the user. Do not write implementation code yet.

---

**Stop here. Present the API surface concisely. Ask for corrections and explicit approval before proceeding.**

---

##### 2. Implement the test suite

Implement a full set of tests needed to verify the feature. Re-use existing fixtures and validation helpers where possible. Add new ones only if needed. Place test functions in the most appropriate existing test file, or create a new one if the feature is large and self-contained enough.

---

**Stop here. Let the user review the implemented tests. Continue on approval, or refine when requested. The user may make changes to the tests, these should be considered as the new approved API where possible.**

--- 

##### 3. Implement the feature

Implement the feature to make the tests pass. Follow the instructions for the relevant code areas (C++ extension, Python API, etc.) and keep the implementation aligned with the approved API design.

---

**Stop here. Let the user review the implementation. Continue on approval, or refine when requested. The user may make changes to the implementation and tests. Check all tests pass before continuing.**

---

##### 4. Update documentation

Update docstrings, docs pages, and agent-instructions to reflect the new feature. Follow the documentation conventions and ensure that all new behavior is clearly documented with examples where appropriate.

---

**Stop here. Let the user review documentation updates. Continue on approval, or refine when requested. Check documentation builds successfully before continuing.**

---

##### Final check

Critically review all changes made since the original feature request. Check for any convention violations, potential bugs, or maintainability issues. If everything looks good, give a ready-to-copy commit command (including changes made by the user while working on this feature) with a message summarizing the complete feature addition (e.g., "Add weighted core distances feature with API, tests, and docs").