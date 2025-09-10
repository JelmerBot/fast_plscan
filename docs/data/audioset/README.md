# Audioset

The Google Audioset dataset (https://research.google.com/audioset).

We create a sub-set of the dataset containing only music samples specifically
for evaluating clustering algorithms. To reproduce the dataset follow these
steps:

- Download the dataset and put the relevant files in `./sources`:
  - The list of training segments at
    http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
  - The actual data from one of these urls:
    - http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz
    - http://storage.googleapis.com/eu_audioset/youtube_corpus/v1/features/features.tar.gz
    - http://storage.googleapis.com/asia_audioset/youtube_corpus/v1/features/features.tar.gz
  - The ontology at
    https://raw.githubusercontent.com/audioset/ontology/refs/heads/master/ontology.json
- Run `genres.py` to extract a list of the most-specific music genres in the
  ontology. The script creates `generated/genres.csv` and
  `generated/unbalanced_music_segments.csv` which list the selected genres and
  training samples with exactly one tag matching the selected genres.
- Run `load_records.py` to create a feature matrix `generated/X_music.npy` and
  target vector `generated/y_music.npy` for the selected samples. This script
  requires tensorflow to read the dataset and can take a a while to complete.