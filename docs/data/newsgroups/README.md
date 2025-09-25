# 20 News groups dataset

The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup
documents, partitioned (nearly) evenly across 20 different newsgroups. See the
description and data at http://qwone.com/~jason/20Newsgroups/.

Run these scripts to reproduce the data files used in the documentation notebooks:
- `load_data.py` to convert text into a numeric feature matrix
  `./generated/X_20newsgroups.npy` and target vector
  `./generated/y_20newsgroups.npy` using the sklearn `TfidfVectorizer`.

  