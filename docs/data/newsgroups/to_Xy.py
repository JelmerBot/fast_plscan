import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    X, y = fetch_20newsgroups(return_X_y=True)
    X = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=5000, stop_words="english"
    ).fit_transform(X)
    np.save("docs/data/newsgroups/generated/X_20newsgroups.npy", X.toarray())
    np.save("docs/data/newsgroups/generated/y_20newsgroups.npy", y)
