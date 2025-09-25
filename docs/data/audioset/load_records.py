import tarfile
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


def extract_tar_gz(tar_gz_path, extract_path):
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=extract_path)


def load_tfrecords_from_tar(in_file, tmpdir):
    extract_tar_gz(in_file, tmpdir)
    tfrecord_dir = Path(tmpdir) / "audioset_v1_embeddings" / "unbal_train"
    tfrecord_files = [str(f) for f in tfrecord_dir.iterdir() if f.suffix == ".tfrecord"]
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    return raw_dataset


def decode_fn(record_bytes):
    context, sequence, _ = tf.io.parse_sequence_example(
        record_bytes,
        context_features={
            "video_id": tf.io.FixedLenFeature([], dtype=tf.string),
            "start_time_seconds": tf.io.FixedLenFeature([], dtype=tf.float32),
            "end_time_seconds": tf.io.FixedLenFeature([], dtype=tf.float32),
            "labels": tf.io.VarLenFeature(dtype=tf.int64),
        },
        sequence_features={
            "audio_embedding": tf.io.VarLenFeature(dtype=tf.string),
        },
    )
    context["audio_embedding"] = tf.map_fn(
        lambda x: tf.io.decode_raw(x, tf.uint8),
        sequence["audio_embedding"].values,
        fn_output_signature=tf.TensorSpec([128], tf.uint8),
    )
    return context


if __name__ == "__main__":
    features = dict()
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_file = "docs/data/audioset/sources/features.tar.gz"
        train_set_file = "docs/data/audioset/generated/unbalanced_music_segments.csv"

        train_set = pd.read_csv(train_set_file)
        dataset = load_tfrecords_from_tar(sources_file, tmpdir)

        for batch in dataset.map(decode_fn):
            video_id = batch["video_id"].numpy().decode("utf-8")
            embedding = batch["audio_embedding"].numpy()
            features[video_id] = embedding

    X = train_set.video_id.apply(lambda x: features[x].reshape(-1)).to_numpy()
    y = train_set.label.to_numpy()
    id = train_set.video_id.to_numpy()
    mask = [x.shape[0] == 1280 for x in X]

    np.save("docs/data/audioset/generated/X.npy", np.stack(X[mask]))
    np.save("docs/data/audioset/generated/y.npy", y[mask])
    np.save("docs/data/audioset/generated/video_id.npy", id[mask])
