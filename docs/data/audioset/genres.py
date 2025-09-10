import json
import pandas as pd


def traverse_down(node_id):
    node = id_map[node_id]
    children = node["child_ids"]
    if len(children) == 0:
        return [node]
    out = []
    for child_id in children:
        out.extend(traverse_down(child_id))
    return out


if __name__ == "__main__":
    # Extract music genres from the ontology
    with open("docs/audioset/sources/ontology.json") as f:
        ontology = json.load(f)

    name_map = {n["name"]: n for n in ontology}
    id_map = {n["id"]: n for n in ontology}

    main_genre_ids = name_map["Music genre"]["child_ids"]
    leaf_genres = []
    for genre_id in main_genre_ids:
        leaf_genres.extend(traverse_down(genre_id))
    genres = pd.DataFrame(
        dict(
            label=list(range(len(leaf_genres))),
            name=[g["name"] for g in leaf_genres],
            id=[g["id"] for g in leaf_genres],
        )
    )

    # Load training segments
    observations = pd.read_csv(
        "sources/unbalanced_train_segments.csv",
        comment="#",
        delimiter=", ",
        quotechar='"',
        engine="python",
        header=None,
    )
    observations.columns = ["video_id", "start", "end", "label_id"]

    # Extract observations with exactly 1 selected genre id
    genre_ids = genres.id.to_list()
    id_set = set(genre_ids)
    get_music_ids = lambda x: set(x.split(",")) & id_set
    observations["music_ids"] = observations.label_id.apply(get_music_ids)
    subset = observations[observations.music_ids.apply(len) == 1].copy()
    subset["label"] = subset.music_ids.apply(lambda x: genre_ids.index([*x][0]))
    subset.drop(columns=["label_id", "music_ids"], inplace=True)

    # Write the resulting files
    genres.to_csv("docs/data/audioset/generated/genres.csv", index=False)
    subset.to_csv(
        "docs/data/audioset/generated/unbalanced_music_segments.csv", index=False
    )
