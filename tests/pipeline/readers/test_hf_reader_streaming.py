import json
from pathlib import Path

import pytest

from datatrove.pipeline.readers import HuggingFaceDatasetReader


pytest.importorskip("datasets")


@pytest.mark.parametrize("n_files", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("explicit_ids", [False, True])
@pytest.mark.parametrize("skip, limit", [(0, -1), (1, 2), (2, 1), (3, -1), (20, -1), (1, 0)])
def test_streaming_skip_limit_and_ids(
    tmp_path: Path, n_files: int, batch_size: int, explicit_ids: bool, skip: int, limit: int
) -> None:
    """Slice each streaming rank's nonempty documents without changing their IDs."""
    paths = []
    for file_index in range(n_files):
        path = tmp_path / f"part-{file_index}.jsonl"
        texts = ["", "a", "b", "", "c", "d", "", "e", "f"]
        with path.open("w", encoding="utf-8") as output:
            for text in texts:
                row = {"text": f"{file_index}-{text}" if text else ""}
                if explicit_ids:
                    row["id"] = f"id-{file_index}-{text}"
                output.write(json.dumps(row) + "\n")
        paths.append(str(path))

    # One input file is split by row (including empty rows); several files are split by file.
    expected_texts = (
        [["0-b", "0-c", "0-f"], ["0-a", "0-d", "0-e"]]
        if n_files == 1
        else [
            [f"{file_index}-{text}" for file_index in [0, 2] for text in "abcdef"],
            [f"1-{text}" for text in "abcdef"],
        ]
    )
    for rank in range(2):
        reader = HuggingFaceDatasetReader(
            "json",
            dataset_options={"data_files": paths, "split": "train"},
            streaming=True,
            batch_size=batch_size,
            skip=skip,
            limit=limit,
        )
        docs = list(reader(rank=rank, world_size=2))
        expected = [
            (text, f"id-{text}" if explicit_ids else f"json/{rank:05d}/{index}")
            for index, text in enumerate(expected_texts[rank])
        ][skip:]
        if limit != -1:
            expected = expected[:limit]
        assert [(doc.text, doc.id) for doc in docs] == expected
