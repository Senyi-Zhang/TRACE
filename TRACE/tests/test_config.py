from pathlib import Path

import pytest

from trace_fc.config import load_config


def test_relative_paths_are_resolved_from_config_file(tmp_path: Path):
    config_file = tmp_path / "configs" / "run.yaml"
    config_file.parent.mkdir()
    config_file.write_text(
        "data:\n  train_file: ../data/train.jsonl\n", encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.data.train_file == str((tmp_path / "data/train.jsonl").resolve())


def test_unknown_config_field_is_rejected(tmp_path: Path):
    config_file = tmp_path / "run.yaml"
    config_file.write_text("mystery: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown TraceConfig fields"):
        load_config(config_file)
