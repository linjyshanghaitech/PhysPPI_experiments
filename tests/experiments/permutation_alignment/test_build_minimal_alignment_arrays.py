import json
from pathlib import Path

import pytest

from experiments.permutation_alignment.build_minimal_alignment_arrays import (
    copy_colabfold_main_npz,
    load_rep_sequences,
    write_alignment_representatives_fasta,
)


def test_load_rep_sequences_sorts_unsorted_mapping(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "mappings"
    mapping_dir.mkdir()
    (mapping_dir / "rep_id_to_seq.json").write_text(
        json.dumps({"rep-b": "BBBB", "rep-a": "AAAA"})
    )

    rep_to_sequence = load_rep_sequences(mapping_dir)

    assert list(rep_to_sequence.items()) == [
        ("rep-a", "AAAA"),
        ("rep-b", "BBBB"),
    ]


def test_write_alignment_representatives_fasta_is_sorted(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "representatives.fasta"

    write_alignment_representatives_fasta(
        {"rep-b": "BBBB", "rep-a": "AAAA"},
        output_path,
    )

    assert output_path.read_text() == ">rep-a\nAAAA\n>rep-b\nBBBB\n"


def test_copy_colabfold_main_npz_preserves_bytes_and_sorts_keys(
    tmp_path: Path,
) -> None:
    main_msa_dir = tmp_path / "main"
    main_msa_dir.mkdir()
    source_bytes = {
        "rep-a": b"\x00npz-a\xff",
        "rep-b": b"\x01npz-b\xfe",
    }
    for rep_id, content in source_bytes.items():
        (main_msa_dir / f"{rep_id}.npz").write_bytes(content)
    output_dir = tmp_path / "alignment_arrays"

    copied = copy_colabfold_main_npz(
        {"rep-b": "BBBB", "rep-a": "AAAA"},
        main_msa_dir,
        output_dir,
    )

    assert list(copied) == ["rep-a", "rep-b"]
    assert copied == {
        "rep-a": output_dir / "rep-a.npz",
        "rep-b": output_dir / "rep-b.npz",
    }
    assert {
        rep_id: path.read_bytes() for rep_id, path in copied.items()
    } == source_bytes


def test_load_rep_sequences_rejects_empty_mapping(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "mappings"
    mapping_dir.mkdir()
    (mapping_dir / "rep_id_to_seq.json").write_text("{}")

    with pytest.raises(ValueError, match="empty ColabFold rep sequence map"):
        load_rep_sequences(mapping_dir)


def test_load_rep_sequences_rejects_missing_mapping(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "missing-mappings"

    with pytest.raises(FileNotFoundError, match="rep_id_to_seq.json"):
        load_rep_sequences(mapping_dir)


def test_copy_colabfold_main_npz_reports_missing_source_path(
    tmp_path: Path,
) -> None:
    main_msa_dir = tmp_path / "main"
    main_msa_dir.mkdir()
    (main_msa_dir / "rep-a.npz").write_bytes(b"npz-a")
    missing_path = main_msa_dir / "rep-b.npz"

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        copy_colabfold_main_npz(
            {"rep-a": "AAAA", "rep-b": "BBBB"},
            main_msa_dir,
            tmp_path / "alignment_arrays",
        )
