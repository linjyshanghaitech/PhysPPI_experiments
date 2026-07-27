from __future__ import annotations

import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    QBioLiP10kReader,
)


def _write_case(
    root: Path,
    sample_id: str,
    payload: object,
    chains: tuple[str, ...] = ("A",),
) -> Path:
    chunk = root / "chunk" / "chunk0001"
    prior_dir = chunk / "prior_pkl"
    prior_dir.mkdir(parents=True)
    (chunk / "gt").mkdir()
    (chunk / "bb_input").mkdir()
    (chunk / "gt" / f"{sample_id}.pdb").write_text("ATOM\n")
    for chain_id in chains:
        (chunk / "bb_input" / f"{sample_id}_{chain_id}.pdb").write_text("ATOM\n")
    path = prior_dir / f"{sample_id}.pkl"
    with path.open("wb") as handle:
        pickle.dump({sample_id: payload}, handle)
    return path


def test_reads_canonical_qbiolip_schema(tmp_path: Path) -> None:
    sample_id = "1abc_1"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(
            dmap={"A|B": [[12, 121, 8.2]]},
            interactions={
                "A|B": [
                    {
                        "r1": 12,
                        "r2": 121,
                        "t": "hbond",
                        "d": 2.6,
                        "a1": "NZ",
                        "a2": "O",
                    }
                ]
            },
            mesh={"A": [[0.0] * 10]},
            opm={
                "res_loc": {"A": {12: 0}},
                "outer_normal": [0.0, 0.0, 1.0],
                "outer_centroid": [0.0, 0.0, 0.0],
            },
        ),
    )

    reader = QBioLiP10kReader(tmp_path)
    sample = reader.read(sample_id)

    assert sample.dmap["A|B"][0][0:2] == (12, 121)
    assert sample.interactions[0].interaction_type == "hbond"
    assert sample.interactions[0].chain_pair == "A|B"
    assert sample.paths.monomer_pdbs == {
        "A": tmp_path / "chunk/chunk0001/bb_input/1abc_1_A.pdb"
    }
    assert sample.paths.main_msa_query_only is True


def test_reads_signed_pdb_resseq_charge_nan_and_opm_none(
    tmp_path: Path,
) -> None:
    sample_id = "signed_resseq"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(
            dmap={"A|B": [[0, -6, 7.25]]},
            interactions={
                "A|B": [
                    {
                        "r1": 0,
                        "r2": -6,
                        "t": "saltbridge",
                        "d": 3.1,
                        "a1": ["NZ", "NH1"],
                        "a2": ["OD1", "OE2"],
                    }
                ]
            },
            mesh={"A": [[1.0, 2.0, 3.0, math.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]},
            opm=None,
        ),
        chains=("A", "B"),
    )

    sample = QBioLiP10kReader(tmp_path).read(sample_id)

    assert sample.dmap == {"A|B": ((0, -6, 7.25),)}
    assert len(sample.interactions) == 1
    interaction = sample.interactions[0]
    assert interaction.chain_pair == "A|B"
    assert interaction.r1 == 0
    assert interaction.r2 == -6
    assert interaction.interaction_type == "saltbridge"
    assert interaction.distance == 3.1
    assert interaction.atom_1 == ("NZ", "NH1")
    assert interaction.atom_2 == ("OD1", "OE2")
    assert math.isnan(sample.mesh["A"][0][3])
    assert sample.opm == {}


def test_indexes_flat_main_and_nested_paired_msas(tmp_path: Path) -> None:
    sample_id = "case1"
    _write_case(tmp_path, sample_id, SimpleNamespace(dmap={}, mesh={}, opm={}))
    chunk = tmp_path / "chunk" / "chunk0001"
    main = chunk / "msa_local" / "main"
    paired = chunk / "msa_local" / "paired" / sample_id
    main.mkdir(parents=True)
    paired.mkdir(parents=True)
    main_path = main / f"{sample_id}_{sample_id}_chain=A.npz"
    paired_path = paired / f"{sample_id}_{sample_id}_chain=A.npz"
    main_path.write_bytes(b"main")
    paired_path.write_bytes(b"paired")

    paths = QBioLiP10kReader(tmp_path).locate(sample_id)

    assert paths.main_msa_npzs == {"A": main_path}
    assert paths.paired_msa_npzs == {"A": paired_path}
    assert paths.main_msa_query_only is False


@pytest.mark.parametrize(
    "basename",
    ("{sample}_{sample}_A.npz", "{sample}_{sample}_chain=A.npz"),
)
def test_accepts_exact_canonical_msa_basenames_in_each_layout(
    tmp_path: Path, basename: str
) -> None:
    sample_id = "patterns"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(dmap={}, mesh={}, opm={}),
    )
    chunk = tmp_path / "chunk" / "chunk0001"
    main = chunk / "msa_local" / "main"
    paired = chunk / "msa_local" / "paired" / sample_id
    main.mkdir(parents=True)
    paired.mkdir(parents=True)
    main_path = main / basename.format(sample=sample_id)
    paired_path = paired / basename.format(sample=sample_id)
    main_path.write_bytes(b"main")
    paired_path.write_bytes(b"paired")

    paths = QBioLiP10kReader(tmp_path).locate(sample_id)

    assert paths.main_msa_npzs == {"A": main_path}
    assert paths.paired_msa_npzs == {"A": paired_path}
    assert paths.main_msa_query_only_chains == ()


def test_indexes_nested_main_msa_and_ignores_other_sample_directories(
    tmp_path: Path,
) -> None:
    sample_id = "nested"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(dmap={}, mesh={}, opm={}),
    )
    main = tmp_path / "chunk" / "chunk0001" / "msa_local" / "main"
    nested_main = main / sample_id
    other_main = main / "other_sample"
    nested_main.mkdir(parents=True)
    other_main.mkdir()
    main_path = nested_main / f"{sample_id}_{sample_id}_A.npz"
    main_path.write_bytes(b"main")
    (other_main / "other_sample_other_sample_A.npz").write_bytes(b"foreign")

    paths = QBioLiP10kReader(tmp_path).locate(sample_id)

    assert paths.main_msa_npzs == {"A": main_path}
    assert paths.main_msa_query_only is False


def test_partial_main_msa_marks_only_missing_monomers_query_only(
    tmp_path: Path,
) -> None:
    sample_id = "partial"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(dmap={}, mesh={}, opm={}),
        chains=("A", "B"),
    )
    main = tmp_path / "chunk" / "chunk0001" / "msa_local" / "main"
    main.mkdir(parents=True)
    main_path = main / f"{sample_id}_{sample_id}_chain=A.npz"
    main_path.write_bytes(b"main")

    paths = QBioLiP10kReader(tmp_path).locate(sample_id)

    assert paths.main_msa_npzs == {"A": main_path}
    assert paths.main_msa_query_only_chains == ("B",)
    assert paths.main_msa_query_only is True


def test_indexes_only_visible_chunk_tree(tmp_path: Path) -> None:
    sample_id = "visible"
    _write_case(tmp_path, sample_id, SimpleNamespace(dmap={}, mesh={}, opm={}))

    hidden_chunk = tmp_path / ".chunk" / "chunk0002" / "prior_pkl"
    hidden_chunk.mkdir(parents=True)
    with (hidden_chunk / "hidden.pkl").open("wb") as handle:
        pickle.dump({"hidden": SimpleNamespace(dmap={}, mesh={}, opm={})}, handle)

    reader = QBioLiP10kReader(tmp_path)

    assert reader.sample_ids() == (sample_id,)


def test_rejects_legacy_cmap_or_attractive_schema(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "bad",
        SimpleNamespace(cmap={}, attractive={}, dmap={}, mesh={}, opm={}),
    )

    with pytest.raises(
        ValueError,
        match="legacy.*(?:cmap|attractive)|(?:cmap|attractive).*legacy",
    ):
        QBioLiP10kReader(tmp_path).read("bad")


def test_rejects_missing_or_malformed_canonical_fields(tmp_path: Path) -> None:
    _write_case(tmp_path, "missing", SimpleNamespace(mesh={}, opm={}))
    reader = QBioLiP10kReader(tmp_path)
    with pytest.raises(ValueError, match="dmap"):
        reader.read("missing")

    _write_case(
        tmp_path / "malformed",
        "bad",
        SimpleNamespace(dmap={"A|B": [[1, 2]]}, mesh={}, opm={}),
    )
    with pytest.raises(ValueError, match="dmap edge"):
        QBioLiP10kReader(tmp_path / "malformed").read("bad")


def test_rejects_unknown_fields_bad_chain_pairs_and_float_residue_ids(
    tmp_path: Path,
) -> None:
    _write_case(
        tmp_path,
        "unknown",
        SimpleNamespace(dmap={}, mesh={}, opm={}, old_cache={}),
    )
    with pytest.raises(ValueError, match="unknown.*old_cache"):
        QBioLiP10kReader(tmp_path).read("unknown")

    _write_case(
        tmp_path / "bad_pair",
        "bad_pair",
        SimpleNamespace(dmap={"A": []}, mesh={}, opm={}),
    )
    with pytest.raises(ValueError, match="chain-pair"):
        QBioLiP10kReader(tmp_path / "bad_pair").read("bad_pair")

    _write_case(
        tmp_path / "float_residue",
        "float_residue",
        SimpleNamespace(dmap={"A|B": [[1.5, 2, 8.2]]}, mesh={}, opm={}),
    )
    with pytest.raises(ValueError, match="integer PDB residue sequence number"):
        QBioLiP10kReader(tmp_path / "float_residue").read("float_residue")


def test_rejects_malformed_or_foreign_msa_files_and_unknown_msa_chain(
    tmp_path: Path,
) -> None:
    sample_id = "malformed_msa"
    _write_case(tmp_path, sample_id, SimpleNamespace(dmap={}, mesh={}, opm={}))
    main = tmp_path / "chunk" / "chunk0001" / "msa_local" / "main"
    main.mkdir(parents=True)
    (main / f"{sample_id}_wrong_name.npz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="malformed.*MSA"):
        QBioLiP10kReader(tmp_path)

    foreign_main_root = tmp_path / "foreign_main"
    _write_case(
        foreign_main_root,
        sample_id,
        SimpleNamespace(dmap={}, mesh={}, opm={}),
    )
    foreign_main = foreign_main_root / "chunk" / "chunk0001" / "msa_local" / "main"
    foreign_main.mkdir(parents=True)
    (foreign_main / "other_case_other_case_A.npz").write_bytes(b"bad")
    foreign_paths = QBioLiP10kReader(foreign_main_root).locate(sample_id)
    assert foreign_paths.main_msa_npzs == {}
    assert foreign_paths.main_msa_query_only is True

    foreign_root = tmp_path / "foreign"
    _write_case(
        foreign_root,
        sample_id,
        SimpleNamespace(dmap={}, mesh={}, opm={}),
    )
    paired = foreign_root / "chunk" / "chunk0001" / "msa_local" / "paired" / sample_id
    paired.mkdir(parents=True)
    (paired / "other_case_other_case_chain=A.npz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="malformed.*MSA"):
        QBioLiP10kReader(foreign_root)

    unknown_root = tmp_path / "unknown_chain"
    _write_case(
        unknown_root,
        sample_id,
        SimpleNamespace(dmap={}, mesh={}, opm={}),
    )
    unknown_main = unknown_root / "chunk" / "chunk0001" / "msa_local" / "main"
    unknown_main.mkdir(parents=True)
    (unknown_main / f"{sample_id}_{sample_id}_chain=B.npz").write_bytes(b"bad")
    unknown_paths = QBioLiP10kReader(unknown_root).locate(sample_id)
    assert unknown_paths.main_msa_npzs == {}
    assert unknown_paths.main_msa_query_only is True

    unknown_paired_root = tmp_path / "unknown_paired_chain"
    _write_case(
        unknown_paired_root,
        sample_id,
        SimpleNamespace(dmap={}, mesh={}, opm={}),
    )
    unknown_paired = (
        unknown_paired_root / "chunk" / "chunk0001" / "msa_local" / "paired" / sample_id
    )
    unknown_paired.mkdir(parents=True)
    (unknown_paired / f"{sample_id}_{sample_id}_chain=B.npz").write_bytes(b"bad")
    unknown_paired_paths = QBioLiP10kReader(unknown_paired_root).locate(sample_id)
    assert unknown_paired_paths.paired_msa_npzs == {}


def test_rejects_missing_mandatory_backbone_paths(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk" / "chunk0001"
    prior_dir = chunk / "prior_pkl"
    prior_dir.mkdir(parents=True)
    with (prior_dir / "case.pkl").open("wb") as handle:
        pickle.dump({"case": SimpleNamespace(dmap={}, mesh={}, opm={})}, handle)

    with pytest.raises(ValueError, match="mandatory.*case"):
        QBioLiP10kReader(tmp_path)


def test_preserves_interaction_atom_groups_as_tuples(tmp_path: Path) -> None:
    sample_id = "atom_groups"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(
            dmap={},
            interactions={
                "A|B": [
                    {
                        "r1": 1,
                        "r2": 2,
                        "t": "hbond",
                        "a1": ["NZ", "NH1"],
                        "a2": ["OD1", "OD2"],
                    }
                ]
            },
            mesh={},
            opm={},
        ),
        chains=("A", "B"),
    )

    record = QBioLiP10kReader(tmp_path).read(sample_id).interactions[0]

    assert record.atom_1 == ("NZ", "NH1")
    assert record.atom_2 == ("OD1", "OD2")


def test_rejects_non_charge_mesh_nan(tmp_path: Path) -> None:
    sample_id = "non_charge_nan"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(
            dmap={"A|B": [[1, 2, 8.2]]},
            interactions={},
            mesh={"A": [[math.nan, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]},
            opm={},
        ),
        chains=("A", "B"),
    )

    with pytest.raises(ValueError, match="mesh.*finite|nonfinite.*mesh"):
        QBioLiP10kReader(tmp_path).read(sample_id)


def test_rejects_mesh_rows_without_exact_canonical_ten_columns(
    tmp_path: Path,
) -> None:
    sample_id = "bad_mesh"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(dmap={}, interactions={}, mesh={"A": [[0.0] * 9]}, opm={}),
    )

    with pytest.raises(ValueError, match="mesh.*10|ten"):
        QBioLiP10kReader(tmp_path).read(sample_id)


def test_deduplicates_repeated_nested_main_msa_aliases_with_direct_precedence(
    tmp_path: Path,
) -> None:
    sample_id = "repeated_nested"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(dmap={}, interactions={}, mesh={}, opm={}),
    )
    nested = tmp_path / "chunk" / "chunk0001" / "msa_local" / "main" / sample_id
    repeated = nested / sample_id
    repeated.mkdir(parents=True)
    direct_path = nested / f"{sample_id}_{sample_id}_A.npz"
    repeated_path = repeated / f"{sample_id}_{sample_id}_A.npz"
    direct_path.write_bytes(b"direct")
    repeated_path.write_bytes(b"repeated")

    paths = QBioLiP10kReader(tmp_path).locate(sample_id)

    assert paths.main_msa_npzs == {"A": direct_path}


def test_deduplicates_repeated_nested_paired_msa_aliases_with_direct_precedence(
    tmp_path: Path,
) -> None:
    sample_id = "repeated_paired"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(dmap={}, interactions={}, mesh={}, opm={}),
    )
    paired = tmp_path / "chunk" / "chunk0001" / "msa_local" / "paired" / sample_id
    repeated = paired / sample_id
    repeated.mkdir(parents=True)
    direct_path = paired / f"{sample_id}_{sample_id}_A.npz"
    repeated_path = repeated / f"{sample_id}_{sample_id}_A.npz"
    direct_path.write_bytes(b"direct")
    repeated_path.write_bytes(b"repeated")

    paths = QBioLiP10kReader(tmp_path).locate(sample_id)

    assert paths.paired_msa_npzs == {"A": direct_path}


def test_deduplicates_paired_msa_chain_aliases_with_explicit_precedence(
    tmp_path: Path,
) -> None:
    sample_id = "alias_pair"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(dmap={}, interactions={}, mesh={}, opm={}),
        chains=("A", "B"),
    )
    paired = tmp_path / "chunk" / "chunk0001" / "msa_local" / "paired" / sample_id
    paired.mkdir(parents=True)
    shorthand = paired / f"{sample_id}_{sample_id}_A.npz"
    explicit = paired / f"{sample_id}_{sample_id}_chain=A.npz"
    b_path = paired / f"{sample_id}_{sample_id}_B.npz"
    shorthand.write_bytes(b"shorthand")
    explicit.write_bytes(b"explicit")
    b_path.write_bytes(b"b")

    paths = QBioLiP10kReader(tmp_path).locate(sample_id)

    assert paths.paired_msa_npzs == {"A": explicit, "B": b_path}


def test_deduplicates_main_msa_flat_and_nested_aliases_with_nested_precedence(
    tmp_path: Path,
) -> None:
    sample_id = "duplicate_main"
    _write_case(
        tmp_path,
        sample_id,
        SimpleNamespace(dmap={}, interactions={}, mesh={}, opm={}),
    )
    main = tmp_path / "chunk" / "chunk0001" / "msa_local" / "main"
    nested = main / sample_id
    nested.mkdir(parents=True)
    flat_path = main / f"{sample_id}_{sample_id}_A.npz"
    nested_path = nested / f"{sample_id}_{sample_id}_A.npz"
    flat_path.write_bytes(b"flat")
    nested_path.write_bytes(b"nested")

    paths = QBioLiP10kReader(tmp_path).locate(sample_id)

    assert paths.main_msa_npzs == {"A": nested_path}


def test_qbiolip_max_samples_zero_indexes_no_samples(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "bounded",
        SimpleNamespace(dmap={}, interactions={}, mesh={}, opm={}),
    )

    reader = QBioLiP10kReader(tmp_path, max_samples=0)

    assert reader.sample_ids() == ()


def test_qbiolip_rejects_negative_max_samples(tmp_path: Path) -> None:
    (tmp_path / "chunk").mkdir()

    with pytest.raises(ValueError, match="max_samples"):
        QBioLiP10kReader(tmp_path, max_samples=-1)
