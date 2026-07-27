from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import AFDB10kReader


def _atom_line(serial: int, residue: str, chain: str, res_id: int) -> str:
    x = float(serial)
    return (
        f"ATOM  {serial:5d}  CA  {residue:>3s} {chain:1s}{res_id:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}           C\n"
    )


def _pdb_text(chain_sequences: dict[str, list[str]]) -> str:
    lines: list[str] = []
    serial = 1
    for chain_id, residues in chain_sequences.items():
        for res_id, residue in enumerate(residues, start=1):
            lines.append(_atom_line(serial, residue, chain_id, res_id))
            serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    return "".join(lines)


def _write_afdb_case(
    root: Path,
    sample_id: str = "AF-0000000000000001-model_v1",
    chain_sequences: dict[str, list[str]] | None = None,
    paired_chains: tuple[str, ...] = (),
    sample_level_paired: bool = False,
    extra_main: bool = False,
    main_sequence: str = "AC",
) -> Path:
    chain_sequences = chain_sequences or {"A": ["ALA", "CYS"], "B": ["ALA", "CYS"]}
    chunk = root / "chunk" / "chunk0001"
    (chunk / "prior_pkl").mkdir(parents=True)
    (chunk / "gt").mkdir()
    (chunk / "bb_input").mkdir()
    (chunk / "gt" / f"{sample_id}.pdb").write_text(_pdb_text(chain_sequences))
    for chain_id, residues in chain_sequences.items():
        (chunk / "bb_input" / f"{sample_id}_{chain_id}.pdb").write_text(
            _pdb_text({chain_id: residues})
        )
    main_case = chunk / "msa_local" / "main" / sample_id
    main_case.mkdir(parents=True)
    msa = np.asarray([list(main_sequence)], dtype="<U1")
    np.savez(
        main_case / f"{sample_id}.npz",
        uniref90={
            "msa": msa,
            "deletion_matrix": np.zeros(msa.shape, dtype=int),
            "metadata": pd.DataFrame(),
        },
    )
    if extra_main:
        np.savez(
            main_case / f"{sample_id}_duplicate.npz",
            uniref90={
                "msa": msa,
                "deletion_matrix": np.zeros(msa.shape, dtype=int),
                "metadata": pd.DataFrame(),
            },
        )
    paired_case = chunk / "msa_local" / "paired" / sample_id
    paired_case.mkdir(parents=True)
    if sample_level_paired:
        (paired_case / f"{sample_id}.npz").write_bytes(b"paired")
    for chain_id in paired_chains:
        (paired_case / f"{sample_id}_{sample_id}_chain={chain_id}.npz").write_bytes(
            b"paired"
        )
    with (chunk / "prior_pkl" / f"{sample_id}.pkl").open("wb") as handle:
        pickle.dump(
            {
                sample_id: SimpleNamespace(
                    dmap={"A|B": [[1, 1, 7.5]]},
                    interactions={},
                    mesh={"A": [[0.0] * 10], "B": [[1.0] * 10]},
                    opm={},
                )
            },
            handle,
        )
    return root


def test_afdb_homodimer_shares_main_msa_without_fabricating_paired(
    tmp_path: Path,
) -> None:
    root = _write_afdb_case(tmp_path, paired_chains=())

    sample = AFDB10kReader(root).read("AF-0000000000000001-model_v1")

    assert set(sample.paths.main_msa_npzs) == {"AF-0000000000000001-model_v1"}
    assert sample.paths.paired_msa_npzs == {}
    assert sample.chain_sequences["A"] == sample.chain_sequences["B"]


def test_afdb_indexes_sample_level_paired_msa_without_chain_fabrication(
    tmp_path: Path,
) -> None:
    sample_id = "AF-0000000000000001-model_v1"
    root = _write_afdb_case(tmp_path, sample_level_paired=True)

    sample = AFDB10kReader(root).read(sample_id)

    assert sample.paths.paired_msa_npzs == {
        sample_id: (
            tmp_path
            / "chunk"
            / "chunk0001"
            / "msa_local"
            / "paired"
            / sample_id
            / f"{sample_id}.npz"
        )
    }
    assert "A" not in sample.paths.paired_msa_npzs
    assert "B" not in sample.paths.paired_msa_npzs


def test_afdb_paired_msa_availability_is_independent_per_chain(
    tmp_path: Path,
) -> None:
    root = _write_afdb_case(tmp_path, paired_chains=("B",))

    sample = AFDB10kReader(root).read("AF-0000000000000001-model_v1")

    assert set(sample.paths.paired_msa_npzs) == {"B"}
    assert sample.paths.paired_msa_npzs["B"].name.endswith("chain=B.npz")
    assert "A" not in sample.paths.paired_msa_npzs


def test_afdb_rejects_non_identical_homodimer_sequences(tmp_path: Path) -> None:
    root = _write_afdb_case(
        tmp_path,
        chain_sequences={"A": ["ALA", "CYS"], "B": ["ALA", "ASP"]},
    )

    with pytest.raises(ValueError, match="homodimer|identical"):
        AFDB10kReader(root)


def test_afdb_rejects_multiple_main_msa_representatives(tmp_path: Path) -> None:
    root = _write_afdb_case(tmp_path, extra_main=True)

    with pytest.raises(ValueError, match="main MSA"):
        AFDB10kReader(root)


def test_afdb_rejects_main_msa_query_mismatch(tmp_path: Path) -> None:
    root = _write_afdb_case(tmp_path, main_sequence="AD")

    with pytest.raises(ValueError, match="query sequence"):
        AFDB10kReader(root)


def test_afdb_rejects_main_msa_query_supersequence(tmp_path: Path) -> None:
    root = _write_afdb_case(tmp_path, main_sequence="ACDX")

    with pytest.raises(ValueError, match="query sequence"):
        AFDB10kReader(root)


def test_afdb_max_samples_zero_indexes_no_samples(tmp_path: Path) -> None:
    root = _write_afdb_case(tmp_path)

    reader = AFDB10kReader(root, max_samples=0)

    assert reader.sample_ids() == ()


def test_afdb_rejects_negative_max_samples(tmp_path: Path) -> None:
    (tmp_path / "chunk").mkdir()

    with pytest.raises(ValueError, match="max_samples"):
        AFDB10kReader(tmp_path, max_samples=-1)
