from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

from scripts.dev.physppi_all_prior_finite_smoke import _load_audit
from scripts.dev.physppi_audit_canonical_priors import audit_dataset


def _atom_line(serial: int, residue: str, chain: str, res_id: int) -> str:
    x = float(serial)
    return (
        f"ATOM  {serial:5d}  CA  {residue:>3s} {chain:1s}{res_id:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}           C\n"
    )


def _pdb_text(residue_ids: dict[str, tuple[int, ...]]) -> str:
    lines = []
    serial = 1
    for chain_id, ids in residue_ids.items():
        for res_id in ids:
            lines.append(_atom_line(serial, "ALA", chain_id, res_id))
            serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    return "".join(lines)


def _write_qbiolip_case(
    root: Path,
    sample_id: str,
    payload: object,
    *,
    main_msa_chains: tuple[str, ...] = ("A", "B"),
) -> None:
    chunk = root / "chunk" / "chunk0001"
    (chunk / "prior_pkl").mkdir(parents=True)
    (chunk / "gt").mkdir()
    (chunk / "bb_input").mkdir()
    (chunk / "gt" / f"{sample_id}.pdb").write_text(_pdb_text({"A": (0,), "B": (-6,)}))
    (chunk / "bb_input" / f"{sample_id}_A.pdb").write_text(_pdb_text({"A": (0,)}))
    (chunk / "bb_input" / f"{sample_id}_B.pdb").write_text(_pdb_text({"B": (-6,)}))
    main = chunk / "msa_local" / "main"
    main.mkdir(parents=True)
    for chain_id in main_msa_chains:
        (main / f"{sample_id}_{sample_id}_chain={chain_id}.npz").write_bytes(b"main")
    with (chunk / "prior_pkl" / f"{sample_id}.pkl").open("wb") as handle:
        pickle.dump({sample_id: payload}, handle)


def test_audit_allows_opm_none_signed_pdb_resseq_and_charge_nan(tmp_path: Path) -> None:
    _write_qbiolip_case(
        tmp_path,
        "ok",
        SimpleNamespace(
            dmap={"A|B": [[0, -6, 7.25]]},
            interactions={"A|B": [{"r1": 0, "r2": -6, "t": "hbond"}]},
            mesh={"A": [[1.0, 2.0, 3.0, math.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]},
            opm=None,
        ),
    )

    report = audit_dataset("qbiolip", tmp_path)

    assert report["scanned"] == 1
    assert report["fatal"] == {"ids": [], "reasons": {}}
    assert report["allowed"] == {"opm_none": ["ok"], "mesh_charge_nan": ["ok"]}


def test_audit_marks_missing_exact_pdb_resseq_fatal(tmp_path: Path) -> None:
    _write_qbiolip_case(
        tmp_path,
        "bad_resseq",
        SimpleNamespace(
            dmap={"A|B": [[1, -6, 7.25]]},
            interactions={},
            mesh={},
            opm={},
        ),
    )

    report = audit_dataset("qbiolip", tmp_path)

    assert report["fatal"]["ids"] == ["bad_resseq"]
    assert report["fatal"]["reasons"] == {"bad_resseq": ["missing_dmap_resseq:A:1"]}


def test_audit_records_query_only_main_msa_as_warning(tmp_path: Path) -> None:
    _write_qbiolip_case(
        tmp_path,
        "missing_msa",
        SimpleNamespace(dmap={}, interactions={}, mesh={}, opm={}),
        main_msa_chains=("A",),
    )

    report = audit_dataset("qbiolip", tmp_path)

    assert report["warnings"] == {"main_msa_query_only": ["missing_msa"]}
    assert report["fatal"] == {"ids": [], "reasons": {}}


def test_smoke_loads_only_fatal_ids_from_repeatable_audit_json(tmp_path: Path) -> None:
    first = tmp_path / "qbiolip.json"
    second = tmp_path / "afdb.json"
    first.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "qbiolip": {
                        "scanned": 2,
                        "allowed": {
                            "opm_none": ["valid_opm"],
                            "mesh_charge_nan": ["valid_nan"],
                        },
                        "warnings": {"main_msa_query_only": ["query_only"]},
                        "fatal": {"ids": ["bad"], "reasons": {"bad": ["missing_gt"]}},
                    }
                },
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "afdb": {
                        "scanned": 1,
                        "allowed": {"opm_none": [], "mesh_charge_nan": []},
                        "warnings": {"main_msa_query_only": []},
                        "fatal": {"ids": [], "reasons": {}},
                    }
                },
            }
        )
    )

    audit = _load_audit([first, second])

    assert audit["fatal_ids"] == {"qbiolip": {"bad"}, "afdb": set()}
    assert audit["allowed_states"]["qbiolip"]["opm_none"] == ["valid_opm"]
    assert "query_only" not in audit["fatal_ids"]["qbiolip"]


def test_prior_only_smoke_summary_separates_audit_and_finite_fields(
    tmp_path: Path, monkeypatch
) -> None:
    import torch

    import scripts.dev.physppi_all_prior_finite_smoke as smoke

    args = SimpleNamespace(
        runner_yaml=tmp_path / "runner.yml",
        audit_json=[tmp_path / "audit.json"],
        max_samples=None,
        out_dir=tmp_path / "out",
        max_tokens=8,
        fail_fast=False,
    )
    audit = {
        "fatal_ids": {"qbiolip": {"bad_data"}, "afdb": set()},
        "data_errors": {
            "qbiolip": [{"sample_id": "bad_data", "reasons": ["missing_main_msa:A"]}],
            "afdb": [],
        },
        "allowed_states": {
            "qbiolip": {"opm_none": ["ok"], "mesh_charge_nan": ["ok"]},
            "afdb": {"opm_none": [], "mesh_charge_nan": []},
        },
    }
    monkeypatch.setattr(
        smoke,
        "_build_prior_readers",
        lambda runner_yaml, audit_ids, max_samples: [("qbiolip", object(), ["ok"])],
    )
    monkeypatch.setattr(
        smoke,
        "_build_prior_only_probe",
        lambda device: (object(), object(), 1, 1),
    )
    monkeypatch.setattr(smoke, "_read_smoke_sample", lambda reader, sample_id: object())
    monkeypatch.setattr(
        smoke,
        "_run_prior_only_one",
        lambda **kwargs: {"sample_id": kwargs["sample_id"], "loss": 0.0},
    )

    assert smoke._run_prior_only(args, audit, torch.device("cpu")) == 1

    summary = json.loads((args.out_dir / "summary.json").read_text())
    records = (args.out_dir / "records.jsonl").read_text().splitlines()
    assert summary["data_errors"] == audit["data_errors"]
    assert summary["finite_failures"] == []
    assert summary["allowed_states"] == audit["allowed_states"]
    assert summary["checked"] == {"qbiolip": ["ok"], "afdb": []}
    assert json.loads(records[0])["sample_id"] == "ok"


def test_prior_only_one_forces_available_slots_and_threads_charge_mask(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    import numpy as np
    import torch

    import scripts.dev.physppi_all_prior_finite_smoke as smoke
    from openfold3.projects.of3_all_atom.physppi.data.adapters import (
        _runtime as structure_adapter,
    )

    class AtomArray:
        chain_id = np.asarray(["A", "B"])
        res_id = np.asarray([1, 2])

        def get_annotation_categories(self):
            return ["entity_id", "token_id"]

    class Conditioner:
        def __call__(self, *, s_init, z_init, batch, token_mask, pair_mask):
            assert batch["physppi_contact_active"].tolist() == [True]
            assert batch["physppi_interaction_active"].tolist() == [True]
            assert batch["physppi_surface_active"].tolist() == [True]
            assert batch["physppi_tm_active"].tolist() == [True]
            assert "physppi_surface_charge_mask" in batch
            assert batch["physppi_surface_charge_mask"].shape == (1, 2, 128)
            assert not bool(batch["physppi_surface_charge_mask"][0, 0, 0])
            return SimpleNamespace(
                s_init=s_init.detach().clone().requires_grad_(),
                z_init=z_init.detach().clone().requires_grad_(),
            )

        def named_parameters(self):
            return ()

    class Optimizer:
        def zero_grad(self, set_to_none=True):
            return None

        def step(self):
            return None

    monkeypatch.setattr(
        structure_adapter,
        "_parse_pdb_atom_array",
        lambda path: AtomArray(),
        raising=False,
    )
    monkeypatch.setattr(
        structure_adapter,
        "_token_starts",
        lambda atom_array: np.asarray([0, 1]),
        raising=False,
    )
    monkeypatch.setattr(
        smoke,
        "build_contact_features",
        lambda *args, **kwargs: {
            "mask": np.ones((2, 2), dtype=bool),
            "surface_interface_mask": np.ones(2, dtype=bool),
        },
    )
    monkeypatch.setattr(
        smoke,
        "build_interaction_features",
        lambda *args, **kwargs: {"type_mask": np.ones((2, 2), dtype=bool)},
    )
    charge_mask = np.ones((2, 128), dtype=bool)
    charge_mask[0, 0] = False
    monkeypatch.setattr(
        smoke,
        "build_surface_features",
        lambda *args, **kwargs: {
            "coords": np.zeros((2, 128, 3), dtype=np.float32),
            "features": np.zeros((2, 128, 6), dtype=np.float32),
            "mask": np.ones((2, 128), dtype=bool),
            "charge_mask": charge_mask,
        },
    )
    monkeypatch.setattr(
        smoke, "build_tm_features", lambda *args, **kwargs: np.ones(2, dtype=bool)
    )
    monkeypatch.setattr(
        smoke, "build_membrane_target", lambda *args, **kwargs: {"valid": True}
    )
    sample = SimpleNamespace(
        dmap={"A|B": ((1, 2, 5.0),)},
        interactions=(object(),),
        mesh={"A": ((0.0,) * 10,)},
        opm={"res_loc": {"A": {1: 1}}},
        paths=SimpleNamespace(gt_pdb=tmp_path / "gt.pdb"),
    )

    result = smoke._run_prior_only_one(
        sample=sample,
        sample_id="case",
        conditioner=Conditioner(),
        optimizer=Optimizer(),
        c_s=1,
        c_z=1,
        device=torch.device("cpu"),
        max_tokens=8,
    )

    assert result["sample_id"] == "case"
