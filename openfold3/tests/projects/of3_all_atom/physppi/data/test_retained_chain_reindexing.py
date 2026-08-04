from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from biotite.structure import AtomArray

from openfold3.projects.of3_all_atom.physppi.data.adapters.chain_views import (
    RetainedChainView,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
    OptionalSlot,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    InteractionRecord,
    PriorComplexSample,
    PriorSamplePaths,
)


def _adapter_module():
    import openfold3.core.data.framework.single_datasets.afdb10k  # noqa: F401
    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module

    return module


def _six_chain_sample(tmp_path: Path) -> PriorComplexSample:
    chain_ids = tuple("ABCDEF")
    paths = PriorSamplePaths(
        sample_id="six-chain",
        gt_pdb=tmp_path / "six-chain.pdb",
        monomer_pdbs={chain_id: tmp_path / f"{chain_id}.pdb" for chain_id in chain_ids},
        main_msa_npzs={
            chain_id: tmp_path / f"{chain_id}.main.npz" for chain_id in chain_ids
        },
        paired_msa_npzs={
            chain_id: tmp_path / f"{chain_id}.paired.npz" for chain_id in chain_ids
        }
        | {"six-chain": tmp_path / "shared.paired.npz"},
        prior_pkl=tmp_path / "prior.pkl",
    )
    dmap = {
        f"{left}|{right}": ((1, 1, 12.0),)
        for left, right in zip(chain_ids[:-1], chain_ids[1:], strict=True)
    }
    interactions = tuple(
        InteractionRecord(
            chain_pair=f"{left}|{right}",
            r1=1,
            r2=1,
            interaction_type="hbond",
            distance=3.0,
            atom_1=("N",),
            atom_2=("O",),
        )
        for left, right in zip(chain_ids[:-1], chain_ids[1:], strict=True)
    )
    return PriorComplexSample(
        sample_id="six-chain",
        dmap=dmap,
        interactions=interactions,
        mesh={
            chain_id: ((float(index), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),)
            for index, chain_id in enumerate(chain_ids)
        },
        opm={
            "res_loc": {chain_id: {1: 1} for chain_id in chain_ids},
            "outer_normal": [0.0, 0.0, 1.0],
            "outer_centroid": [0.0, 0.0, 0.0],
        },
        paths=paths,
        chain_sequences={chain_id: "A" * 140 for chain_id in chain_ids},
        dataset_name="Q-BioLiP",
    )


def _decode_chain_ids(encoded: torch.Tensor) -> tuple[str, ...]:
    return tuple(
        bytes(int(value) for value in row if int(value)).decode("utf-8")
        for row in encoded
    )


def test_restrict_prior_sample_filters_every_chain_owned_payload(tmp_path: Path):
    module = _adapter_module()
    sample = _six_chain_sample(tmp_path)

    restricted = module.restrict_prior_sample_to_chains(sample, ("B", "C", "D"))

    assert restricted.chain_ids == ("B", "C", "D")
    assert set(restricted.paths.monomer_pdbs) == {"B", "C", "D"}
    assert set(restricted.paths.main_msa_npzs) == {"B", "C", "D"}
    assert set(restricted.paths.paired_msa_npzs) == {
        "B",
        "C",
        "D",
        "six-chain",
    }
    assert set(restricted.chain_sequences) == {"B", "C", "D"}
    assert set(restricted.mesh) == {"B", "C", "D"}
    assert set(restricted.opm["res_loc"]) == {"B", "C", "D"}
    assert set(restricted.dmap) == {"B|C", "C|D"}
    assert {edge.chain_pair for edge in restricted.interactions} == {"B|C", "C|D"}


def test_restrict_prior_sample_filters_canonical_opm_interval_rows(tmp_path: Path):
    module = _adapter_module()
    sample = _six_chain_sample(tmp_path)
    sample.opm["res_loc"] = [
        ["A", 1, 10, -1],
        ["B", 11, 20, 0],
        ["D", 21, 30, 1],
        ["F", 31, 40, -1],
    ]

    restricted = module.restrict_prior_sample_to_chains(sample, ("B", "D"))

    assert restricted.opm["res_loc"] == [
        ["B", 11, 20, 0],
        ["D", 21, 30, 1],
    ]


def test_restricted_sample_filters_gt_atom_rows_before_canonical_processing(
    tmp_path: Path,
):
    module = _adapter_module()
    sample = module.restrict_prior_sample_to_chains(
        _six_chain_sample(tmp_path), ("B", "D")
    )
    atoms = AtomArray(6)
    atoms.chain_id = np.asarray(tuple("ABCDEF"))
    atoms.res_id = np.ones(6, dtype=int)
    atoms.atom_name = np.asarray(["CA"] * 6)
    atoms.res_name = np.asarray(["ALA"] * 6)
    atoms.coord = np.zeros((6, 3), dtype=np.float32)

    restricted = module._restrict_gt_atom_array_to_sample(atoms, sample)

    assert tuple(restricted.chain_id) == ("B", "D")


def test_restricted_optional_availability_matches_filtered_payloads(tmp_path: Path):
    module = _adapter_module()
    sample = _six_chain_sample(tmp_path)
    connected = module.restrict_prior_sample_to_chains(sample, ("B", "C"))
    disconnected = module.restrict_prior_sample_to_chains(sample, ("B", "D"))

    assert module.PhysPPIPriorComplexDatasetAdapter._available_slots(connected) == {
        OptionalSlot.INTERFACE_RESIDUE: True,
        OptionalSlot.INTERACTION: True,
        OptionalSlot.SURFACE: True,
        OptionalSlot.TM: True,
    }
    assert module.PhysPPIPriorComplexDatasetAdapter._available_slots(disconnected) == {
        OptionalSlot.INTERFACE_RESIDUE: False,
        OptionalSlot.INTERACTION: False,
        OptionalSlot.SURFACE: True,
        OptionalSlot.TM: True,
    }


def test_build_sample_features_selects_one_view_before_canonical_featurization(
    tmp_path: Path,
    monkeypatch,
):
    module = _adapter_module()
    sample = _six_chain_sample(tmp_path)
    adapter = object.__new__(module.PhysPPIPriorComplexDatasetAdapter)
    adapter._physppi_data_seed = 17
    adapter.name = "qbiolip"
    adapter.occurrence = SimpleNamespace(epoch=2, global_slot=31)
    adapter.debug_mode = False
    seen: dict[str, PriorComplexSample] = {}

    def fake_canonical(*, sample, **_):
        seen["sample"] = sample
        return {"features": {}, "atom_array_cropped": object()}

    monkeypatch.setattr(adapter, "create_canonical_all_features", fake_canonical)
    monkeypatch.setattr(adapter, "finish_physppi_features", lambda _sample, _data: {})

    features = adapter.build_sample_features(
        sample, {"preferred_chain_or_interface": None}
    )

    retained = _decode_chain_ids(features["physppi_retained_chain_ids"])
    assert 2 <= len(retained) <= 4
    assert retained == seen["sample"].chain_ids
    assert _decode_chain_ids(features["physppi_original_chain_ids"]) == tuple("ABCDEF")
    assert set(seen["sample"].paths.monomer_pdbs) == set(retained)
    assert all(
        set(chain_pair.split("|")) <= set(retained)
        for chain_pair in seen["sample"].dmap
    )
    assert int(features["physppi_retained_target_chain_count"]) == len(retained)
    assert float(features["physppi_retained_dmap_cutoff"]) == 18.0
    assert features["physppi_retained_occurrence_seed"].item() != 0
    assert features["physppi_retained_chain_ids"].shape == (len(retained), 16)
    assert features["physppi_retained_seed_edge"].shape == (2, 16)
    assert features["pdb_id"] == "six-chain"


def test_untriggered_sample_retains_all_chains(tmp_path: Path):
    module = _adapter_module()
    sample = _six_chain_sample(tmp_path)
    sample = PriorComplexSample(
        **{
            **sample.__dict__,
            "chain_sequences": {chain_id: "A" * 20 for chain_id in sample.chain_ids},
        }
    )
    occurrence_rng = module.build_occurrence_rng(5, 0, 0, "qbiolip", sample.sample_id)

    view = module.select_retained_chain_view(
        sample, occurrence_rng.chain_view, occurrence_rng.seed
    )

    assert view.trigger == "none"
    assert view.source_chain_ids == tuple("ABCDEF")
    assert view.target_chain_count == 6


def test_retained_view_provenance_collates_with_variable_chain_counts():
    module = _adapter_module()
    from openfold3.core.data.framework.data_module import openfold_batch_collator

    first = module.encode_retained_chain_view(
        RetainedChainView(
            source_chain_ids=("A", "B"),
            trigger="token_budget",
            component_chain_ids=("A", "B", "C"),
            seed_edge=("A", "B"),
            target_chain_count=2,
            parent_sample_id="first",
            original_chain_ids=("A", "B", "C", "D", "E"),
            occurrence_seed=11,
        )
    )
    second = module.encode_retained_chain_view(
        RetainedChainView(
            source_chain_ids=("X", "Y", "Z"),
            trigger="compatibility_group",
            component_chain_ids=("W", "X", "Y", "Z"),
            seed_edge=("X", "Y"),
            target_chain_count=3,
            parent_sample_id="second",
            original_chain_ids=("W", "X", "Y", "Z", "Q", "R"),
            occurrence_seed=12,
        )
    )

    batch = openfold_batch_collator([first, second])

    assert batch["physppi_retained_chain_ids"].shape == (2, 3, 16)
    assert batch["physppi_original_chain_ids"].shape == (2, 6, 16)
    assert batch["physppi_retained_component_chain_ids"].shape == (2, 4, 16)
    assert batch["physppi_retained_seed_edge"].shape == (2, 2, 16)
