import json
from pathlib import Path

import pytest
import torch

from experiments.permutation_alignment.inspect_chain_identity_alignment import (
    inspect_token_identity_features,
    write_json_atomic,
)
from openfold3.core.data.resources.residues import STANDARD_RESIDUES_WITH_GAP_3
from openfold3.projects.of3_all_atom.physppi.screening.models import DatasetName


def _minimal_features() -> tuple[dict, torch.Tensor]:
    token_count = 2
    restype = torch.zeros(token_count, len(STANDARD_RESIDUES_WITH_GAP_3))
    restype[:, 0] = 1
    common = {
        "token_index": torch.tensor([10, 11]),
        "token_mask": torch.ones(token_count, dtype=torch.bool),
        "atom_mask": torch.ones(token_count, dtype=torch.bool),
        "num_atoms_per_token": torch.ones(token_count, dtype=torch.long),
        "start_atom_index": torch.arange(token_count),
        "restype": restype,
        "is_protein": torch.zeros(token_count, dtype=torch.long),
        "is_rna": torch.zeros(token_count, dtype=torch.long),
        "is_dna": torch.zeros(token_count, dtype=torch.long),
        "is_ligand": torch.zeros(token_count, dtype=torch.long),
        "is_atomized": torch.ones(token_count, dtype=torch.long),
        "asym_id": torch.tensor([1, 2]),
        "entity_id": torch.tensor([7, 8]),
        "sym_id": torch.tensor([1, 1]),
    }
    atom_positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    features = {
        **common,
        "ref_space_uid": torch.tensor([0, 1]),
        "ref_space_uid_to_perm": {
            0: torch.zeros(1, 1, dtype=torch.long),
            1: torch.zeros(1, 1, dtype=torch.long),
        },
        "ground_truth": {
            **common,
            "atom_positions": atom_positions.clone(),
            "atom_resolved_mask": torch.ones(token_count, dtype=torch.bool),
        },
    }
    return features, atom_positions


def test_inspect_token_identity_features_reports_chain_labels_and_alignment() -> None:
    features, atom_positions = _minimal_features()

    report = inspect_token_identity_features(
        dataset=DatasetName.AFDB,
        sample_id="AF-fixture",
        pred_features=features,
        predicted_atom_positions=atom_positions,
        max_token_rows=8,
    )

    assert report["dataset"] == "afdb"
    assert report["sample_id"] == "AF-fixture"
    assert report["chain_label_keys"]["gt"] == {
        "token_index": True,
        "asym_id": True,
        "entity_id": True,
        "sym_id": True,
    }
    assert report["counts"] == {
        "pred_tokens": 2,
        "gt_tokens": 2,
        "pred_atoms": 2,
        "gt_atoms": 2,
    }
    assert report["selector_error"] is None
    assert report["mismatch_counts"] == {
        "token_index": 0,
        "asym_id": 0,
        "entity_id": 0,
        "sym_id": 0,
    }
    assert report["token_identity_alignment"]["status"] == "accepted"
    assert report["token_identity_alignment"]["reason"] == "token_index_identity"
    assert report["token_identity_alignment"]["atom_index_count"] == 2
    assert report["token_rows"] == [
        {
            "pred_row": 0,
            "gt_row": 0,
            "pred_token_index": 10,
            "gt_token_index": 10,
            "pred_asym_id": 1,
            "gt_asym_id": 1,
            "pred_entity_id": 7,
            "gt_entity_id": 7,
            "pred_sym_id": 1,
            "gt_sym_id": 1,
        },
        {
            "pred_row": 1,
            "gt_row": 1,
            "pred_token_index": 11,
            "gt_token_index": 11,
            "pred_asym_id": 2,
            "gt_asym_id": 2,
            "pred_entity_id": 8,
            "gt_entity_id": 8,
            "pred_sym_id": 1,
            "gt_sym_id": 1,
        },
    ]


def test_write_json_atomic_rejects_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("keep\n")

    with pytest.raises(FileExistsError, match="inspection output already exists"):
        write_json_atomic({"status": "new"}, output)

    assert output.read_text() == "keep\n"


def test_write_json_atomic_writes_sorted_pretty_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"

    write_json_atomic({"b": 2, "a": 1}, output)

    assert json.loads(output.read_text()) == {"a": 1, "b": 2}
    assert output.read_text().splitlines()[1].startswith('  "a"')
