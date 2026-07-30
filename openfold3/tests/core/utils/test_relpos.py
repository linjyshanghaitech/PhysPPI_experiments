import torch

from openfold3.core.utils.relpos import relpos_complex


def test_relpos_complex_ignores_residue_offsets_across_crop_segments() -> None:
    batch = {
        "residue_index": torch.tensor([[1, 2, 10, 11]], dtype=torch.int32),
        "token_index": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        "asym_id": torch.tensor([[1, 1, 1, 1]], dtype=torch.int32),
        "entity_id": torch.tensor([[1, 1, 1, 1]], dtype=torch.int32),
        "sym_id": torch.tensor([[1, 1, 1, 1]], dtype=torch.int32),
        "crop_segment_id": torch.tensor([[1, 1, 2, 2]], dtype=torch.int32),
    }

    relpos = relpos_complex(batch, max_relative_idx=2, max_relative_chain=2)
    rel_pos_bins = relpos[..., :6]

    assert rel_pos_bins[0, 0, 1, 1].item() == 1
    assert rel_pos_bins[0, 1, 2, 5].item() == 1
    assert rel_pos_bins[0, 2, 3, 1].item() == 1


def test_relpos_complex_preserves_legacy_behavior_without_crop_segments() -> None:
    batch = {
        "residue_index": torch.tensor([[1, 2, 10]], dtype=torch.int32),
        "token_index": torch.tensor([[0, 1, 2]], dtype=torch.int32),
        "asym_id": torch.tensor([[1, 1, 1]], dtype=torch.int32),
        "entity_id": torch.tensor([[1, 1, 1]], dtype=torch.int32),
        "sym_id": torch.tensor([[1, 1, 1]], dtype=torch.int32),
    }

    relpos = relpos_complex(batch, max_relative_idx=2, max_relative_chain=2)
    rel_pos_bins = relpos[..., :6]

    assert rel_pos_bins[0, 1, 2, 0].item() == 1
