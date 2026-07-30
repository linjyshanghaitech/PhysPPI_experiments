import torch

from openfold3.core.utils.atom_attention_block_utils import convert_single_rep_to_blocks


def test_convert_single_rep_to_blocks_masks_cross_crop_segment_pairs() -> None:
    ql = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    atom_mask = torch.ones(1, 4)
    atom_crop_segment_id = torch.tensor([[1, 1, 2, 2]], dtype=torch.int32)

    _, _, pair_mask = convert_single_rep_to_blocks(
        ql=ql,
        n_query=2,
        n_key=4,
        atom_mask=atom_mask,
        atom_crop_segment_id=atom_crop_segment_id,
    )

    assert pair_mask[0, 0, 0, 0].item() == 1
    assert pair_mask[0, 0, 1, 1].item() == 1
    assert pair_mask[0, 0, 0, 2].item() == 0
    assert pair_mask[0, 0, 1, 3].item() == 0


def test_convert_single_rep_to_blocks_preserves_mask_without_crop_segments() -> None:
    ql = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    atom_mask = torch.ones(1, 4)

    _, _, pair_mask = convert_single_rep_to_blocks(
        ql=ql,
        n_query=2,
        n_key=4,
        atom_mask=atom_mask,
    )

    assert pair_mask[0, 0].sum().item() == 8
