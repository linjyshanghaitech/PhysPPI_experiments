import torch

from openfold3.core.data.framework.data_module import openfold_batch_collator


def test_physppi_single_chain_mask_keeps_chain_dimension() -> None:
    batch = openfold_batch_collator(
        [{"physppi_paired_msa_chain_mask": torch.tensor([True])}]
    )

    assert batch["physppi_paired_msa_chain_mask"].shape == (1, 1)
    assert batch["physppi_paired_msa_chain_mask"].tolist() == [[True]]
