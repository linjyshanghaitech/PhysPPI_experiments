import importlib

import numpy as np
from biotite.structure import AtomArray


def _structure_adapter_module():
    importlib.import_module(
        "openfold3.core.data.framework.single_datasets.prior_complex"
    )
    return importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.data.adapters"
    )


def _atomized_fme(atom_names: list[str]) -> AtomArray:
    atoms = AtomArray(len(atom_names))
    atoms.coord = np.zeros((len(atom_names), 3), dtype=np.float32)
    atoms.chain_id = np.asarray(["A"] * len(atom_names))
    atoms.res_id = np.asarray([1] * len(atom_names))
    atoms.res_name = np.asarray(["FME"] * len(atom_names))
    atoms.atom_name = np.asarray(atom_names)
    atoms.element = np.asarray([name[0] for name in atom_names])
    atoms.set_annotation("is_atomized", np.ones(len(atom_names), dtype=bool))
    atoms.set_annotation("token_id", np.arange(len(atom_names)))
    atoms.set_annotation("token_position", np.arange(len(atom_names)))
    return atoms


def test_atomized_template_matches_each_shared_atom_once():
    module = _structure_adapter_module()
    query = _atomized_fme(["N", "CN", "O1", "CA", "C", "O"])
    template = _atomized_fme(["N", "CA", "C", "O"])

    template_slice = module._crop_monomer_template_to_query_tokens(template, query)

    assert template_slice is not None
    assert template_slice.atom_array.atom_name.tolist() == ["N", "CA", "C", "O"]
    assert template_slice.query_token_positions.tolist() == [0, 3, 4, 5]
    # Template features are defined per residue; the single FME residue expands
    # across its four query tokens.
    assert template_slice.template_residue_repeats.tolist() == [4]
