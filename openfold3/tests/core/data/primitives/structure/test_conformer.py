# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import pytest
from func_timeout import FunctionTimedOut
from rdkit import Chem

from openfold3.core.data.primitives.structure.conformer import (
    CONFORMER_STRATEGIES,
    ConformerGenerationError,
    ConformerResult,
    _compute_conformer,
    multistrategy_compute_conformer,
    renumber_permutations,
)


@pytest.fixture
def ethanol():
    return Chem.MolFromSmiles("CCO")


def _stub_with_outcomes(outcomes):
    """Build a side_effect callable that consumes `outcomes` in order.

    Each entry is either a return value `(mol, conf_id)` or a BaseException to raise.
    Note: FunctionTimedOut extends BaseException (not Exception).
    """
    iterator = iter(outcomes)

    def side_effect(mol, **kwargs):
        outcome = next(iterator)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return side_effect


def test_renumber_permutations_accepts_empty_list():
    out = renumber_permutations([], required_gt_atoms=[1, 2, 4])

    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_renumber_permutations_accepts_empty_ndarray():
    out = renumber_permutations(np.array([], dtype=int), required_gt_atoms=[1, 2, 4])

    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_all_strategies_fail_raises(ethanol):
    timeout_exc = FunctionTimedOut("timed out")
    # One outcome per strategy in the chain.
    outcomes = [timeout_exc] * len(CONFORMER_STRATEGIES)
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer._compute_conformer",
            side_effect=_stub_with_outcomes(outcomes),
        ),
        pytest.raises(ConformerGenerationError) as excinfo,
    ):
        multistrategy_compute_conformer(ethanol)

    assert excinfo.value.__cause__ is timeout_exc


@pytest.mark.parametrize(
    ("start_from", "expected_strategy"),
    [
        pytest.param("default", "default", id="start_from_default"),
        pytest.param(
            "small_ring_torsions",
            "small_ring_torsions",
            id="start_from_small_ring_torsions",
        ),
        pytest.param(
            "random_init", "random_init", id="start_from_random_init_skips_earlier"
        ),
    ],
)
def test_start_from_slices_chain(ethanol, start_from, expected_strategy):
    with patch(
        "openfold3.core.data.primitives.structure.conformer._compute_conformer",
        side_effect=_stub_with_outcomes([(ethanol, 0)]),
    ) as mock_cc:
        result = multistrategy_compute_conformer(ethanol, start_from=start_from)

    assert result.strategy == expected_strategy
    assert mock_cc.call_count == 1
    expected_kwargs = next(
        s.kwargs for s in CONFORMER_STRATEGIES if s.name == start_from
    )
    for k, v in expected_kwargs.items():
        assert mock_cc.call_args.kwargs[k] == v


def test_start_from_unknown_raises(ethanol):
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer._compute_conformer",
            side_effect=_stub_with_outcomes([]),
        ),
        pytest.raises(ValueError, match="Unknown conformer strategy"),
    ):
        multistrategy_compute_conformer(ethanol, start_from="bogus")


def test_result_is_frozen_dataclass():
    result = ConformerResult(mol=Chem.MolFromSmiles("C"), conf_id=0, strategy="default")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.strategy = "random_init"


def test_strategy_names_match_cache_wire_values():
    # Hard constraint: cached `conformer_gen_strategy` strings are these literals.
    names = [s.name for s in CONFORMER_STRATEGIES]
    assert "default" in names
    assert "random_init" in names
    assert "small_ring_torsions" in names


def test_end_to_end_real_rdkit(ethanol):
    # No patching: exercises the real RDKit path on a small molecule.
    result = multistrategy_compute_conformer(ethanol)
    assert result.strategy in {s.name for s in CONFORMER_STRATEGIES}
    assert result.mol.GetNumConformers() >= 1
    result.mol.GetConformer(result.conf_id)  # would raise if id is invalid


# ----------------------------------------------------------------------------
# compute_conformer tests
# ----------------------------------------------------------------------------

# A fixed seed for the random-coord initializer so snapshots are reproducible
# regardless of where this test sits in the seeded_rng-derived global RNG state.
_FIXED_RDKIT_SEED = 424242


@pytest.mark.parametrize(
    ("smiles", "use_random_coord_init", "remove_hs"),
    [
        pytest.param("CCO", False, True, id="ethanol_default_no_hs"),
        pytest.param("CCO", True, True, id="ethanol_random_init_no_hs"),
        pytest.param("CCO", False, False, id="ethanol_default_keep_hs"),
        pytest.param("c1ccccc1", False, True, id="benzene_default_no_hs"),
        pytest.param("CC(=O)Nc1ccc(O)cc1", False, True, id="paracetamol_default_no_hs"),
    ],
)
def test_compute_conformer_snapshot(
    smiles, use_random_coord_init, remove_hs, ndarrays_regression
):
    mol = Chem.MolFromSmiles(smiles)

    # Pin the RDKit seed (sourced from random.randint inside compute_conformer) so the
    # generated coordinates are reproducible across test runs and parametrize ordering.
    with patch(
        "openfold3.core.data.primitives.structure.conformer.random.randint",
        return_value=_FIXED_RDKIT_SEED,
    ):
        out_mol, conf_id = _compute_conformer(
            mol,
            use_random_coord_init=use_random_coord_init,
            remove_hs=remove_hs,
            timeout=None,
        )

    assert conf_id == 0
    coords = out_mol.GetConformer(conf_id).GetPositions().astype(np.float64)
    elements = np.array([a.GetSymbol() for a in out_mol.GetAtoms()], dtype="U2")
    if remove_hs:
        assert "H" not in set(elements.tolist())
    else:
        assert "H" in set(elements.tolist())

    ndarrays_regression.check(
        {"coords": coords},
        # ETKDGv3 is deterministic for a fixed seed within one RDKit build, but
        # numerics drift across RDKit minor versions and CPU archs (aarch64 vs
        # x86_64). Observed drift on paracetamol is ~1.5e-3 Å; give it headroom
        # while staying ≪ chemistry scale (~0.1 Å) to still catch real regressions.
        default_tolerance=dict(atol=5e-3, rtol=5e-3),
    )


@pytest.mark.parametrize(
    ("use_random_coord_init",),
    [
        pytest.param(False, id="default_init"),
        pytest.param(True, id="random_init"),
    ],
)
def test_compute_conformer_embed_returns_minus_one_raises(
    ethanol, use_random_coord_init
):
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer.AllChem.EmbedMolecule",
            return_value=-1,
        ),
        pytest.raises(ConformerGenerationError, match="Failed to generate"),
    ):
        _compute_conformer(
            ethanol,
            use_random_coord_init=use_random_coord_init,
            timeout=None,
        )


def test_compute_conformer_timeout_propagates(ethanol):
    timeout_exc = FunctionTimedOut("timed out")
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer.func_timeout",
            side_effect=timeout_exc,
        ) as mock_ft,
        pytest.raises(FunctionTimedOut),
    ):
        _compute_conformer(ethanol, timeout=0.001)

    assert mock_ft.call_count == 1
    # The wrapper should have been handed the actual EmbedMolecule callable + the
    # supplied timeout value.
    assert mock_ft.call_args.kwargs["timeout"] == 0.001


def test_compute_conformer_timeout_none_skips_func_timeout(ethanol):
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer.func_timeout"
        ) as mock_ft,
        patch(
            "openfold3.core.data.primitives.structure.conformer.AllChem.EmbedMolecule",
            return_value=0,
        ) as mock_embed,
    ):
        _compute_conformer(ethanol, timeout=None)

    mock_ft.assert_not_called()
    mock_embed.assert_called_once()


def test_compute_conformer_addhs_failure_is_non_fatal(ethanol, caplog):
    # AddHs raising must not abort conformer generation — it should log and proceed.
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer.Chem.AddHs",
            side_effect=RuntimeError("addhs blew up"),
        ),
        patch(
            "openfold3.core.data.primitives.structure.conformer.AllChem.EmbedMolecule",
            return_value=0,
        ),
        caplog.at_level("WARNING"),
    ):
        mol_out, conf_id = _compute_conformer(ethanol, timeout=None)

    assert conf_id == 0
    assert mol_out is not None
    assert any("Failed to add hydrogens" in rec.message for rec in caplog.records), (
        caplog.text
    )


def test_compute_conformer_empty_mol_raises():
    # An empty Mol has no atoms; RDKit's EmbedMolecule raises ValueError for that
    # rather than returning -1, so the error propagates as-is. Either way the
    # function must not return silently. ConformerGenerationError subclasses
    # ValueError, so this assertion accepts either.
    empty_mol = Chem.Mol()
    with pytest.raises(ValueError):
        _compute_conformer(empty_mol, timeout=None)


@pytest.mark.parametrize(
    "use_small_ring_torsions",
    [
        pytest.param(False, id="flag_off"),
        pytest.param(True, id="flag_on"),
    ],
)
def test_compute_conformer_small_ring_torsions_flag_propagates(
    ethanol, use_small_ring_torsions
):
    # Verify the kwarg actually flips the corresponding ETKDGv3 attribute on the
    # strategy object handed to EmbedMolecule.
    with patch(
        "openfold3.core.data.primitives.structure.conformer.AllChem.EmbedMolecule",
        return_value=0,
    ) as mock_embed:
        _compute_conformer(
            ethanol,
            use_small_ring_torsions=use_small_ring_torsions,
            timeout=None,
        )

    embed_strategy = mock_embed.call_args.args[1]
    assert embed_strategy.useSmallRingTorsions is use_small_ring_torsions


# ----------------------------------------------------------------------------
# PDB-CCD metallotetrapyrrole exercise of the multistrategy chain (real RDKit)
# ----------------------------------------------------------------------------

_DEFAULT_FAILURE_EXCEPTIONS = (ConformerGenerationError, FunctionTimedOut)


class _TetrapyrroleCase(NamedTuple):
    ccd_code: str
    smiles: str
    default_succeeds: bool


_PDB_TETRAPYRROLES = [
    pytest.param(
        *_TetrapyrroleCase(
            ccd_code="HEM",
            smiles=(
                "Cc1c2n3c(c1CCC(=O)O)C=C4C(=C(C5=[N]4[Fe]36[N]7=C(C=C8N6C(=C5)"
                "C(=C8C)C=C)C(=C(C7=C2)C)C=C)C)CCC(=O)O"
            ),
            default_succeeds=False,
        ),
        id="hem_fe_porphyrin",
    ),
    pytest.param(
        *_TetrapyrroleCase(
            ccd_code="CLA",
            smiles=(
                "CCC1=C(C2=Cc3c(c(c4n3[Mg]56[N]2=C1C=C7N5C8=C([C@H](C(=O)C8=C7C)"
                "C(=O)OC)C9=[N]6C(=C4)[C@H]([C@@H]9CCC(=O)OC/C=C(\\C)/CCC[C@H](C)"
                "CCC[C@H](C)CCCC(C)C)C)C)C=C)C"
            ),
            default_succeeds=True,
        ),
        id="cla_chlorophyll_a",
    ),
    pytest.param(
        *_TetrapyrroleCase(
            ccd_code="CHL",
            smiles=(
                "CCC1=C(c2cc3c(c(c4n3[Mg]56[n+]2c1cc7n5c8c(c9[n+]6c(c4)"
                "C(C9CCC(=O)OC/C=C(\\C)/CCC[C@H](C)CCC[C@H](C)CCCC(C)C)C)"
                "[C@H](C(=O)c8c7C)C(=O)OC)C)C=C)C=O"
            ),
            default_succeeds=True,
        ),
        id="chl_chlorophyll_b",
    ),
    pytest.param(
        *_TetrapyrroleCase(
            ccd_code="BCL",
            smiles=(
                "CC[C@@H]1[C@H](C2=CC3=C(C(=C4[N-]3[Mg+2]56[N]2=C1C=C7[N-]5C8="
                "C([C@H](C(=O)C8=C7C)C(=O)OC)C9=[N]6C(=C4)[C@H]([C@@H]9CCC(=O)OC"
                "/C=C(\\C)/CCC[C@H](C)CCC[C@H](C)CCCC(C)C)C)C)C(=O)C)C"
            ),
            default_succeeds=True,
        ),
        id="bcl_bacteriochlorophyll_a",
    ),
]


@pytest.mark.parametrize(("ccd_code", "smiles", "default_succeeds"), _PDB_TETRAPYRROLES)
def test_pdb_tetrapyrrole_default_strategy_outcome(ccd_code, smiles, default_succeeds):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"failed to parse PDB CCD {ccd_code!r} SMILES"

    if default_succeeds:
        mol_out, conf_id = _compute_conformer(
            mol, use_small_ring_torsions=False, timeout=15.0
        )
        assert conf_id == 0
        assert mol_out.GetNumConformers() >= 1
    else:
        with pytest.raises(_DEFAULT_FAILURE_EXCEPTIONS):
            _compute_conformer(mol, use_small_ring_torsions=False, timeout=5.0)


@pytest.mark.parametrize(("ccd_code", "smiles", "default_succeeds"), _PDB_TETRAPYRROLES)
def test_pdb_tetrapyrrole_small_ring_torsions_succeeds(
    ccd_code,
    smiles,
    default_succeeds,
):
    del (
        ccd_code,
        default_succeeds,
    )  # unused since small_ring_torsions should succeed for all cases
    mol = Chem.MolFromSmiles(smiles)
    # Pin the RDKit seed so the test is reproducible: tetrapyrroles are large
    # macrocycles and ETKDGv3 + small-ring-torsions has high variance — some
    # seeds succeed in < 5s, others time out or return -1. 60s headroom on top
    # because x86_64 CI under `pytest-xdist -n auto` is slower than local aarch64.
    with patch(
        "openfold3.core.data.primitives.structure.conformer.random.randint",
        return_value=_FIXED_RDKIT_SEED,
    ):
        mol_out, conf_id = _compute_conformer(
            mol, use_small_ring_torsions=True, timeout=60.0
        )
    assert conf_id == 0
    assert mol_out.GetNumConformers() >= 1
    coords = mol_out.GetConformer(conf_id).GetPositions()
    assert np.isfinite(coords).all()


@pytest.mark.parametrize(("ccd_code", "smiles", "default_succeeds"), _PDB_TETRAPYRROLES)
def test_pdb_tetrapyrrole_chain_resolves(ccd_code, smiles, default_succeeds):
    # Chain order is default → small_ring_torsions → random_init. Whichever
    # entry first delivers a conformer wins; for current RDKit that's `default`
    # on the chlorin/bacteriochlorin cores and `small_ring_torsions` on HEM.
    mol = Chem.MolFromSmiles(smiles)
    # Pin seed + 60s timeouts: see note on the small_ring_torsions test above.
    with patch(
        "openfold3.core.data.primitives.structure.conformer.random.randint",
        return_value=_FIXED_RDKIT_SEED,
    ):
        result = multistrategy_compute_conformer(
            mol,
            timeouts={
                "default": 60.0,
                "small_ring_torsions": 60.0,
                "random_init": 30.0,
            },
        )
    assert isinstance(result, ConformerResult)
    expected = "default" if default_succeeds else "small_ring_torsions"
    assert result.strategy == expected


# ----------------------------------------------------------------------------
# renumber_permutations tests
# ----------------------------------------------------------------------------


class TestRenumberPermutations:
    """Tests for ``renumber_permutations`` covering the NaN-fallback regression
    (list input, empty input) and forward-looking edge cases."""

    # -- empty inputs --------------------------------------------------------

    @pytest.mark.parametrize(
        "empty_perm",
        [
            pytest.param([], id="empty_list"),
            pytest.param(np.array([], dtype=int), id="empty_ndarray_1d"),
            pytest.param(np.array([], dtype=int).reshape(0, 2), id="empty_ndarray_2d"),
        ],
    )
    def test_empty_permutation_returns_empty(self, empty_perm):
        """Empty permutations (list or ndarray) must be returned as-is after
        conversion to ndarray, without raising AttributeError."""
        result = renumber_permutations(empty_perm, required_gt_atoms=[1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    # -- list input (the exact regression) -----------------------------------

    def test_list_input_renumbers_correctly(self):
        """Non-empty Python list must be accepted and renumbered identically
        to an ndarray with the same values."""
        perm_list = [[2, 4], [4, 2]]
        required = [1, 2, 4, 5, 6]  # superset of perm indices

        result_list = renumber_permutations(perm_list, required)
        result_ndarray = renumber_permutations(np.array(perm_list), required)

        np.testing.assert_array_equal(result_list, result_ndarray)
        # required_gt_atoms [1,2,4,5,6] → mapping: 1→0, 2→1, 4→2, 5→3, 6→4
        # [2,4] → [1,2]  and  [4,2] → [2,1]
        np.testing.assert_array_equal(result_list, np.array([[1, 2], [2, 1]]))

    def test_list_int32_renumbers_correctly(self):
        """List input with int32 ndarray produces identical result."""
        perm_list = [[2, 4], [4, 2]]
        required = [1, 2, 4, 5, 6]

        result_list = renumber_permutations(perm_list, required)
        result_ndarray = renumber_permutations(
            np.array(perm_list, dtype=np.int32), required
        )

        np.testing.assert_array_equal(result_list, result_ndarray)

    def test_single_atom_permutation_list(self):
        """Single-atom permutations (1D list) must produce correct result."""
        result = renumber_permutations([2, 4], required_gt_atoms=[1, 2, 4, 7])
        # required [1,2,4,7] → mapping: 1→0, 2→1, 4→2, 7→3
        # [2,4] → [1,2]
        np.testing.assert_array_equal(result, np.array([1, 2]))

    # -- ndarray input (existing behaviour, regression guard) ----------------

    def test_ndarray_standard_case_matches_docstring(self):
        """Match the documented example exactly."""
        perm = np.array([[2, 4], [4, 2]])
        required = [1, 2, 4, 5, 6]
        result = renumber_permutations(perm, required)
        # required [1,2,4,5,6] → mapping: 1→0, 2→1, 4→2, 5→3, 6→4
        # [2,4]→[1,2], [4,2]→[2,1]
        np.testing.assert_array_equal(result, np.array([[1, 2], [2, 1]]))

    def test_ndarray_no_renumbering_when_already_compact(self):
        """When required_gt_atoms is already 0..N-1, indices should stay unchanged."""
        perm = np.array([[0, 1], [1, 0]])
        required = [0, 1]
        result = renumber_permutations(perm, required)
        np.testing.assert_array_equal(result, perm)

    def test_ndarray_identity_single_atom(self):
        """Single-element permutation that only references itself."""
        perm = np.array([0])
        required = [0, 1, 2]
        result = renumber_permutations(perm, required)
        np.testing.assert_array_equal(result, np.array([0]))

    def test_ndarray_renumbering_with_gaps(self):
        """Large gaps in required_gt_atoms must not affect correctness."""
        perm = np.array([[5, 10], [10, 5]])
        required = [5, 7, 10, 100]
        # 5→0, 7→1, 10→2, 100→3
        expected = np.array([[0, 2], [2, 0]])
        result = renumber_permutations(perm, required)
        np.testing.assert_array_equal(result, expected)

    def test_ndarray_3d_permutation_renumbered_correctly(self):
        """3-column permutations must work (e.g., 3 symmetry-equivalent atoms)."""
        perm = np.array([[1, 3, 5], [3, 5, 1]])
        required = [1, 3, 5, 7]
        # 1→0, 3→1, 5→2, 7→3
        expected = np.array([[0, 1, 2], [1, 2, 0]])
        result = renumber_permutations(perm, required)
        np.testing.assert_array_equal(result, expected)

    # -- forward-looking: hostile inputs -------------------------------------

    def test_zero_length_required_gt_atoms_raises(self):
        """If required_gt_atoms is empty, the empty dict causes KeyError
        on the first lookup (model consistency constraint)."""
        with pytest.raises(KeyError):
            renumber_permutations(np.array([[2, 4]]), required_gt_atoms=[])

    def test_missing_required_atom_raises_keyerror(self):
        """A permutation index absent from required_gt_atoms must raise
        KeyError (hard error — data integrity)."""
        perm = np.array([[2, 4]])
        required = [1, 2]  # 4 is missing
        with pytest.raises(KeyError):
            renumber_permutations(perm, required)

    def test_negative_indices_raises_keyerror(self):
        """Negative indices in permutation must raise KeyError."""
        perm = np.array([[-1, 2]])
        required = [2, 3]
        with pytest.raises(KeyError):
            renumber_permutations(perm, required)

    def test_non_integer_input_truncated_silently(self):
        """Float values are truncated to int by np.asarray(dtype=int).
        This is acceptable because permutation indices are always integers
        in practice — the silent truncation matches standard numpy behaviour."""
        perm = [[2.5, 4.0]]
        required = [2, 4, 5]
        result = renumber_permutations(perm, required)
        # 2.5→2→0, 4.0→4→1 (after truncation and remapping)
        np.testing.assert_array_equal(result, np.array([[0, 1]]))

    # -- forward-looking: result dtype and shape invariants ------------------

    def test_result_dtype_is_int(self):
        """Output dtype must be integer (any int variant)."""
        perm = np.array([[2, 4], [4, 2]])
        result = renumber_permutations(perm, required_gt_atoms=[1, 2, 4, 5])
        assert np.issubdtype(result.dtype, np.integer)

    def test_result_shape_preserved(self):
        """Output shape must match input shape."""
        shapes = [(2, 2), (1, 3), (5, 1), (3,), (3, 4)]
        required = list(range(20))
        for shape in shapes:
            perm = np.arange(np.prod(shape)).reshape(shape) % 10
            result = renumber_permutations(perm, required_gt_atoms=required)
            assert result.shape == shape, f"shape {shape} became {result.shape}"

    def test_result_values_are_monotonic_remap(self):
        """Every output index must be in [0, len(required_gt_atoms)-1]."""
        perm = np.array([[5, 8, 2], [2, 5, 8]])
        required = [2, 5, 8, 10, 20]
        result = renumber_permutations(perm, required)
        assert result.min() >= 0
        assert result.max() < len(required)

    # -- forward-looking: list-of-lists beyond 2D ----------------------------

    def test_list_of_lists_3d_renumbers_correctly(self):
        """list-of-lists with 3 columns must work."""
        perm = [[1, 3, 5], [5, 1, 3]]
        required = [1, 3, 5]
        result = renumber_permutations(perm, required)
        np.testing.assert_array_equal(result, np.array([[0, 1, 2], [2, 0, 1]]))
