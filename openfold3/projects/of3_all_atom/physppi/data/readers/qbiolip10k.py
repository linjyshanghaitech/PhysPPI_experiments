"""Strict reader for the canonical Q-BioLiP-10k-new prior schema."""

from __future__ import annotations

from pathlib import Path

from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    INTERACTION_TYPES,
    InteractionRecord,
    PriorComplexSample,
    PriorSamplePaths,
    parse_dmap,
    parse_interactions,
    read_prior_pkl,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.index_cache import (
    load_or_create_reader_index,
)

QBioLiPSamplePaths = PriorSamplePaths
QBioLiPSample = PriorComplexSample


def _file_chain_id(path: Path, sample_id: str) -> str:
    """Extract the chain id from ``<sample>_<chain>.<suffix>`` or ``<chain>``."""

    stem = path.stem
    prefix = f"{sample_id}_"
    if "chain=" in stem:
        chain_id = stem.rsplit("chain=", 1)[1].split("_", 1)[0]
    else:
        chain_id = (
            stem[len(prefix) :] if stem.startswith(prefix) else stem.rsplit("_", 1)[-1]
        )
    if not chain_id:
        raise ValueError(f"Q-BioLiP path has an empty chain id: {path}")
    if "|" in chain_id:
        raise ValueError(f"Q-BioLiP path has an invalid chain id: {path}")
    return chain_id


def _canonical_msa_alias(path: Path, sample_id: str) -> tuple[str, int]:
    """Parse an exact MSA basename and return ``(chain_id, alias_rank)``.

    Real Q-BioLiP paired caches contain two canonical aliases for the same
    chain: ``..._<chain>.npz`` and ``..._chain=<chain>.npz``.  The explicit
    ``chain=`` form has higher precedence when both aliases are present.
    """

    prefix = f"{sample_id}_{sample_id}_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"malformed Q-BioLiP MSA basename: {path.name}")
    suffix = path.stem[len(prefix) :]
    alias_rank = 1
    if suffix.startswith("chain="):
        suffix = suffix[len("chain=") :]
        alias_rank = 0
    if not suffix or "_" in suffix or "=" in suffix or "|" in suffix:
        raise ValueError(f"malformed Q-BioLiP MSA chain suffix: {path.name}")
    return suffix, alias_rank


def _canonical_msa_chain(path: Path, sample_id: str) -> str:
    """Parse an exact ``<sample>_<sample>_<chain>.npz`` basename."""

    return _canonical_msa_alias(path, sample_id)[0]


def _flat_msa_matches(path: Path, sample_id: str) -> bool:
    """Return whether a flat main-MSA file is prefixed for ``sample_id``."""

    return path.stem.startswith(f"{sample_id}_")


def _validate_max_samples(max_samples: int | None) -> int | None:
    if max_samples is None:
        return None
    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be None or a non-negative integer")
    return max_samples


class QBioLiP10kReader:
    """Index and parse only the canonical visible Q-BioLiP-10k-new tree.

    ``root`` must contain ``chunk/chunk*/prior_pkl/*.pkl``.  Staging trees such
    as ``.chunk`` and legacy ``prior_processed``/``cmap`` layouts are never
    considered.
    """

    def __init__(
        self,
        root: str | Path,
        max_samples: int | None = None,
        index_cache_file: str | Path | None = None,
    ):
        self.root = Path(root)
        self.max_samples = _validate_max_samples(max_samples)
        self.final_root = self.root / "chunk"
        if not self.final_root.is_dir():
            raise FileNotFoundError(f"missing final Q-BioLiP root: {self.final_root}")
        if index_cache_file is None:
            self._paths = self._index()
        else:
            self._paths = load_or_create_reader_index(
                index_cache_file,
                {
                    "reader": "qbiolip10k",
                    "root": str(self.root.resolve()),
                    "max_samples": self.max_samples,
                },
                self._index,
            )

    def _index(self) -> dict[str, QBioLiPSamplePaths]:
        indexed: dict[str, QBioLiPSamplePaths] = {}
        if self.max_samples == 0:
            return indexed
        for pkl_path in sorted(self.final_root.glob("chunk*/prior_pkl/*.pkl")):
            if not pkl_path.is_file():
                raise ValueError(f"Q-BioLiP prior path is not a file: {pkl_path}")
            chunk = pkl_path.parents[1]
            sample_id = pkl_path.stem

            monomer_pdbs: dict[str, Path] = {}
            for path in sorted((chunk / "bb_input").glob(f"{sample_id}_*.pdb")):
                if not path.is_file():
                    raise ValueError(f"Q-BioLiP monomer path is not a file: {path}")
                chain_id = _file_chain_id(path, sample_id)
                if chain_id in monomer_pdbs:
                    raise ValueError(
                        f"duplicate monomer chain {chain_id!r} for {sample_id}"
                    )
                monomer_pdbs[chain_id] = path

            msa_root = chunk / "msa_local"
            main_root = msa_root / "main"
            main_msa_npzs: dict[str, Path] = {}
            main_msa_precedence_ranks: dict[str, tuple[int, int]] = {}
            if main_root.exists() and not main_root.is_dir():
                raise ValueError(
                    f"Q-BioLiP main MSA root is not a directory: {main_root}"
                )

            def add_msa_path(
                path: Path,
                destination: dict[str, Path],
                msa_kind: str,
                current_sample_id: str = sample_id,
                precedence_ranks: dict[str, tuple[int, int]] | None = None,
                layout_rank: int = 0,
            ) -> None:
                if not path.is_file():
                    raise ValueError(f"Q-BioLiP MSA path is not a file: {path}")
                chain_id, alias_rank = _canonical_msa_alias(path, current_sample_id)
                precedence_rank = (alias_rank, layout_rank)
                if chain_id in destination:
                    previous_rank = (
                        precedence_ranks.get(chain_id)
                        if precedence_ranks is not None
                        else None
                    )
                    if previous_rank is not None and previous_rank != precedence_rank:
                        if precedence_rank < previous_rank:
                            destination[chain_id] = path
                            precedence_ranks[chain_id] = precedence_rank
                        return
                    raise ValueError(
                        f"duplicate {msa_kind} MSA chain {chain_id!r} "
                        f"for {current_sample_id}"
                    )
                destination[chain_id] = path
                if precedence_ranks is not None:
                    precedence_ranks[chain_id] = precedence_rank

            for path in sorted(main_root.glob("*.npz")):
                if _flat_msa_matches(path, sample_id):
                    add_msa_path(
                        path,
                        main_msa_npzs,
                        "main",
                        precedence_ranks=main_msa_precedence_ranks,
                        layout_rank=2,
                    )
            nested_main = main_root / sample_id
            if nested_main.exists() and not nested_main.is_dir():
                raise ValueError(
                    f"Q-BioLiP main MSA case path is not a directory: {nested_main}"
                )
            for path in sorted(nested_main.glob("*.npz")):
                add_msa_path(
                    path,
                    main_msa_npzs,
                    "main",
                    precedence_ranks=main_msa_precedence_ranks,
                )
            repeated_nested_main = nested_main / sample_id
            if repeated_nested_main.exists() and not repeated_nested_main.is_dir():
                raise ValueError(
                    "Q-BioLiP repeated main MSA case path is not a directory: "
                    f"{repeated_nested_main}"
                )
            for path in sorted(repeated_nested_main.glob("*.npz")):
                add_msa_path(
                    path,
                    main_msa_npzs,
                    "main",
                    precedence_ranks=main_msa_precedence_ranks,
                    layout_rank=1,
                )
            if nested_main.is_dir():
                allowed_parents = {nested_main}
                if repeated_nested_main.is_dir():
                    allowed_parents.add(repeated_nested_main)
                for path in sorted(nested_main.rglob("*.npz")):
                    if path.parent not in allowed_parents:
                        raise ValueError(f"malformed Q-BioLiP MSA path: {path}")

            paired_root = msa_root / "paired"
            paired_msa_npzs: dict[str, Path] = {}
            paired_msa_precedence_ranks: dict[str, tuple[int, int]] = {}
            if paired_root.exists() and not paired_root.is_dir():
                raise ValueError(
                    f"Q-BioLiP paired MSA root is not a directory: {paired_root}"
                )
            for path in sorted(paired_root.glob("*.npz")):
                raise ValueError(
                    f"Q-BioLiP paired MSA must use nested case layout: {path}"
                )
            paired_case = paired_root / sample_id
            if paired_case.exists() and not paired_case.is_dir():
                raise ValueError(
                    f"Q-BioLiP paired MSA case path is not a directory: {paired_case}"
                )
            for path in sorted(paired_case.glob("*.npz")):
                add_msa_path(
                    path,
                    paired_msa_npzs,
                    "paired",
                    precedence_ranks=paired_msa_precedence_ranks,
                )
            repeated_paired_case = paired_case / sample_id
            if repeated_paired_case.exists() and not repeated_paired_case.is_dir():
                raise ValueError(
                    "Q-BioLiP repeated paired MSA case path is not a directory: "
                    f"{repeated_paired_case}"
                )
            for path in sorted(repeated_paired_case.glob("*.npz")):
                add_msa_path(
                    path,
                    paired_msa_npzs,
                    "paired",
                    precedence_ranks=paired_msa_precedence_ranks,
                    layout_rank=1,
                )
            if paired_case.is_dir():
                allowed_parents = {paired_case}
                if repeated_paired_case.is_dir():
                    allowed_parents.add(repeated_paired_case)
                for path in sorted(paired_case.rglob("*.npz")):
                    if path.parent not in allowed_parents:
                        raise ValueError(f"malformed Q-BioLiP MSA path: {path}")

            monomer_chains = set(monomer_pdbs)
            for msa_chains in (main_msa_npzs, paired_msa_npzs):
                for unknown_chain in sorted(set(msa_chains) - monomer_chains):
                    del msa_chains[unknown_chain]

            paths = QBioLiPSamplePaths(
                sample_id=sample_id,
                gt_pdb=chunk / "gt" / f"{sample_id}.pdb",
                monomer_pdbs=monomer_pdbs,
                main_msa_npzs=main_msa_npzs,
                paired_msa_npzs=paired_msa_npzs,
                prior_pkl=pkl_path,
            )
            if not paths.gt_pdb.is_file() or not paths.monomer_pdbs:
                raise ValueError(f"mandatory Q-BioLiP path missing for {sample_id}")
            if sample_id in indexed:
                raise ValueError(f"duplicate sample id: {sample_id}")
            indexed[sample_id] = paths
            if self.max_samples is not None and len(indexed) >= self.max_samples:
                break
        return indexed

    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths))

    def locate(self, sample_id: str) -> QBioLiPSamplePaths:
        try:
            return self._paths[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown Q-BioLiP sample: {sample_id}") from exc

    @staticmethod
    def _parse_dmap(raw: object) -> dict[str, tuple[tuple[int, int, float], ...]]:
        return parse_dmap(raw, "Q-BioLiP")

    @staticmethod
    def _parse_interactions(raw: object) -> tuple[InteractionRecord, ...]:
        return parse_interactions(raw, "Q-BioLiP")

    def read(self, sample_id: str) -> QBioLiPSample:
        paths = self.locate(sample_id)
        return read_prior_pkl(paths.prior_pkl, sample_id, paths, "Q-BioLiP")


__all__ = [
    "INTERACTION_TYPES",
    "InteractionRecord",
    "QBioLiPSample",
    "QBioLiPSamplePaths",
    "QBioLiP10kReader",
]
