"""Strict reader for the canonical AFDB-10k-1 homodimer prior tree."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openfold3.core.data.io.sequence.msa import parse_msas_preparsed
from openfold3.core.data.resources.residues import STANDARD_RESIDUES_WITH_GAP_1
from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    INTERACTION_TYPES,
    InteractionRecord,
    PriorComplexSample,
    PriorSamplePaths,
    read_prior_pkl,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.index_cache import (
    load_or_create_reader_index,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    _canonical_msa_chain,
    _load_precomputed_dmap,
    _load_precomputed_interactions,
    _load_precomputed_mesh_presence,
    _load_precomputed_opm,
    _load_precomputed_seq,
    _precomputed_chain_id,
    _precomputed_sample_id,
)

AFDBSamplePaths = PriorSamplePaths
AFDBSample = PriorComplexSample

_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def _file_chain_id(path: Path, sample_id: str) -> str:
    stem = path.stem
    prefix = f"{sample_id}_"
    chain_id = (
        stem[len(prefix) :] if stem.startswith(prefix) else stem.rsplit("_", 1)[-1]
    )
    if chain_id.startswith("chain="):
        chain_id = chain_id[len("chain=") :]
    if not chain_id or "_" in chain_id or "=" in chain_id or "|" in chain_id:
        raise ValueError(f"AFDB path has an invalid chain id: {path}")
    return chain_id


def _pdb_chain_sequences(path: Path) -> dict[str, str]:
    residues: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            residue_name = line[17:20].strip().upper()
            chain_id = line[21].strip()
            residue_id = line[22:27].strip()
            if not chain_id:
                continue
            key = (chain_id, residue_id, line[16].strip())
            if key in seen:
                continue
            seen.add(key)
            residues.setdefault(chain_id, []).append(_AA3_TO_1.get(residue_name, "X"))
    return {chain_id: "".join(seq) for chain_id, seq in residues.items() if seq}


def _afdb_main_msa_path(main_root: Path, sample_id: str) -> Path | None:
    candidates: list[Path] = []
    nested = main_root / sample_id
    if nested.exists() and not nested.is_dir():
        raise ValueError(f"AFDB main MSA case path is not a directory: {nested}")
    if nested.is_dir():
        candidates.extend(sorted(nested.glob("*.npz")))
        for path in sorted(nested.rglob("*.npz")):
            if path.parent != nested:
                raise ValueError(f"malformed AFDB MSA path: {path}")
    candidates.extend(sorted(main_root.glob(f"{sample_id}.npz")))
    if len(candidates) > 1:
        raise ValueError(
            f"AFDB main MSA must have exactly one representative for {sample_id}"
        )
    if candidates and not candidates[0].is_file():
        raise ValueError(f"AFDB main MSA path is not a file: {candidates[0]}")
    return candidates[0] if candidates else None


def _main_msa_query_sequence(path: Path) -> str:
    parsed = parse_msas_preparsed([path])
    if not parsed:
        raise ValueError(f"AFDB main MSA is empty: {path}")
    first = next(iter(parsed.values())).msa
    if first.ndim != 2 or first.shape[0] == 0:
        raise ValueError(f"AFDB main MSA has no query row: {path}")
    row = np.asarray(first[0])
    if np.issubdtype(row.dtype, np.integer):
        symbols = [
            STANDARD_RESIDUES_WITH_GAP_1[int(value)]
            if 0 <= int(value) < len(STANDARD_RESIDUES_WITH_GAP_1)
            else "X"
            for value in row
        ]
    else:
        symbols = row.astype(str).tolist()
    return "".join(symbols)


def _afdb_paired_msa_paths(paired_root: Path, sample_id: str) -> dict[str, Path]:
    paired_msa_npzs: dict[str, Path] = {}
    if paired_root.exists() and not paired_root.is_dir():
        raise ValueError(f"AFDB paired MSA root is not a directory: {paired_root}")
    for path in sorted(paired_root.glob("*.npz")):
        raise ValueError(f"AFDB paired MSA must use nested case layout: {path}")
    paired_case = paired_root / sample_id
    if paired_case.exists() and not paired_case.is_dir():
        raise ValueError(f"AFDB paired MSA case path is not a directory: {paired_case}")
    for path in sorted(paired_case.glob("*.npz")):
        if path.stem == sample_id:
            if paired_msa_npzs:
                raise ValueError(f"mixed AFDB paired MSA layouts for {sample_id}")
            paired_msa_npzs[sample_id] = path
            continue
        if sample_id in paired_msa_npzs:
            raise ValueError(f"mixed AFDB paired MSA layouts for {sample_id}")
        chain_id = _canonical_msa_chain(path, sample_id)
        if chain_id in paired_msa_npzs:
            raise ValueError(
                f"duplicate AFDB paired MSA chain {chain_id!r} for {sample_id}"
            )
        paired_msa_npzs[chain_id] = path
    if paired_case.is_dir():
        for path in sorted(paired_case.rglob("*.npz")):
            if path.parent != paired_case:
                raise ValueError(f"malformed AFDB MSA path: {path}")
    return paired_msa_npzs


def _validate_max_samples(max_samples: int | None) -> int | None:
    if max_samples is None:
        return None
    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be None or a non-negative integer")
    return max_samples


class AFDB10kReader:
    """Index and parse visible AFDB-10k-1 homodimer samples."""

    def __init__(
        self,
        root: str | Path,
        max_samples: int | None = None,
        index_cache_file: str | Path | None = None,
    ):
        self.root = Path(root)
        self.max_samples = _validate_max_samples(max_samples)
        self.precomputed_only = next(self.root.glob("chunk*/*_gt.pt"), None) is not None
        self.final_root = self.root if self.precomputed_only else self.root / "chunk"
        if not self.final_root.is_dir():
            raise FileNotFoundError(f"missing final AFDB root: {self.final_root}")
        self._paths: dict[str, AFDBSamplePaths] = {}
        self._chain_sequences: dict[str, dict[str, str]] = {}
        if index_cache_file is None:
            self._index()
        else:
            self._paths, self._chain_sequences = load_or_create_reader_index(
                index_cache_file,
                {
                    "reader": "afdb10k",
                    "root": str(self.root.resolve()),
                    "max_samples": self.max_samples,
                    "precomputed_only": self.precomputed_only,
                },
                self._build_index_payload,
            )

    def _build_index_payload(
        self,
    ) -> tuple[dict[str, AFDBSamplePaths], dict[str, dict[str, str]]]:
        self._index()
        return self._paths, self._chain_sequences

    def _index(self) -> None:
        if self.max_samples == 0:
            return
        if self.precomputed_only:
            for gt_pt in sorted(self.final_root.glob("chunk*/*_gt.pt")):
                if not gt_pt.is_file():
                    raise ValueError(f"AFDB precomputed GT path is not a file: {gt_pt}")
                chunk = gt_pt.parent
                sample_id = _precomputed_sample_id(gt_pt)
                monomer_pdbs = {
                    _precomputed_chain_id(path, sample_id): path
                    for path in sorted(chunk.glob(f"{sample_id}_*_monomer.pt"))
                }
                if not monomer_pdbs:
                    raise ValueError(f"mandatory AFDB monomer pt missing for {sample_id}")
                chain_sequences = _load_precomputed_seq(
                    chunk / f"{sample_id}_seq.pt",
                    "AFDB",
                )
                if len(chain_sequences) != 2:
                    raise ValueError(
                        f"AFDB sample must have exactly two protein chains: {sample_id}"
                    )
                seq_values = tuple(chain_sequences.values())
                if seq_values[0] != seq_values[1]:
                    raise ValueError(
                        f"AFDB homodimer chains must have identical sequences: {sample_id}"
                    )
                if set(monomer_pdbs) != set(chain_sequences):
                    raise ValueError(
                        f"AFDB monomer chains must match seq.pt chains for {sample_id}"
                    )
                self._paths[sample_id] = AFDBSamplePaths(
                    sample_id=sample_id,
                    gt_pdb=gt_pt,
                    monomer_pdbs=monomer_pdbs,
                    main_msa_npzs={},
                    paired_msa_npzs={},
                    prior_pkl=chunk / f"{sample_id}.pkl",
                )
                self._chain_sequences[sample_id] = chain_sequences
                if self.max_samples is not None and len(self._paths) >= self.max_samples:
                    break
            return
        for pkl_path in sorted(self.final_root.glob("chunk*/prior_pkl/*.pkl")):
            if not pkl_path.is_file():
                raise ValueError(f"AFDB prior path is not a file: {pkl_path}")
            chunk = pkl_path.parents[1]
            sample_id = pkl_path.stem

            gt_pdb = chunk / "gt" / f"{sample_id}.pdb"
            if not gt_pdb.is_file():
                raise ValueError(f"mandatory AFDB GT path missing for {sample_id}")
            chain_sequences = _pdb_chain_sequences(gt_pdb)
            if len(chain_sequences) != 2:
                raise ValueError(
                    f"AFDB sample must have exactly two protein chains: {sample_id}"
                )
            seq_values = tuple(chain_sequences.values())
            if seq_values[0] != seq_values[1]:
                raise ValueError(
                    f"AFDB homodimer chains must have identical sequences: {sample_id}"
                )

            monomer_pdbs: dict[str, Path] = {}
            for path in sorted((chunk / "bb_input").glob(f"{sample_id}_*.pdb")):
                if not path.is_file():
                    raise ValueError(f"AFDB monomer path is not a file: {path}")
                chain_id = _file_chain_id(path, sample_id)
                if chain_id in monomer_pdbs:
                    raise ValueError(
                        f"duplicate monomer chain {chain_id!r} for {sample_id}"
                    )
                monomer_pdbs[chain_id] = path
            if set(monomer_pdbs) != set(chain_sequences):
                raise ValueError(
                    f"AFDB monomer chains must match GT chains for {sample_id}"
                )

            msa_root = chunk / "msa_local"
            main_root = msa_root / "main"
            if main_root.exists() and not main_root.is_dir():
                raise ValueError(f"AFDB main MSA root is not a directory: {main_root}")
            main_msa = _afdb_main_msa_path(main_root, sample_id)
            if main_msa is not None:
                main_query = _main_msa_query_sequence(main_msa)
                normalized_query = main_query.replace("-", "")
                expected_query = seq_values[0]
                trailing_padding = normalized_query[len(expected_query) :]
                if not (
                    normalized_query.startswith(expected_query)
                    and set(trailing_padding) <= {"X"}
                ):
                    raise ValueError(
                        "AFDB main MSA query sequence does not match the "
                        "homodimer chain (apart from trailing X padding) for "
                        f"{sample_id}"
                    )
            # Keep one representative key so the structure adapter can apply
            # the AFDB-specific shared-MSA policy and record provenance.  The
            # adapter copies this validated representative to the other chain;
            # expanding it here would bypass that policy.
            main_msa_npzs = {sample_id: main_msa} if main_msa is not None else {}

            paired_msa_npzs = _afdb_paired_msa_paths(msa_root / "paired", sample_id)
            paired_chain_keys = set(paired_msa_npzs) - {sample_id}
            unknown_chains = sorted(paired_chain_keys - set(monomer_pdbs))
            if unknown_chains:
                raise ValueError(
                    "paired MSA chains are absent from monomer PDBs for "
                    f"{sample_id}: {unknown_chains}"
                )

            if sample_id in self._paths:
                raise ValueError(f"duplicate sample id: {sample_id}")
            self._paths[sample_id] = AFDBSamplePaths(
                sample_id=sample_id,
                gt_pdb=gt_pdb,
                monomer_pdbs=monomer_pdbs,
                main_msa_npzs=main_msa_npzs,
                paired_msa_npzs=paired_msa_npzs,
                prior_pkl=pkl_path,
            )
            self._chain_sequences[sample_id] = chain_sequences
            if self.max_samples is not None and len(self._paths) >= self.max_samples:
                break

    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths))

    def locate(self, sample_id: str) -> AFDBSamplePaths:
        try:
            return self._paths[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown AFDB sample: {sample_id}") from exc

    def read(self, sample_id: str) -> AFDBSample:
        paths = self.locate(sample_id)
        if self.precomputed_only:
            chunk = paths.gt_pdb.parent
            return AFDBSample(
                sample_id=sample_id,
                dmap=_load_precomputed_dmap(chunk, sample_id, "AFDB"),
                interactions=_load_precomputed_interactions(chunk, sample_id, "AFDB"),
                mesh=_load_precomputed_mesh_presence(chunk, sample_id),
                opm=_load_precomputed_opm(chunk, sample_id),
                paths=paths,
                chain_sequences=self._chain_sequences[sample_id],
                dataset_name="AFDB",
            )
        return read_prior_pkl(
            paths.prior_pkl,
            sample_id,
            paths,
            "AFDB",
            chain_sequences=self._chain_sequences[sample_id],
        )


__all__ = [
    "AFDB10kReader",
    "AFDBSample",
    "AFDBSamplePaths",
    "INTERACTION_TYPES",
    "InteractionRecord",
]
