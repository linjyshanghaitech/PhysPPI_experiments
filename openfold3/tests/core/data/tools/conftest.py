"""Pytest configuration for tools tests -- VCR cassette directory."""

from pathlib import Path

import pytest

import openfold3

_CASSETTE_DIR = (
    Path(openfold3.__file__).parent / "tests" / "test_data" / "cassettes" / "test_rscb"
)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "cassette_library_dir": str(_CASSETTE_DIR),
    }

class _RCSBResponse:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def mock_rscb_requests(monkeypatch):

    def fake_post(_url, *, json, timeout):
        assert timeout == 30
        variables = json["variables"]
        if "ids" in variables:
            entries = []
            chain_mappings = {"1rnb": (["A", "B"], ["C", "A"]), "4pqx": (["A"], ["A"])}
            for pdb_id in variables["ids"]:
                chain_ids = chain_mappings.get(pdb_id.lower())
                if chain_ids is None:
                    continue
                entries.append(
                    {
                        "rcsb_id": pdb_id.upper(),
                        "polymer_entities": [
                            {
                                "rcsb_polymer_entity_container_identifiers": {
                                    "asym_ids": chain_ids[0],
                                    "auth_asym_ids": chain_ids[1],
                                }
                            }
                        ],
                    }
                )
            return _RCSBResponse({"data": {"entries": entries}})
        pdb_id = variables["pdb_id"].lower()
        if pdb_id == "4pqx":
            instance = {
                "rcsb_id": "4PQX.C",
                "rcsb_nonpolymer_instance_validation_score": [
                    {"ranking_model_fit": 0.8}
                ],
            }
            return _RCSBResponse(
                {"data": {"entry": {"nonpolymer_entities": [{"nonpolymer_entity_instances": [instance]}]}}}
            )
        return _RCSBResponse({"data": {"entry": None}})

    monkeypatch.setattr(
        "openfold3.core.data.tools.rscb.requests.post", fake_post
    )
