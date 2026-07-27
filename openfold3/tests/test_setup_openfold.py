from openfold3.setup_openfold import setup_biotite_ccd


def test_setup_biotite_ccd(tmp_path, monkeypatch):
    ccd_path = tmp_path / "test_ccd.cif"
    monkeypatch.setattr(
        "openfold3.setup_openfold.download_s3_file", lambda *_: ccd_path.touch()
    )
    has_downloaded = setup_biotite_ccd(ccd_path=ccd_path, force_download=False)
    assert ccd_path.exists()
    assert has_downloaded

    has_downloaded = setup_biotite_ccd(ccd_path=ccd_path, force_download=False)
    assert not has_downloaded
