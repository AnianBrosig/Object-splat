from gsturn.cleanup import clean_job


def test_clean_job_keeps_outputs_data_and_symlink_targets(tmp_path):
    job = tmp_path / "job"
    (job / "work" / "frames" / "images").mkdir(parents=True)
    (job / "work" / "downscale").mkdir(parents=True)
    (job / "outputs").mkdir()
    (job / "work" / "frames" / "images" / "a.jpg").write_bytes(b"x" * 100)
    (job / "work" / "frames" / "frames.json").write_text("{}")
    (job / "work" / "features.npz").write_bytes(b"n")
    (job / "outputs" / "preview.png").write_bytes(b"p")
    (job / "outputs" / "turntable.mp4").write_bytes(b"v")
    original = tmp_path / "orig.MOV"
    original.write_bytes(b"o" * 50)
    (job / "work" / "downscale" / "video_small.mp4").symlink_to(original)

    dry = clean_job(job, dry_run=True)
    assert dry["files"] == 2 and (job / "work" / "frames" / "images" / "a.jpg").exists()

    r = clean_job(job)
    assert r["files"] == 2 and r["bytes"] == 100
    assert not (job / "work" / "frames" / "images").exists()          # emptied folder pruned
    assert original.exists()                                           # link removed, target intact
    assert (job / "work" / "frames" / "frames.json").exists() and (job / "work" / "features.npz").exists()
    assert (job / "outputs" / "preview.png").exists() and (job / "outputs" / "turntable.mp4").exists()
