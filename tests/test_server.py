import json

import pytest
from fastapi.testclient import TestClient

import gsturn.server as server
from gsturn.config import Config
from gsturn.events import Events


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(server, "VIDEOS_DIR", tmp_path / "videos")
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "a.mp4").write_bytes(b"x")
    (tmp_path / "videos" / "notes.txt").write_text("x")
    (tmp_path / "videos" / "calibration.MOV").write_bytes(b"x")     # never offered as a job video
    job = tmp_path / "jobs" / "j1"
    (job / "work" / "frames").mkdir(parents=True)
    Config(video=str(tmp_path / "videos" / "a.mp4")).save(job / "config.json")
    (job / "work" / "frames" / "plot.png").write_bytes(b"png")
    (tmp_path / "secret.txt").write_text("secret")
    ev = Events(job, echo=False)
    ev.stage_status("downscale", "running")
    ev.stage_status("downscale", "done", "ok")
    ev.close()
    return TestClient(server.app)


def test_videos_and_jobs(client):
    assert [v["name"] for v in client.get("/api/videos").json()] == ["a.mp4"]
    jobs = client.get("/api/jobs").json()
    assert jobs[0]["id"] == "j1" and jobs[0]["status"]["downscale"] == "done"
    st = client.get("/api/jobs/j1").json()
    assert st["stages"]["downscale"]["status"] == "done" and st["running"] is False


def test_files_are_confined_to_job(client):
    assert client.get("/api/jobs/j1/files/work/frames/plot.png").content == b"png"
    assert client.get("/api/jobs/j1/files/../../secret.txt").status_code == 404
    assert client.get("/api/jobs/j1/files/%2e%2e/%2e%2e/secret.txt").status_code == 404
    assert client.get("/api/jobs/..%2fjobs/files/config.json").status_code in (400, 404)


def test_run_validates_input(client):
    r = client.post("/api/jobs/j1/run", json={"config": {"bogus": 1}})
    assert r.status_code == 400
    r = client.post("/api/jobs/j1/run", json={"from": "nonsense"})
    assert r.status_code == 400


def test_create_job_rejects_missing(client):
    assert client.post("/api/jobs", data={"path": "/nope.mp4"}).status_code == 400
    assert client.post("/api/jobs", data={}).status_code == 400


def test_defaults(client):
    d = client.get("/api/defaults").json()
    assert d["target_height"] == 0 and d["max_frames"] == 200


def test_stages_single_source(client):
    from gsturn.config import STAGES
    assert client.get("/api/stages").json() == STAGES
    st = client.get("/api/jobs/j1").json()
    assert st["order"] == STAGES and set(st["stages"]) == set(STAGES)
    assert STAGES.index("quality") == STAGES.index("segment") + 1
    assert STAGES.index("refine") == STAGES.index("sfm") + 1
