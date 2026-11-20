"""Resource scheduler: exclusion, job order, overlap of different resources, backfill, dead holders."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import sys, time
from gsturn import scheduler
res, job, prio, hold, log = sys.argv[1].split(","), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
t0 = time.time()
with scheduler.acquire(res, job, "test", prio, poll=0.05) as lease:
    start = time.time()
    open(log, "a").write(f"{job} start {start:.3f}\n")
    time.sleep(hold)
    open(log, "a").write(f"{job} end {time.time():.3f}\n")
"""


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ, GSTURN_LOCK_DIR=str(tmp_path / "locks"), PYTHONPATH=str(ROOT))
    return e, tmp_path / "log.txt"


def spawn(env, res, job, prio, hold, log):
    return subprocess.Popen([sys.executable, "-c", SCRIPT, res, job, str(prio), str(hold), str(log)], env=env,
                            cwd=ROOT)


def events(log):
    out = {}
    for line in Path(log).read_text().splitlines():
        job, kind, t = line.split()
        out[(job, kind)] = float(t)
    return out


def test_one_at_a_time_and_earliest_job_first(env):
    e, log = env
    a = spawn(e, "cpu", "A", 1, 1.0, log)
    time.sleep(0.4)                          # A holds the cpu
    b = spawn(e, "cpu", "B", 3, 0.3, log)    # B asks first but started its job later
    time.sleep(0.2)
    c = spawn(e, "cpu", "C", 2, 0.3, log)
    for p in (a, b, c):
        assert p.wait(30) == 0
    ev = events(log)
    assert ev[("A", "end")] <= ev[("C", "start")] + 0.01       # exclusion
    assert ev[("C", "end")] <= ev[("B", "start")] + 0.01       # job C started before job B: it goes first


def test_different_resources_overlap(env):
    e, log = env
    a = spawn(e, "gpu", "A", 1, 1.2, log)
    time.sleep(0.4)
    b = spawn(e, "cpu", "B", 2, 0.2, log)
    for p in (a, b):
        assert p.wait(30) == 0
    ev = events(log)
    assert ev[("B", "end")] < ev[("A", "end")]               # the cpu stage ran while the gpu was busy


def test_backfill_when_the_earlier_job_is_blocked_elsewhere(env):
    e, log = env
    x = spawn(e, "gpu", "X", 0, 1.5, log)                    # holds the gpu
    time.sleep(0.4)
    early = spawn(e, "cpu,gpu", "E", 1, 0.2, log)            # first in line, but blocked on the gpu
    time.sleep(0.2)
    late = spawn(e, "cpu", "L", 2, 0.2, log)                 # only needs the cpu: no reason to wait
    for p in (x, early, late):
        assert p.wait(30) == 0
    ev = events(log)
    assert ev[("L", "end")] < ev[("X", "end")]
    assert ev[("E", "start")] >= ev[("X", "end")] - 0.01


def test_dead_holder_does_not_block(env):
    e, log = env
    a = spawn(e, "cpu", "A", 1, 30, log)
    time.sleep(0.5)
    a.kill()
    a.wait()
    t0 = time.time()
    b = spawn(e, "cpu", "B", 2, 0.1, log)
    assert b.wait(30) == 0
    assert time.time() - t0 < 5
    assert ("B", "end") in events(log)


def test_stage_resources():
    from gsturn.config import Config
    from gsturn.scheduler import stage_resources
    cfg = Config(device="cpu", train_backend="brush", feature_backend="superpoint")
    assert stage_resources("sfm", cfg) == ["cpu"]
    assert stage_resources("train", cfg) == ["gpu"]
    assert stage_resources("features", cfg) == ["cpu"]
    cfg2 = Config(device="mps", feature_backend="superpoint")
    assert stage_resources("features", cfg2) == ["cpu", "gpu"]
    assert stage_resources("export", cfg2) == []
