"""Shared Colab conventions for the revision-robustness notebooks.

Every notebook does, in order:
  1. mount_drive()            - mounts Google Drive at /content/drive
  2. setup_workspace()        - unzips DRIVE/colab_bundle.zip to /content/work
                                (LOCAL disk; never run pandas/torch against
                                Drive paths), pip-installs requirements,
                                pulls any newer results from Drive
  3. verify_frozen_hashes()   - the five modeling tables vs data/SHA256SUMS,
                                before any compute
  4. export_api_key()         - reads DRIVE/.openalex_key into the
                                OPENALEX_API_KEY env var (stripped, asserted
                                non-empty, never printed); only the fetch and
                                repair paths consume it
  5. start_checkpoint_thread()- rsyncs /content/work/results/robustness/ to
                                Drive every CHECKPOINT_MINUTES; notebooks also
                                call sync_to_drive() after each completed unit
  6. ... job cells ...
  7. write_done_flag(job)     - DONE_<job>.flag on Drive + final sync

Resume contract: a rerun of any notebook syncs Drive -> local first and the
job scripts skip completed units (r1/r6 skip existing partial JSONs, r5 skips
cached authors, r3 skips existing per-config JSONs), so a runtime disconnect
costs at most one unit of work.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/who-inherits")
WORK = Path("/content/work")
BUNDLE = DRIVE / "colab_bundle.zip"
RESULTS = WORK / "results" / "robustness"
DRIVE_RESULTS = DRIVE / "results" / "robustness"
CHECKPOINT_MINUTES = 10


def mount_drive():
    from google.colab import drive as gdrive
    gdrive.mount("/content/drive")
    assert DRIVE.exists(), f"{DRIVE} not found - upload the bundle first"


def setup_workspace(install: bool = True):
    if not (WORK / "data" / "SHA256SUMS").exists():
        WORK.mkdir(parents=True, exist_ok=True)
        print("[setup] unzipping bundle to local disk ...")
        subprocess.run(["unzip", "-q", "-o", str(BUNDLE), "-d", str(WORK)],
                       check=True)
    if install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(WORK / "requirements-colab.txt")], check=True)
    pull_from_drive()
    print("[setup] workspace ready at", WORK)


def pull_from_drive():
    """Drive -> local: pick up results a previous (crashed) session synced."""
    if DRIVE_RESULTS.exists():
        subprocess.run(["rsync", "-a", str(DRIVE_RESULTS) + "/",
                        str(RESULTS) + "/"], check=True)
        print("[sync] pulled results from Drive")


def sync_to_drive():
    """Local -> Drive: called after every completed unit and by the
    checkpoint thread."""
    DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", str(RESULTS) + "/",
                    str(DRIVE_RESULTS) + "/"], check=True)
    print(f"[sync] pushed results to Drive at {time.strftime('%H:%M:%S')}")


_stop_ckpt = threading.Event()


def start_checkpoint_thread():
    def loop():
        while not _stop_ckpt.wait(CHECKPOINT_MINUTES * 60):
            try:
                sync_to_drive()
            except Exception as e:                     # keep the job alive
                print(f"[sync] checkpoint failed: {e}")
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"[sync] checkpoint thread running every {CHECKPOINT_MINUTES} min")
    return t


def stop_checkpoint_thread():
    _stop_ckpt.set()


def verify_frozen_hashes():
    sums = {}
    for line in open(WORK / "data" / "SHA256SUMS"):
        h, fn = line.split()
        sums[fn.lstrip("*").strip()] = h.strip()
    for f in ["econ", "math", "neuro", "physics", "chemistry"]:
        fn = f"clean_dataset_{f}.parquet"
        got = hashlib.sha256((WORK / "data" / fn).read_bytes()).hexdigest()
        assert got == sums[fn], f"hash mismatch for {fn} - corrupted bundle"
    print("[verify] all five frozen-table hashes match SHA256SUMS")


def export_api_key():
    """DRIVE/.openalex_key -> OPENALEX_API_KEY env. Never printed, never
    written under WORK (the bundle and the synced results stay key-free)."""
    p = DRIVE / ".openalex_key"
    assert p.exists(), f"place the key at {p}"
    key = p.read_text().strip()
    assert key, ".openalex_key is empty"
    os.environ["OPENALEX_API_KEY"] = key
    print("[key] OPENALEX_API_KEY set from Drive (length "
          f"{len(key)}, value not shown)")


def run_meta() -> dict:
    """device + versions + bundle commit, attached to result files written
    at the notebook level."""
    import numpy, pandas, scipy, sklearn
    meta = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__, "pandas": pandas.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "bundle_git_commit": (WORK / "GIT_COMMIT.txt").read_text().strip()
        if (WORK / "GIT_COMMIT.txt").exists() else "unknown",
    }
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["device_name"] = torch.cuda.get_device_name(0)
        import torch_geometric
        meta["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        pass
    return meta


def write_done_flag(job: str):
    payload = json.dumps({"job": job, "finished_utc":
                          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          **run_meta()}, indent=2)
    # one copy inside the synced results dir (travels with the merge-back)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"DONE_{job}.flag").write_text(payload)
    sync_to_drive()
    # and one at the Drive root for at-a-glance checking
    (DRIVE / f"DONE_{job}.flag").write_text(payload)
    print(f"[done] DONE_{job}.flag written (Drive root + results/robustness)")


def run_script(rel_script: str, env_extra: dict | None = None, args=()):
    """Run one of the repo's real scripts as a subprocess from WORK, with the
    per-discipline DATASET/DATASET_PATH convention. Streams output."""
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = [sys.executable, str(WORK / rel_script), *args]
    print("[run]", " ".join(cmd), {k: v for k, v in (env_extra or {}).items()
                                   if "KEY" not in k})
    r = subprocess.run(cmd, cwd=str(WORK), env=env)
    if r.returncode:
        raise RuntimeError(f"{rel_script} exited {r.returncode}")
