"""The external state store: snapshots, resume, and the run archive.

The spec requires "shared claim state in an external store so any run can be resumed from a
snapshot", and the ship gate requires one run replayed from a stored snapshot to reach the same
terminal state. Both are only meaningful if the state genuinely lives outside the process -
otherwise "resuming" is just calling a function with its own arguments and the guarantee is
vacuous.

So this is a real store on disk, keyed by claim id, holding:
  - `state/<claim_id>.json`   the latest snapshot
  - `snapshots/<claim_id>/<n>.json`  every step, so a run can be replayed from any point
  - `runs/<run_id>.json`      the completed-run archive the evaluation reads

The file backend is chosen over Redis or Postgres deliberately. This project is about
termination and replay semantics, and a networked store would add a dependency, a failure mode
and a startup step without changing a single one of those semantics. Swapping in Redis means
reimplementing `put`/`get`/`list_snapshots`; nothing above this layer changes. That trade is
recorded in the README rather than hidden.

Atomic writes matter more than they look. A snapshot written by a run that is killed mid-write
is corrupt, and a corrupt snapshot makes the replay guarantee untestable. So every write goes
to a temp file and is renamed, and `load` verifies a checksum rather than trusting the JSON.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .budget import Budget
from .state import ClaimState

# A claim id becomes a filename, so the allowed set is narrow and explicit. Note the explicit
# exclusion of "..": a character class of [A-Za-z0-9_.-] accepts it, and ".." is the one shape
# in that class that means "parent directory". The `.json` suffix appended afterwards stops it
# from escaping on its own, but a guard that has to be reasoned about is a guard that will
# eventually be wrong, so it is rejected outright.
_SAFE = re.compile(r"^(?!\.\.?$)[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class StoreError(Exception):
    pass


@dataclass
class Snapshot:
    claim_id: str
    step: int
    state: ClaimState
    budget: Budget
    checksum: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id, "step": self.step,
            "state": self.state.to_dict(), "budget": self.budget.to_dict(),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "Snapshot":
        state = ClaimState.from_dict(blob["state"])
        budget = Budget.from_dict(blob["budget"])
        snap = cls(claim_id=blob["claim_id"], step=blob["step"], state=state,
                   budget=budget, checksum=blob.get("checksum", ""))
        if snap.checksum and snap.compute_checksum() != snap.checksum:
            raise StoreError(
                f"snapshot for {snap.claim_id} step {snap.step} failed its checksum - the "
                f"file was truncated or edited. Refusing to resume from a state I cannot trust."
            )
        return snap

    def compute_checksum(self) -> str:
        """Hash the decision-relevant content, not the whole file.

        Cost counters and timestamps are excluded for the same reason they are excluded from
        `ClaimState.fingerprint`: a resume that made identical decisions with different
        accounting is a correct resume, and a checksum that demanded otherwise would reject
        every legitimate replay.
        """
        material = json.dumps(
            {"state": self.state.fingerprint(), "step": self.step,
             "terminal": self.state.terminal.value if self.state.terminal else None},
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / "state").mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- paths

    def _validate(self, claim_id: str) -> str:
        """Reject ids that could escape the store directory.

        Claim ids come from the corpus, which is nominally trusted, but a path built from an
        unchecked string is the kind of thing that becomes a vulnerability the moment the ids
        come from a webhook instead of a file. One regex, checked once, at the boundary.
        """
        if not _SAFE.match(claim_id):
            raise StoreError(f"unsafe claim id {claim_id!r}")
        return claim_id

    def _state_path(self, claim_id: str) -> Path:
        return self.root / "state" / f"{self._validate(claim_id)}.json"

    def _snap_dir(self, claim_id: str) -> Path:
        return self.root / "snapshots" / self._validate(claim_id)

    # ---------------------------------------------------------------- writes

    def save(self, state: ClaimState, budget: Budget) -> Snapshot:
        """Write the current snapshot and the latest-state file, atomically."""
        snap = Snapshot(claim_id=state.claim_id, step=state.step, state=state, budget=budget,
                        checksum="")
        snap.checksum = snap.compute_checksum()
        blob = snap.to_dict()

        snap_dir = self._snap_dir(state.claim_id)
        snap_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(snap_dir / f"{state.step:04d}.json", json.dumps(blob, indent=2))

        # Latest state carries the highest step, so resuming is a single read. A run that
        # resumes and is then snapshotted at a lower step than a previous attempt would
        # otherwise make "latest" ambiguous.
        latest = self._latest_step(state.claim_id)
        if state.step >= latest:
            _atomic_write(self._state_path(state.claim_id), json.dumps(blob, indent=2))
        return snap

    def archive_run(self, run_id: str, payload: dict) -> Path:
        path = self.root / "runs" / f"{self._validate(run_id)}.json"
        _atomic_write(path, json.dumps(payload, indent=2, default=str))
        return path

    # ---------------------------------------------------------------- reads

    def load(self, claim_id: str) -> Snapshot | None:
        path = self._state_path(claim_id)
        if not path.exists():
            return None
        return Snapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_step(self, claim_id: str, step: int) -> Snapshot | None:
        path = self._snap_dir(claim_id) / f"{step:04d}.json"
        if not path.exists():
            return None
        return Snapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _latest_step(self, claim_id: str) -> int:
        steps = [int(p.stem) for p in self._snap_dir(claim_id).glob("*.json")] \
            if self._snap_dir(claim_id).exists() else []
        return max(steps) if steps else -1

    def snapshots(self, claim_id: str) -> list[Snapshot]:
        """Every snapshot for a claim, in step order."""
        d = self._snap_dir(claim_id)
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.json")):
            out.append(Snapshot.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        return out

    def iter_runs(self) -> Iterator[dict]:
        for p in sorted((self.root / "runs").glob("*.json")):
            yield json.loads(p.read_text(encoding="utf-8"))

    def clear_claim(self, claim_id: str) -> None:
        self._state_path(claim_id).unlink(missing_ok=True)
        d = self._snap_dir(claim_id)
        if d.exists():
            for p in d.glob("*.json"):
                p.unlink()
            d.rmdir()


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename.

    The temp name keeps the `.json` suffix because `json.dump` does not add one, but the point
    is that `Path.replace` is atomic on both Windows and POSIX - a reader sees either the old
    file or the complete new one, never a half-written one. This project pushes to Windows
    routinely and a snapshot truncated by a crash is exactly the failure the replay guarantee
    must not have.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
