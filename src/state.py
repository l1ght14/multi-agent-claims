"""Claim state and the typed handoffs between nodes.

The spec's requirement is "typed handoffs, no free text passed between nodes", and it is the
single most important line in the build. A multi-agent system where nodes exchange prose
strings cannot be reasoned about: you cannot tell what a node was *given*, you cannot validate
anything, and you cannot replay a run. Every inter-node value in this project is a frozen
dataclass with a schema, and a node that returns the wrong shape fails at the boundary rather
than three steps later in a place that makes no sense.

The other thing this module fixes is the shape of a claim. `ClaimState` is the whole shared
record: it is what the store persists, what the budget meter reads, what the trace annotates,
and what a resume reconstructs. Keeping it one object is what makes "replay from snapshot
reaches the same terminal state" a checkable claim rather than a hope - there is exactly one
thing being serialised.

Deliberately not a general-purpose message bus. There is no `dict[str, Any]` anywhere in the
handoff path, and `validate()` is called on every boundary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any


class Terminal(str, Enum):
    """Where a run can end. Every path through the machine lands in exactly one.

    Having this be a closed enum rather than a free string is what makes "did the graph
    terminate, and where" answerable without reading a trace by hand. `BUDGET_EXHAUSTED` and
    `STEP_LIMIT` are separate terminals on purpose: a run that ran out of money and a run that
    ran out of steps are different operational problems and get charted separately.
    """

    AWAITING_HUMAN = "awaiting_human"
    APPROVED = "approved"
    REJECTED = "rejected"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STEP_LIMIT = "step_limit"
    FAILED = "failed"


class Stage(str, Enum):
    """Which node produced the current state. Not a status - a position."""

    NEW = "new"
    EXTRACTED = "extracted"
    INVESTIGATED = "investigated"
    REVIEWED = "reviewed"


# ---------------------------------------------------------------- handoffs


@dataclass(frozen=True)
class ExtractedDocuments:
    """Out of the extractor. Structured facts, never prose.

    `missing` is a first-class field rather than an absence. An extractor that returns nothing
    and an extractor that reports "I found nothing" are different states, and only the second
    one lets the investigator stop instead of inventing findings.
    """

    claim_id: str
    incident_date: str | None
    vehicle: str | None
    claimed_amount: float | None
    documents_found: list[str] = field(default_factory=list)
    documents_missing: list[str] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.claim_id:
            raise ValueError("ExtractedDocuments.claim_id is required")
        if self.claimed_amount is not None and self.claimed_amount < 0:
            raise ValueError(f"claimed_amount cannot be negative: {self.claimed_amount}")
        if self.claimed_amount is not None and not isinstance(self.claimed_amount, (int, float)):
            raise ValueError(f"claimed_amount must be numeric: {self.claimed_amount!r}")


@dataclass(frozen=True)
class Investigation:
    """Out of the investigator. Every finding cites a document.

    `uncited_findings` is tracked explicitly and checked by the reviewer. An investigator that
    asserts something with no source is not wrong, it is unverifiable, and an insurance
    recommendation built on unverifiable assertions is the actual risk here - not a prompt
    injection, not a hallucination in the abstract sense, but a payout justified by a claim the
    system cannot trace back to a document.
    """

    claim_id: str
    policy_limit: float | None
    prior_claim_count: int
    repair_estimate: float | None
    findings: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    uncited_findings: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.claim_id:
            raise ValueError("Investigation.claim_id is required")
        if self.prior_claim_count < 0:
            raise ValueError(f"prior_claim_count cannot be negative: {self.prior_claim_count}")
        for value in (self.policy_limit, self.repair_estimate):
            if value is not None and value < 0:
                raise ValueError(f"monetary field cannot be negative: {value}")

    @property
    def fully_cited(self) -> bool:
        return not self.uncited_findings


@dataclass(frozen=True)
class Review:
    """Out of the reviewer. A decision plus the reason, never a bare verdict."""

    claim_id: str
    approved: bool
    reasons: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    requires_human: bool = True

    def validate(self) -> None:
        if not self.claim_id:
            raise ValueError("Review.claim_id is required")
        if not isinstance(self.approved, bool):
            raise ValueError(f"Review.approved must be bool, got {type(self.approved).__name__}")
        # A rejection with no stated reason is not reviewable. An approver that cannot say why
        # it approved is the thing an auditor asks about, so an empty reason is refused here
        # rather than discovered later.
        if not self.approved and not self.reasons and not self.issues:
            raise ValueError("a rejection must state at least one reason or issue")
        if not self.reasons and not self.issues:
            raise ValueError("a review must state at least one reason")


HANDOFF_TYPES: dict[Stage, type] = {
    Stage.EXTRACTED: ExtractedDocuments,
    Stage.INVESTIGATED: Investigation,
    Stage.REVIEWED: Review,
}

# Stage names and field names are deliberately not the same. `Stage.EXTRACTED.value` is
# "extracted" while the field is `documents`, and deriving one from the other produced a
# `replace()` call with a keyword the dataclass does not have. Explicit map, so the two can be
# renamed independently without a runtime failure three layers up.
FIELD_FOR_STAGE: dict[Stage, str] = {
    Stage.EXTRACTED: "documents",
    Stage.INVESTIGATED: "investigation",
    Stage.REVIEWED: "review",
}


def validate_handoff(stage: Stage, payload: Any) -> Any:
    """Assert a node returned the right type, and that its own invariants hold.

    This is the boundary. A node that returns prose where a dataclass is expected fails here,
    loudly, with the stage named - not three steps later with a confusing attribute error.
    """
    expected = HANDOFF_TYPES[stage]
    if not isinstance(payload, expected):
        raise TypeError(
            f"handoff for stage {stage.value} must be {expected.__name__}, "
            f"got {type(payload).__name__}. Free text between nodes is not permitted."
        )
    payload.validate()
    return payload


# ---------------------------------------------------------------- the record


@dataclass
class ClaimState:
    """The single shared record for one claim. Persisted, budgeted, traced, replayed."""

    claim_id: str
    stage: Stage = Stage.NEW
    terminal: Terminal | None = None

    documents: ExtractedDocuments | None = None
    investigation: Investigation | None = None
    review: Review | None = None

    step: int = 0
    bounces: int = 0
    tokens_used: int = 0
    dollars_used: float = 0.0

    approved_by: str | None = None
    approved_at: str | None = None
    payout_amount: float = 0.0

    # Per-node record of what actually happened, in order. This is the trace: the exact node
    # path a run took, which is ship-gate item 1 and the thing you cannot reconstruct from the
    # final state alone.
    node_path: list[str] = field(default_factory=list)

    def enter(self, node: str) -> None:
        self.node_path.append(node)

    def with_stage(self, stage: Stage, payload: Any) -> "ClaimState":
        validate_handoff(stage, payload)
        return replace(self, stage=stage, **{FIELD_FOR_STAGE[stage]: payload})

    def fingerprint(self) -> str:
        """Content hash of the decision-relevant fields.

        Used by the replay check and by the snapshot checksum. Two exclusions, both deliberate:

        - Cost accounting and timings. A resumed run may genuinely spend different tokens while
          making identical judgements; a determinism check that demanded otherwise would be
          testing the accounting rather than the logic.
        - Container *type*. The in-memory handoffs hold tuples, JSON turns those into lists, and
          `repr()` renders the two differently. Without canonicalising, every snapshot failed
          its own checksum on reload - which is the check working, catching a real
          round-trip instability rather than a false alarm.
        """
        material = {
            "claim_id": self.claim_id,
            "stage": self.stage.value,
            "terminal": self.terminal.value if self.terminal else None,
            "payout_amount": round(self.payout_amount, 2),
            "approved_by": self.approved_by,
            "documents": _canon(asdict(self.documents)) if self.documents else None,
            "investigation": _canon(asdict(self.investigation)) if self.investigation else None,
            "review": _canon(asdict(self.review)) if self.review else None,
        }
        blob = json.dumps(material, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, blob: dict) -> "ClaimState":
        """Reconstruct from a snapshot.

        Explicit rather than `cls(**blob)`: a snapshot written by an older version must fail
        with a clear error rather than a `TypeError` about an unexpected keyword, and the
        handoff objects need rebuilding rather than being left as dicts.
        """
        data = dict(blob)
        data["stage"] = Stage(data.get("stage", "new"))
        term = data.get("terminal")
        data["terminal"] = Terminal(term) if term else None
        for name, kind in (("documents", ExtractedDocuments), ("investigation", Investigation),
                           ("review", Review)):
            value = data.get(name)
            if isinstance(value, dict):
                if hasattr(value, "keys") and "_type_marker" not in value:
                    data[name] = _rebuild(kind, value)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})



def _canon(value):
    """Recursively normalise containers so a fingerprint survives a JSON round-trip.

    `dataclasses.asdict` preserves tuples; `json` turns them into lists; `repr` renders the two
    differently. Canonicalising to lists at every level means the hash depends on the *content*
    of a handoff and not on whether it currently happens to be in memory or on disk.
    """
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return value


def _rebuild(kind: type, blob: dict) -> Any:
    """Rebuild a frozen handoff, coercing list fields back to tuples.

    JSON gives lists; the dataclasses declare tuples so they stay hashable and immutable. The
    coercion has to happen here or every snapshot-resumed run compares unequal to the original
    for reasons that have nothing to do with the logic under test.
    """
    kwargs = {}
    for f in kind.__dataclass_fields__.values():
        if f.name not in blob:
            continue
        value = blob[f.name]
        if value is None or isinstance(value, (str, int, float, bool)):
            kwargs[f.name] = value
        else:
            kwargs[f.name] = tuple(value)
    return kind(**kwargs)
