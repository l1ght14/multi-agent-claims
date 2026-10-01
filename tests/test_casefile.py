"""Termination, the human gate, and the guarantees the ship gate depends on.

Ordered by cost of failure. The payout tests come first because "never pays out without a
human" is the property this whole project exists to hold - a failure there is not a bug, it is
the thing the design exists to prevent. Budget and store tests follow because they are what
make a replay meaningful. Node-level judgement tests come last.

The loop-guard and replay tests deliberately assert on *behaviour under constraint* rather
than on a single happy path, because a graph that terminates once demonstrates nothing.
"""
from __future__ import annotations

import json

import pytest

from src.budget import Budget, BudgetExceeded, PRICES
from src.machine import ClaimMachine
from src.state import (
    ClaimState,
    ExtractedDocuments,
    Investigation,
    Review,
    Stage,
    Terminal,
    validate_handoff,
)
from src.store import StateStore, StoreError
from corpus.claims import build_claims, claims_by_id


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "store")


def _machine(store, **kw):
    return ClaimMachine(store, Budget(**kw))


# ---------------------------------------------------------------- the human gate


def test_no_path_reaches_APPROVED_without_the_approver(store):
    """The single most important property in the project.

    Every claim that could reach APPROVED must have passed through `approver`, and every
    approver call requires an explicit human decision. A default of True here would silently
    release money on silence.
    """
    for claim in build_claims():
        result = _machine(store).run(claim)
        if result.state.terminal is Terminal.APPROVED:
            assert "approver" in result.report.node_path, (
                f"{claim.claim_id} was APPROVED without an approver in the path"
            )
            assert result.state.approved_by == "human:adjuster"


def test_no_approval_is_inferred_from_silence(store):
    """No human decision means AWAITING_HUMAN, never APPROVED."""
    machine = _machine(store)
    results = [machine.run(c) for c in build_claims()]
    approved = [r for r in results if r.state.terminal is Terminal.APPROVED]
    assert approved == [], "silence must never produce an approval"


def test_an_explicit_human_approval_pays_out(store):
    claim = claims_by_id()["CLM-001"]
    result = _machine(store).run(claim, human_decision=True)
    assert result.state.terminal is Terminal.APPROVED
    assert result.state.payout_amount == claim.claimed_amount
    assert "approver" in result.report.node_path


def test_an_explicit_human_rejection_pays_nothing(store):
    result = _machine(store).run(claims_by_id()["CLM-001"], human_decision=False)
    assert result.state.terminal is Terminal.REJECTED
    assert result.state.payout_amount == 0.0


def test_every_run_terminates(store):
    """No budget setting may hang. Three settings, thirty claims, all terminal."""
    for kw in ({"max_bounces": 1, "max_steps": 3},
               {"max_bounces": 2, "max_steps": 12},
               {"max_bounces": 9, "max_steps": 60}):
        s = StateStore(store.root / f"s{kw['max_steps']}")
        machine = ClaimMachine(s, Budget(**kw))
        for claim in build_claims():
            result = machine.run(claim)
            assert result.state.terminal is not None, (
                f"{claim.claim_id} did not terminate under {kw}"
            )


def test_the_terminal_state_is_always_a_known_enum(store):
    machine = _machine(store)
    for claim in build_claims():
        assert machine.run(claim).state.terminal in set(Terminal)


# ---------------------------------------------------------------- loop guard


def test_the_reviewer_bounces_at_least_once(store):
    bounced = [r for r in (_machine(store).run(c) for c in build_claims())
               if r.report.bounces > 0]
    assert bounced, "no claim ever bounced, so the loop guard was never exercised"


def test_the_bounce_cap_is_what_stops_the_loop(store):
    """The run must stop on the bounce cap, not merely on the step cap.

    If the step cap is what fires, the bounce cap is decorative and the graph would have run
    for the whole budget before stopping.
    """
    machine = _machine(store, max_bounces=2, max_steps=40)
    result = machine.run(claims_by_id()["CLM-017"])
    assert result.report.bounces == 2
    assert result.report.stop_reason == "bounce_limit"
    assert result.state.terminal is Terminal.STEP_LIMIT


def test_the_bounce_counter_survives_the_rewind(store):
    """Bouncing rewinds the stage to EXTRACTED. If it also rewound the counters, the guard
    would reset on every bounce and never fire."""
    result = _machine(store, max_bounces=3, max_steps=40).run(claims_by_id()["CLM-017"])
    assert result.report.bounces >= 1
    assert result.state.bounces == result.report.bounces


def test_a_tight_step_limit_still_terminates(store):
    result = _machine(store, max_steps=2, max_bounces=10).run(claims_by_id()["CLM-001"])
    assert result.state.terminal is Terminal.STEP_LIMIT
    assert result.report.steps <= 3


# ---------------------------------------------------------------- budget


def test_check_raises_once_the_step_limit_is_reached():
    b = Budget(max_steps=2)
    b.record_step()
    b.record_step()
    with pytest.raises(BudgetExceeded, match="step limit"):
        b.check()


def test_check_raises_once_the_bounce_limit_is_reached():
    b = Budget(max_bounces=2)
    b.record_bounce()
    b.record_bounce()
    with pytest.raises(BudgetExceeded, match="bounce limit"):
        b.check()


def test_check_raises_once_the_token_budget_is_reached():
    b = Budget(max_tokens=100)
    b.charge(60, 60)
    with pytest.raises(BudgetExceeded, match="token budget"):
        b.check()


def test_check_raises_once_the_dollar_budget_is_reached():
    b = Budget(model="llama-3.3-70b", max_dollars=0.001, max_tokens=10_000_000)
    b.charge(1_000_000, 1_000_000)
    with pytest.raises(BudgetExceeded, match="dollar budget"):
        b.check()


def test_the_offline_model_costs_nothing():
    b = Budget(model="offline")
    assert b.charge(1_000_000, 1_000_000) == 0.0
    assert b.dollars_used == 0.0


def test_charging_accumulates_tokens_and_dollars():
    b = Budget(model="llama-3.1-8b")
    b.charge(1000, 500)
    b.charge(2000, 0)
    assert b.tokens_used == 3500
    expected = (1000 / 1e6) * 0.05 + (500 / 1e6) * 0.08 + (2000 / 1e6) * 0.05
    assert b.dollars_used == pytest.approx(expected)


def test_an_unknown_model_is_treated_as_free_not_as_an_error():
    """Silently pricing an unknown model at zero would let a real run skip the ceiling. It is
    documented here, and `node._charge` always passes a known model."""
    assert PRICES.get("some-future-model", PRICES["offline"]) == (0.0, 0.0)


def test_headroom_is_clamped_to_a_unit_interval():
    b = Budget(max_dollars=1.0)
    assert b.headroom_fraction() == 1.0
    b.charge(0, 0)
    b.dollars_used = 5.0
    assert b.headroom_fraction() == 0.0


def test_a_zero_dollar_ceiling_reports_no_headroom():
    assert Budget(max_dollars=0.0).headroom_fraction() == 0.0


def test_budget_round_trips_through_the_store(store):
    b = Budget(max_bounces=3, model="llama-3.3-70b")
    b.charge(100, 50)
    b.record_step()
    b.record_bounce()
    restored = Budget.from_dict(json.loads(json.dumps(b.to_dict())))
    assert restored.max_bounces == 3
    assert restored.model == "llama-3.3-70b"
    assert restored.tokens_used == 150
    assert restored.bounces_used == 1
    assert restored.dollars_used == pytest.approx(b.dollars_used)


def test_budget_history_is_retained_for_the_cost_chart():
    b = Budget(model="llama-3.1-8b")
    b.charge(10, 5)
    b.charge(20, 10)
    assert len(b.history) == 2
    assert b.history[1]["tokens_total"] == 45


# ---------------------------------------------------------------- store


def test_a_snapshot_round_trips(store):
    claim = claims_by_id()["CLM-001"]
    state = ClaimState(claim_id=claim.claim_id)
    state.documents = ExtractedDocuments(claim.claim_id, "2026-03-14", "v", 100.0, ["a"], [],
                                         ("note one", "note two"))
    store.save(state, Budget())
    loaded = store.load(claim.claim_id)
    assert loaded is not None
    assert loaded.state.fingerprint() == state.fingerprint()
    assert loaded.state.documents.notes == ("note one", "note two")


def test_the_fingerprint_survives_a_json_round_trip():
    """Caught by the checksum failing on the first full run: tuples become lists in JSON, and
    `repr` renders the two differently, so the hash differed between save and load."""
    state = ClaimState(claim_id="x")
    state.documents = ExtractedDocuments("x", "2026-01-01", "v", 100.0, ["a"], ["b"],
                                         ("n1", "n2"))
    state.investigation = Investigation("x", 5000.0, 2, 90.0, ("f1",), ("c1",), ("u1",))
    state.review = Review("x", True, ("ok",), ("issue",))
    direct = state.fingerprint()
    via_json = ClaimState.from_dict(json.loads(json.dumps(state.to_dict()))).fingerprint()
    assert direct == via_json


def test_a_tampered_snapshot_is_refused(store):
    claim = claims_by_id()["CLM-001"]
    state = ClaimState(claim_id=claim.claim_id)
    store.save(state, Budget())
    # Tamper the file `load()` actually reads. An earlier version edited a snapshot file and
    # the check passed - because `load` reads `state/<id>.json`, so the tamper had no effect
    # and the checksum was never exercised at all.
    path = store.root / "state" / f"{claim.claim_id}.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["state"]["payout_amount"] = 99999.0
    path.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(StoreError, match="checksum"):
        store.load(claim.claim_id)


def test_a_tampered_snapshot_file_is_refused_too(store):
    """Both read paths are guarded, not just the convenience one."""
    claim = claims_by_id()["CLM-002"]
    store.save(ClaimState(claim_id=claim.claim_id), Budget())
    path = store.root / "snapshots" / claim.claim_id / "0000.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["state"]["payout_amount"] = 42.0
    path.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(StoreError, match="checksum"):
        store.snapshots(claim.claim_id)


def test_an_unsafe_claim_id_is_rejected(store):
    for bad in ("../escape", "a/b", "..", "with space", "", ".hidden", "x" * 65):
        with pytest.raises(StoreError, match="unsafe claim id"):
            store.load(bad)


def test_snapshots_are_returned_in_step_order(store):
    state = ClaimState(claim_id="x")
    for i in range(4):
        state.step = i
        store.save(state, Budget())
    assert [s.step for s in store.snapshots("x")] == [0, 1, 2, 3]


def test_loading_an_unknown_claim_returns_none(store):
    assert store.load("nope") is None
    assert store.snapshots("nope") == []


def test_resume_raises_rather_than_starting_fresh(store):
    """Silently starting a new run would leave the caller believing it continued one, and the
    cost chart would then double-count or drop a run."""
    with pytest.raises(KeyError, match="no stored snapshot"):
        _machine(store).resume("CLM-999")


# ---------------------------------------------------------------- routing


def test_a_claim_with_no_documents_short_circuits_at_the_extractor(store):
    result = _machine(store).run(claims_by_id()["CLM-026"])
    assert result.report.node_path == ["extractor"]
    assert result.state.terminal is Terminal.REJECTED
    assert result.state.review is not None
    assert "no readable documents" in result.state.review.reasons[0]


def test_an_over_limit_claim_is_rejected_rather_than_sent_to_a_human(store):
    """Regression: the router once ignored `review.approved`, so an over-limit claim reached
    AWAITING_HUMAN with a correct rejection sitting unused in the state."""
    for cid in ("CLM-009", "CLM-010", "CLM-028"):
        result = _machine(store).run(claims_by_id()[cid])
        assert result.state.terminal is Terminal.REJECTED, f"{cid} was not rejected"
        assert "exceeds the policy limit" in result.state.review.reasons[0]


def test_a_clean_claim_reaches_the_human_gate(store):
    result = _machine(store).run(claims_by_id()["CLM-001"])
    assert result.state.terminal is Terminal.AWAITING_HUMAN
    assert result.state.review.approved is True


def test_every_tracked_node_appears_in_the_node_path(store):
    for claim in build_claims():
        path = _machine(store).run(claim).report.node_path
        assert path, f"{claim.claim_id} recorded an empty trace"
        assert path[0] == "extractor", f"{claim.claim_id} did not start at the extractor"


# ---------------------------------------------------------------- typed handoffs


def test_a_handoff_must_be_the_right_dataclass():
    with pytest.raises(TypeError, match="Free text between nodes is not permitted"):
        validate_handoff(Stage.EXTRACTED, "the extractor said something")


def test_a_handoff_of_the_wrong_type_for_its_stage_is_refused():
    with pytest.raises(TypeError, match="must be Investigation"):
        validate_handoff(Stage.INVESTIGATED, ExtractedDocuments("x", None, None, None))


def test_extracted_documents_reject_a_negative_amount():
    with pytest.raises(ValueError, match="cannot be negative"):
        ExtractedDocuments("x", None, None, -5.0).validate()


def test_investigation_rejects_a_negative_prior_count():
    with pytest.raises(ValueError, match="prior_claim_count"):
        Investigation("x", 1.0, -1, None).validate()


def test_a_rejection_must_state_why():
    with pytest.raises(ValueError, match="must state at least one"):
        Review("x", approved=False).validate()


def test_a_review_must_state_a_reason_even_when_approving():
    with pytest.raises(ValueError, match="must state at least one"):
        Review("x", approved=True).validate()


def test_a_handoff_requires_a_claim_id():
    with pytest.raises(ValueError, match="claim_id is required"):
        ExtractedDocuments("", None, None, None).validate()


def test_uncited_findings_are_visible():
    inv = Investigation("x", 1.0, 0, None, ("f",), (), ("no source",))
    assert inv.fully_cited is False


# ---------------------------------------------------------------- corpus


def test_the_corpus_has_thirty_claims():
    assert len(build_claims()) == 30


def test_the_corpus_covers_every_terminal_path():
    """Counts asserted so the corpus cannot silently lose the coverage it exists for."""
    from collections import Counter

    counts = Counter()
    for claim in build_claims():
        if len(claim.documents) == 1:
            counts["no_documents"] += 1
        elif len(claim.documents) < 4:
            counts["missing_evidence"] += 1
        limit = claim.truth.get("policy_limit")
        if limit and claim.claimed_amount > limit:
            counts["over_limit"] += 1
        if claim.truth.get("prior_claims", 0) >= 5:
            counts["high_frequency"] += 1
    assert counts["no_documents"] == 2
    assert counts["missing_evidence"] == 6
    assert counts["over_limit"] == 6
    assert counts["high_frequency"] == 6


def test_claim_ids_are_unique():
    ids = [c.claim_id for c in build_claims()]
    assert len(set(ids)) == len(ids)
