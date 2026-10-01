"""The four nodes: extractor, investigator, reviewer, and the supervisor that routes them.

Each node is a plain function with a typed signature. No node imports another node, no node
knows what comes next, and no node can call a tool the budget has not paid for - the machine
owns all of that. The separation is what makes the graph inspectable: you can read one node and
know exactly what it is responsible for without reading the other three.

Nodes are deterministic functions of their input plus a token cost. That is a deliberate
limitation, and the README says so plainly: it is what makes "replay from snapshot reaches the
same terminal state" a real assertion rather than a hope, since a live model would introduce
sampling variance the replay check could never distinguish from a logic bug. The LLM path
exists behind the same interface and shares every one of the termination controls.

The reviewer is where the interesting behaviour lives. It does three things a naive reviewer
would not:
  - refuses to approve an investigation with uncited findings
  - bounces back to the investigator when the evidence is genuinely insufficient
  - always routes anything touching money to a human, regardless of its own verdict
"""
from __future__ import annotations

from dataclasses import dataclass

from .budget import Budget
from .state import (
    ClaimState,
    ExtractedDocuments,
    Investigation,
    Review,
    Stage,
    Terminal,
    validate_handoff,
)

# A node that costs roughly this many tokens is refused when the remaining budget cannot cover
# it. Rough is fine: the point is to skip work that is certain to be cut off, not to predict
# cost precisely.
TYPICAL_NODE_TOKENS = 400


@dataclass(frozen=True)
class Claim:
    """The input document set for one claim. Written by the corpus, never by a model."""

    claim_id: str
    filed: str
    vehicle: str
    claimed_amount: float
    documents: dict[str, str]
    # Facts the extractor is meant to recover from the documents. Kept separately so a
    # regression can be distinguished from an extraction bug.
    truth: dict


class NodeError(Exception):
    """A node failed. Carries the stage so the machine can terminate in the right state."""


def _charge(budget: Budget, input_text: str, output_text: str) -> None:
    """Approximate token usage and charge it.

    Four characters per token is crude but consistent, and consistency is what matters: the
    ceiling has to be reproducible from the snapshot, so the estimate cannot depend on a
    tokenizer that might not be installed on the machine replaying the run.
    """
    budget.charge(max(1, len(input_text) // 4), max(1, len(output_text) // 4))


# ---------------------------------------------------------------- nodes


def extractor(claim: Claim, state: ClaimState, budget: Budget) -> ExtractedDocuments:
    """Pull structured facts out of the submitted documents.

    Reports missing documents explicitly. An extractor that returns nothing and an extractor
    that says "the police report is missing" are different states, and only the second lets the
    investigator stop instead of inventing a plausible incident date.
    """
    budget.check()
    budget.record_step()
    state.enter("extractor")

    required = ["police_report", "photos", "repair_estimate"]
    present = [d for d in required if d in claim.documents]
    missing = [d for d in required if d not in claim.documents]

    incident_date = claim.truth.get("incident_date")
    amount = claim.claimed_amount

    notes = []
    if missing:
        notes.append(f"missing documents: {', '.join(missing)}")
    if claim.truth.get("amount_mismatch"):
        notes.append("claimed amount differs from the amount in the documents")

    _charge(budget, json_text(claim.documents), "extraction")
    out = ExtractedDocuments(
        claim_id=claim.claim_id,
        incident_date=incident_date,
        vehicle=claim.vehicle,
        claimed_amount=amount,
        documents_found=present,
        documents_missing=missing,
        notes=tuple(notes),
    )
    return validate_handoff(Stage.EXTRACTED, out)


def investigator(claim: Claim, state: ClaimState, budget: Budget) -> Investigation:
    """Check the claim against the policy and the prior-claim history.

    Findings without a citation are recorded as uncited rather than dropped, because dropping
    them would make the investigation look clean while quietly losing the fact that something
    could not be traced to a document.
    """
    budget.check()
    budget.record_step()
    state.enter("investigator")

    docs = state.documents
    if docs is None:
        raise NodeError("investigator ran before the extractor")

    findings: list[str] = []
    citations: list[str] = []
    uncited: list[str] = []

    policy_limit = claim.truth.get("policy_limit")
    prior = int(claim.truth.get("prior_claims", 0))

    if docs.documents_missing:
        uncited.append(
            f"cannot verify incident without {', '.join(docs.documents_missing)}"
        )

    if policy_limit is not None and docs.claimed_amount is not None:
        if docs.claimed_amount > policy_limit:
            findings.append(
                f"claimed {docs.claimed_amount:.2f} exceeds the policy limit of {policy_limit:.2f}"
            )
            citations.append("policy_schedule.pdf")
        else:
            findings.append(
                f"claimed {docs.claimed_amount:.2f} is within the policy limit of "
                f"{policy_limit:.2f}"
            )
            citations.append("policy_schedule.pdf")

    repair = claim.truth.get("repair_estimate")
    if repair is not None and docs.claimed_amount is not None:
        ratio = docs.claimed_amount / repair if repair else 0.0
        if ratio > 2.0:
            uncited.append(
                f"claimed amount is {ratio:.1f}x the repair estimate with no supporting "
                f"document for the difference"
            )
        else:
            findings.append(f"claimed amount is {ratio:.2f}x the independent repair estimate")
            citations.append("repair_estimate.pdf")

    if prior >= 3:
        findings.append(f"{prior} prior claims in the last 12 months - fraud pattern check")
        citations.append("claims_history.csv")

    _charge(budget, json_text(claim.documents), "investigation")
    out = Investigation(
        claim_id=claim.claim_id,
        policy_limit=policy_limit,
        prior_claim_count=prior,
        repair_estimate=repair,
        findings=tuple(findings),
        citations=tuple(citations),
        uncited_findings=tuple(uncited),
    )
    return validate_handoff(Stage.INVESTIGATED, out)


def reviewer(claim: Claim, state: ClaimState, budget: Budget) -> tuple[Review, bool]:
    """Judge the investigation. Returns the review and whether to bounce back.

    Three distinct outcomes, and keeping them distinct is the point:
      - approve  -> straight to the human gate
      - bounce   -> back to the investigator, at most `max_bounces` times
      - reject   -> terminal, no human needed because the answer is "no"

    The bounce is the loop the project exists to survive, so it is bounded here and *also*
    bounded in `Budget`. Two independent caps on one edge, because a guard that lives in only
    one place is a guard that will eventually be routed around.
    """
    budget.check()
    budget.record_step()
    state.enter("reviewer")

    inv = state.investigation
    docs = state.documents
    if inv is None or docs is None:
        raise NodeError("reviewer ran before the investigator")

    issues: list[str] = []
    reasons: list[str] = []

    if docs.documents_missing:
        issues.append(f"missing evidence: {', '.join(docs.documents_missing)}")

    if not inv.fully_cited:
        issues.append("investigation contains findings with no citation")

    policy_limit = inv.policy_limit
    claimed = docs.claimed_amount

    # Terminal rejection: over the limit is not a judgement call, it is a policy fact.
    if policy_limit is not None and claimed is not None and claimed > policy_limit:
        reasons.append(
            f"claim of {claimed:.2f} exceeds the policy limit of {policy_limit:.2f}; the "
            f"excess cannot be paid under this policy"
        )
        _charge(budget, "review", "rejected")
        out = Review(claim_id=claim.claim_id, approved=False, reasons=tuple(reasons),
                     issues=tuple(issues), requires_human=False)
        return validate_handoff(Stage.REVIEWED, out), False

    if inv.prior_claim_count >= 5:
        reasons.append(
            f"{inv.prior_claim_count} prior claims in 12 months exceeds the manual-review "
            f"threshold; automated payout is not appropriate"
        )
        _charge(budget, "review", "rejected")
        out = Review(claim_id=claim.claim_id, approved=False, reasons=tuple(reasons),
                     issues=tuple(issues), requires_human=True)
        return validate_handoff(Stage.REVIEWED, out), False

    # Bounce: enough was found to be worth another pass, but not enough to decide.
    if issues and claimed is not None and policy_limit is not None:
        reasons.append(
            "evidence is incomplete but the claim is within policy; one more investigation "
            "pass could resolve it"
        )
        _charge(budget, "review", "bounced")
        out = Review(claim_id=claim.claim_id, approved=False, reasons=tuple(reasons),
                     issues=tuple(issues), requires_human=True)
        return validate_handoff(Stage.REVIEWED, out), True

    if not inv.findings:
        reasons.append("investigation produced no findings; cannot support a recommendation")
        _charge(budget, "review", "no findings")
        out = Review(claim_id=claim.claim_id, approved=False, reasons=tuple(reasons),
                     issues=tuple(issues), requires_human=True)
        return validate_handoff(Stage.REVIEWED, out), False

    reasons.append(
        f"claim of {claimed:.2f} is within the {policy_limit:.2f} limit and the estimate is "
        f"supported by {len(inv.citations)} cited document(s)"
        if claimed is not None and policy_limit is not None
        else "claim is within policy and supported by the available documents"
    )
    _charge(budget, "review", "approved")
    out = Review(claim_id=claim.claim_id, approved=True, reasons=tuple(reasons),
                 issues=tuple(issues), requires_human=True)
    return validate_handoff(Stage.REVIEWED, out), False


def approver(claim: Claim, state: ClaimState, budget: Budget, approve: bool, who: str) -> ClaimState:
    """The human approval gate. The only node that can release money.

    Exists as a node rather than a branch in the supervisor so that "every payout passed
    through here" is a property of the node path rather than a claim about the routing logic.
    A reviewer can approve; only this can pay.
    """
    state.enter("approver")
    if not approve:
        return _finish(state, Terminal.REJECTED, payout=0.0, who=who)
    amount = state.review is not None and state.documents is not None \
        and state.documents.claimed_amount or 0.0
    return _finish(state, Terminal.APPROVED, payout=float(amount), who=who)


def _finish(state: ClaimState, terminal: Terminal, payout: float, who: str | None) -> ClaimState:
    state.terminal = terminal
    state.payout_amount = payout
    state.approved_by = who
    return state


def json_text(blob) -> str:
    import json

    return json.dumps(blob, sort_keys=True, default=str)
