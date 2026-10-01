"""The state machine: routing, termination, snapshots, and the human gate.

Hand-rolled, and the spec allows either this or LangGraph. The choice is the argument: this
project's entire claim is that termination conditions and budget ceilings are the interesting
part of a multi-agent system, and a framework that hides them behind a decorator is hiding the
thing being demonstrated. Every gate is visible in `step()`.

The run is a loop over exactly four positions:

    EXTRACTED -> INVESTIGATED -> REVIEWED -> { approver | investigator }

and it leaves that loop for a terminal state on any of five conditions:

    reviewer approves           -> AWAITING_HUMAN   (a human must still release the money)
    reviewer rejects outright   -> REJECTED
    bounce budget exhausted     -> STEP_LIMIT
    step budget exhausted       -> STEP_LIMIT
    token or dollar ceiling hit -> BUDGET_EXHAUSTED

Two properties are worth naming because they are the difference between this and a demo:

**The machine cannot pay out.** `Terminal.APPROVED` is only reachable through `approver`,
which requires a human decision. There is no path from a model's approval to a transfer.

**A snapshot is written at every step, including the one that exhausts a ceiling.** A run that
dies on budget is still resumable, and its final state still has a recorded reason. Losing the
last step of a failed run is how a cost bug becomes invisible.
"""
from __future__ import annotations

from dataclasses import dataclass
from .budget import Budget, BudgetExceeded, RunReport, report_for
from .nodes import Claim, NodeError, approver, extractor, investigator, reviewer
from .state import ClaimState, Review, Stage, Terminal
from .store import StateStore


@dataclass
class RunResult:
    state: ClaimState
    budget: Budget
    report: RunReport
    resumed_from: int | None = None

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "budget": self.budget.to_dict(),
            "report": self.report.to_dict(),
            "resumed_from": self.resumed_from,
        }


class ClaimMachine:
    def __init__(self, store: StateStore, budget: Budget | None = None) -> None:
        self.store = store
        self.budget_template = budget or Budget()

    # ---------------------------------------------------------------- resume

    def resume(self, claim_id: str, from_step: int | None = None) -> tuple[ClaimState, Budget, int]:
        """Load a stored snapshot. Raises if there is nothing to resume from.

        Refusing rather than silently starting fresh is deliberate. A resume that quietly
        begins a new run leaves the caller believing it continued one, and the cost chart then
        double-counts or loses a run depending on which.
        """
        snap = (
            self.store.load_step(claim_id, from_step)
            if from_step is not None
            else self.store.load(claim_id)
        )
        if snap is None:
            raise KeyError(f"no stored snapshot for claim {claim_id!r}")
        return snap.state, snap.budget, snap.step

    # ---------------------------------------------------------------- the loop

    def run(
        self,
        claim: Claim,
        state: ClaimState | None = None,
        budget: Budget | None = None,
        human_decision: bool | None = None,
        max_iterations: int = 64,
    ) -> RunResult:
        """Drive one claim to a terminal state.

        `max_iterations` is a fourth safety net and is deliberately set absurdly high (64)
        against a default `max_steps` of 12. It exists only to turn "the loop never exits" into
        a test failure rather than a hung test run - the real termination is done by the
        budget, and if that ever stops working this should be the thing that notices.

        `human_decision`: None means no human responded, which resolves to AWAITING_HUMAN. That
        is the default and it is the point - the machine's default behaviour is to stop and ask
        a person, never to guess.
        """
        if state is None:
            state = ClaimState(claim_id=claim.claim_id)
        if budget is None:
            budget = Budget(**{k: v for k, v in self.budget_template.to_dict().items()
                               if k in ("max_tokens", "max_dollars", "max_steps",
                                        "max_bounces", "model")})
        resumed_from = state.step or None

        iterations = 0
        while state.terminal is None:
            iterations += 1
            if iterations > max_iterations:
                # Reaching this means the budget gates failed to terminate the loop. Fail loudly.
                state.terminal = Terminal.STEP_LIMIT
                budget.stop_reason = "machine_iteration_guard"
                self.store.save(state, budget)
                return RunResult(state, budget, report_for(state, budget), resumed_from)

            try:
                state = self._transition(claim, state, budget, human_decision)
            except BudgetExceeded as exc:
                state.terminal = exc.terminal
                self.store.save(state, budget)
                return RunResult(state, budget, report_for(state, budget), resumed_from)
            except NodeError as exc:
                state.terminal = Terminal.FAILED
                budget.stop_reason = f"node_error:{exc}"
                self.store.save(state, budget)
                return RunResult(state, budget, report_for(state, budget), resumed_from)

            self.store.save(state, budget)

        return RunResult(state, budget, report_for(state, budget), resumed_from)

    # ---------------------------------------------------------------- routing

    def _transition(
        self, claim: Claim, state: ClaimState, budget: Budget, human_decision: bool | None
    ) -> ClaimState:
        """One node visit. Pure routing: every gate lives in `Budget.check`."""
        state.step += 1

        if state.stage is Stage.NEW:
            docs = extractor(claim, state, budget)
            return state.with_stage(Stage.EXTRACTED, docs)

        if state.stage is Stage.EXTRACTED:
            # A claim with no extractable evidence cannot be investigated, and inventing
            # findings to satisfy the next node is the failure mode this branch prevents.
            if state.documents and not state.documents.documents_found:
                state.terminal = Terminal.REJECTED
                state.review = Review(
                    claim_id=claim.claim_id, approved=False,
                    reasons=("no readable documents were submitted with this claim",),
                    requires_human=False,
                )
                return state
            inv = investigator(claim, state, budget)
            return state.with_stage(Stage.INVESTIGATED, inv)

        if state.stage is Stage.INVESTIGATED:
            rev, bounce = reviewer(claim, state, budget)
            state = state.with_stage(Stage.REVIEWED, rev)
            if bounce:
                budget.record_bounce()
                # Rewind to the investigator, discarding the rejected review. The step counter
                # and bounce counter are NOT rewound - that is the loop guard.
                return ClaimState(
                    claim_id=state.claim_id,
                    stage=Stage.EXTRACTED,
                    documents=state.documents,
                    investigation=None,
                    review=None,
                    step=state.step,
                    bounces=state.bounces + 1,
                    tokens_used=state.tokens_used,
                    dollars_used=state.dollars_used,
                    node_path=list(state.node_path),
                )
            return state

        if state.stage is Stage.REVIEWED:
            # The verdict has to be read here. An earlier version routed every REVIEWED state to
            # the human gate without looking at `review.approved`, which meant a claim the
            # reviewer had rejected on a policy limit still reached AWAITING_HUMAN. The
            # reviewer was computing a correct verdict and the router was discarding it - the
            # exact failure this project exists to prevent, sitting in the one place that
            # should have been checked first.
            review = state.review
            if review is None:
                state.terminal = Terminal.FAILED
                budget.stop_reason = "reviewed_without_a_review"
                return state
            if not review.approved:
                # Rejected outright. If the reviewer said a human is needed (high-frequency
                # claimant), still tell one - but do not present it as a recommendation to pay.
                state.terminal = Terminal.REJECTED
                return state
            if human_decision is None:
                # Approved but no human answered: stop here. Never infer approval from silence.
                state.terminal = Terminal.AWAITING_HUMAN
                return state
            return approver(claim, state, budget, approve=human_decision, who="human:adjuster")

        # Unreachable: every stage is handled and the loop exits on `terminal`.
        state.terminal = Terminal.FAILED
        return state
