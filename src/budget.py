"""Budget ceilings and the loop guard: the three conditions that make a graph terminate.

The ship gate asks for a cost ceiling "enforced by code and not by hope", and the trap it warns
about is "agents that hand off in a circle with no stop condition". Both are addressed here, and
the design decision worth stating is that there are **three independent** termination
conditions rather than one:

    1. max_steps        - a hard cap on node visits. Trivially terminating, and blunt.
    2. max_bounces      - caps reviewer -> investigator rejections specifically. This is the
                          one that targets the actual failure mode: a reviewer that keeps
                          rejecting an investigator that keeps re-answering is an infinite loop
                          that a global step cap would only stop after burning the whole
                          budget first.
    3. budget ceiling   - tokens and dollars. Stops a run that is looping *slowly*.

One is not enough. A step cap alone means a loop runs until it is expensive. A budget cap alone
means a cheap loop runs until it is slow. A bounce cap alone misses loops that are not
reviewer/investigator ping-pong. All three are checked in one place, `Budget.check()`, before
any node runs, so no node can route around them.

The cost model is explicit and simple: a per-model price table, tokens counted at the call
site, dollars derived. It is not a vendor invoice, and pretending otherwise would be the
over-engineering this project exists to avoid - what matters is that the number is *derived
from the same counter that stops the run*, not tracked in a spreadsheet afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .state import ClaimState, Terminal


class BudgetExceeded(Exception):
    """Raised when a run must stop because a ceiling was reached."""

    def __init__(self, terminal: Terminal, detail: str) -> None:
        super().__init__(detail)
        self.terminal = terminal
        self.detail = detail


# USD per 1M tokens, (input, output). Deliberately the cheapest plausible free-tier models:
# the point is that the ceiling is enforced regardless of price, and testing with a model that
# costs nothing makes the accounting trivially reproducible.
PRICES: dict[str, tuple[float, float]] = {
    "offline": (0.0, 0.0),
    "llama-3.3-70b": (0.59, 0.79),
    "llama-3.1-8b": (0.05, 0.08),
    "gemini-2.0-flash": (0.10, 0.40),
}


@dataclass
class Budget:
    """Per-claim ceilings. One instance per claim; the run cannot exceed it."""

    max_tokens: int = 20_000
    max_dollars: float = 0.50
    max_steps: int = 12
    # Per-edge cap on reviewer rejections. Separate from max_steps because it is the only one
    # that distinguishes "this reviewer disagrees twice" from "this run did a lot of work".
    max_bounces: int = 2
    model: str = "offline"

    tokens_used: int = 0
    dollars_used: float = 0.0
    steps_used: int = 0
    bounces_used: int = 0
    # Which ceiling stopped the run, for the chart. Kept separate from the exception detail so
    # the report can group runs by *why* they stopped.
    stop_reason: str | None = None
    history: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ accounting

    def price(self, model: str | None = None) -> tuple[float, float]:
        return PRICES.get(model or self.model, PRICES["offline"])

    def charge(self, input_tokens: int, output_tokens: int, model: str | None = None) -> float:
        """Record usage and return the dollar cost of this call.

        `input_tokens` and `output_tokens` are separated because output is typically an order
        of magnitude more expensive, and a single blended token count would hide that. The
        offline model costs nothing, which is what makes the offline run reproducible to the
        cent while still exercising every code path.
        """
        pin, pout = self.price(model)
        cost = (input_tokens / 1_000_000) * pin + (output_tokens / 1_000_000) * pout
        self.tokens_used += input_tokens + output_tokens
        self.dollars_used += cost
        self.history.append({
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "dollars": round(cost, 8), "tokens_total": self.tokens_used,
            "dollars_total": round(self.dollars_used, 8),
        })
        return cost

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    def headroom_fraction(self) -> float:
        """Fraction of the dollar ceiling still available, clamped to [0, 1].

        The supervisor uses this to decide whether a node is worth running at all. A node whose
        worst-case cost exceeds the remaining budget is skipped rather than started, because
        starting it and being cut off mid-way leaves a half-written record that is worse than a
        clean refusal.
        """
        if self.max_dollars <= 0:
            return 0.0
        return max(0.0, min(1.0, (self.max_dollars - self.dollars_used) / self.max_dollars))

    # ------------------------------------------------------------------ gates

    def check(self) -> None:
        """Raise if any ceiling is already reached. Called before every node runs.

        Checking *before* the node rather than after is the whole point: a run that discovers
        it is out of budget only once the API call has already been paid for has enforced
        nothing.
        """
        if self.steps_used >= self.max_steps:
            self.stop_reason = "step_limit"
            raise BudgetExceeded(
                Terminal.STEP_LIMIT,
                f"step limit reached: {self.steps_used}/{self.max_steps} node visits",
            )
        if self.bounces_used >= self.max_bounces:
            self.stop_reason = "bounce_limit"
            raise BudgetExceeded(
                Terminal.STEP_LIMIT,
                f"reviewer bounce limit reached: {self.bounces_used}/{self.max_bounces} "
                f"rejections - the graph would otherwise ping-pong between reviewer and "
                f"investigator until the budget ran out",
            )
        if self.tokens_used >= self.max_tokens:
            self.stop_reason = "token_budget"
            raise BudgetExceeded(
                Terminal.BUDGET_EXHAUSTED,
                f"token budget exhausted: {self.tokens_used}/{self.max_tokens}",
            )
        if self.dollars_used >= self.max_dollars:
            self.stop_reason = "dollar_budget"
            raise BudgetExceeded(
                Terminal.BUDGET_EXHAUSTED,
                f"dollar budget exhausted: ${self.dollars_used:.4f}/${self.max_dollars:.4f}",
            )

    def record_step(self) -> None:
        self.steps_used += 1

    def record_bounce(self) -> None:
        self.bounces_used += 1

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens, "max_dollars": self.max_dollars,
            "max_steps": self.max_steps, "max_bounces": self.max_bounces,
            "model": self.model, "tokens_used": self.tokens_used,
            "dollars_used": self.dollars_used, "steps_used": self.steps_used,
            "bounces_used": self.bounces_used, "stop_reason": self.stop_reason,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "Budget":
        b = cls(
            max_tokens=blob["max_tokens"], max_dollars=blob["max_dollars"],
            max_steps=blob["max_steps"], max_bounces=blob.get("max_bounces", 2),
            model=blob.get("model", "offline"),
        )
        b.tokens_used = blob.get("tokens_used", 0)
        b.dollars_used = blob.get("dollars_used", 0.0)
        b.steps_used = blob.get("steps_used", 0)
        b.bounces_used = blob.get("bounces_used", 0)
        b.stop_reason = blob.get("stop_reason")
        b.history = list(blob.get("history", []))
        return b


@dataclass
class RunReport:
    """Per-run accounting, aggregated into the cost-per-claim chart."""

    claim_id: str
    terminal: Terminal
    steps: int
    bounces: int
    tokens: int
    dollars: float
    stop_reason: str | None
    node_path: list[str]
    fingerprint: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id, "terminal": self.terminal.value, "steps": self.steps,
            "bounces": self.bounces, "tokens": self.tokens,
            "dollars": round(self.dollars, 6), "stop_reason": self.stop_reason,
            "node_path": self.node_path, "fingerprint": self.fingerprint,
        }


def report_for(state: ClaimState, budget: Budget) -> RunReport:
    return RunReport(
        claim_id=state.claim_id,
        terminal=state.terminal or Terminal.FAILED,
        steps=budget.steps_used,
        bounces=budget.bounces_used,
        tokens=budget.tokens_used,
        dollars=budget.dollars_used,
        stop_reason=budget.stop_reason,
        node_path=list(state.node_path),
        fingerprint=state.fingerprint(),
    )
