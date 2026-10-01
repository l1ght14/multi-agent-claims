"""Thirty synthetic insurance claims, built to exercise every termination path.

A corpus of thirty clean claims would produce thirty identical traces and would prove that the
machine can walk a straight line. The value here is in the awkward cases, so the claims are
constructed backwards from the terminal states the ship gate requires:

  * clean claims that reach the human gate                 -> the happy path
  * over-limit claims that must be rejected                 -> a policy fact, not a judgement
  * claims missing evidence, which make the reviewer bounce -> ship gate item 3
  * claims that bounce repeatedly and must terminate        -> the loop guard, under load
  * a claim with no documents at all                        -> the no-evidence short circuit
  * high-frequency claimants                               -> manual-review threshold

Written by hand rather than generated, because the interesting property of each claim is which
branch it drives and a template cannot express "this one specifically has a repair estimate 3x
below the claimed amount".

Nothing here is real data and nothing is real money. The amounts are small and the policies are
invented. This exists to drive a state machine, not to model an insurer.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes import Claim

POLICY = {"collision": 5_000.0, "comprehensive": 12_000.0}


def _claim(
    cid: str,
    *,
    amount: float,
    repair: float | None,
    prior: int,
    limit: float | None = 5_000.0,
    missing: tuple[str, ...] = (),
    incident: str = "2026-03-14",
    vehicle: str = "2021 Volvo XC40",
    all_docs: bool = True,
) -> Claim:
    docs: dict[str, str] = {}
    # `all_docs=False` means the claim arrived with nothing but the policy schedule, which is
    # the short-circuit case. Otherwise each required document is included unless it is named
    # in `missing`. These are independent switches, not alternatives: an earlier version used
    # `all_docs or ... not in missing`, so the default of True silently discarded every
    # `missing` in the corpus and all 30 claims came out complete.
    if all_docs:
        if "police_report" not in missing:
            docs["police_report"] = f"Police report {cid}: collision at junction 4, no injuries."
        if "photos" not in missing:
            docs["photos"] = f"12 photographs attached to claim {cid}."
        if "repair_estimate" not in missing:
            docs["repair_estimate"] = (
                f"Independent repair estimate for {cid}: {repair:.2f} EUR." if repair else ""
            )
    docs["policy_schedule.pdf"] = f"Policy limit: {limit:.2f} EUR." if limit else ""

    truth: dict = {"incident_date": incident, "policy_limit": limit, "prior_claims": prior}
    if repair:
        truth["repair_estimate"] = repair
    if amount != repair:
        truth["amount_mismatch"] = True
    return Claim(claim_id=cid, filed="2026-03-20", vehicle=vehicle, claimed_amount=amount,
                 documents=docs, truth=truth)


def build_claims() -> list[Claim]:
    """Thirty claims. Counts per outcome are asserted by the tests, so the corpus cannot
    silently lose the coverage it exists to provide."""
    claims: list[Claim] = []

    # ---- 8 clean claims: reach the human gate, awaiting a human decision
    for i in range(1, 9):
        claims.append(_claim(f"CLM-{i:03d}", amount=1_200 + i * 90, repair=1_150 + i * 85,
                             prior=i % 3))

    # ---- 4 over the policy limit: terminal rejection, no human needed
    for i, amount in enumerate((5_400.0, 6_100.0, 7_850.0, 9_999.0), start=9):
        claims.append(_claim(f"CLM-{i:03d}", amount=amount, repair=5_200.0, prior=1,
                             limit=POLICY["collision"]))

    # ---- 4 high-frequency claimants: manual review threshold, terminal rejection + human
    for i in range(13, 17):
        claims.append(_claim(f"CLM-{i:03d}", amount=900.0 + i * 40, repair=880.0 + i * 38,
                             prior=5 + (i % 3)))

    # ---- 6 missing evidence: within policy, so the reviewer BOUNCES back
    for i in range(17, 23):
        claims.append(_claim(f"CLM-{i:03d}", amount=1_600 + i * 55, repair=1_500 + i * 50,
                             prior=1, missing=("photos",)))

    # ---- 3 wildly inflated: within the limit but ~3x the repair estimate, so bounced.
    # The amounts have to stay UNDER the 5000 policy limit or the reviewer rejects them on the
    # limit check before it ever reaches the evidence check, and the claim stops exercising the
    # bounce path. An earlier version used 4400 + i*60, which for i >= 20 exceeds 5000 and
    # silently turned all three into plain rejections.
    for i in range(23, 26):
        claims.append(_claim(f"CLM-{i:03d}", amount=4_000 + i * 30, repair=1_100 + i * 20,
                             prior=1))

    # ---- 2 with no documents at all: short circuit at the extractor
    claims.append(_claim("CLM-026", amount=2_100.0, repair=None, prior=0, all_docs=False))
    claims.append(_claim("CLM-027", amount=3_300.0, repair=None, prior=2, all_docs=False))

    # ---- 2 that are both over-limit and high-frequency: rejection wins, as it should
    claims.append(_claim("CLM-028", amount=8_400.0, repair=7_900.0, prior=7,
                         limit=POLICY["collision"]))
    claims.append(_claim("CLM-029", amount=6_200.0, repair=5_800.0, prior=6,
                         limit=POLICY["collision"]))

    # ---- 1 near the limit: approves on a technicality, still routed to a human
    claims.append(_claim("CLM-030", amount=4_999.0, repair=4_400.0, prior=1,
                         limit=POLICY["collision"]))

    return claims


def claims_by_id() -> dict[str, Claim]:
    return {c.claim_id: c for c in build_claims()}


if __name__ == "__main__":
    from collections import Counter

    cs = build_claims()
    over = [c for c in cs
            if c.truth.get("policy_limit") and c.claimed_amount > c.truth["policy_limit"]]
    print(f"claims: {len(cs)}")
    print(f"  document count histogram : {dict(Counter(len(c.documents) for c in cs))}")
    print(f"  over the policy limit    : {len(over)} -> {[c.claim_id for c in over]}")
    print(f"  high frequency (>=5)     : "
          f"{sum(1 for c in cs if c.truth.get('prior_claims', 0) >= 5)}")
    print(f"  missing evidence (<4)    : "
          f"{sum(1 for c in cs if len(c.documents) < 4)}")
    print(f"  no claim documents       : {sum(1 for c in cs if len(c.documents) == 1)}")
    print(f"  inflated (>2x repair)    : "
          f"{sum(1 for c in cs if c.truth.get('repair_estimate') and c.claimed_amount / c.truth['repair_estimate'] > 2)}")
