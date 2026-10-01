# Casefile — a bounded multi-agent claims pipeline

A regional auto insurer receives 900 claims a day. Each one needs documents extracted, prior
claims and policy limits checked, a damage estimate sanity-checked against repair costs, and a
recommendation written. One prompt cannot hold that. Four coordinated agents can — **as long as
the system stops when it should and never pays out without a human.**

This project is about the second clause. The first is a routing problem; the second is the
whole design.

---

## The claim the project makes

A multi-agent system is defined by its **termination conditions, loop guards and cost
ceilings** — not by its agent personas. Everything here is built around that ordering:

| requirement | how it is enforced |
|---|---|
| the graph terminates | three independent ceilings, checked before every node |
| the loop is bounded | a per-edge bounce cap on `reviewer → investigator` |
| the budget is real | the same counter that charges a call raises the ceiling |
| money never moves automatically | `Terminal.APPROVED` is reachable only through `approver` |
| a run is auditable | a snapshot at every step, checksummed |
| a run can be restarted | resume from any snapshot, verified to reach the same decisions |

---

## Results

30 claims, three terminal states, **every run terminated**.

| terminal | n | why |
|---|---|---|
| `awaiting_human` | 9 | the reviewer approved; a person still has to release the money |
| `rejected` | 12 | over the policy limit, manual-review threshold, or no readable documents |
| `step_limit` | 9 | the reviewer bounced twice and the loop guard stopped it |

### Ship gate

| # | requirement | result |
|---|---|---|
| 1 | 30 recorded runs, each with a trace | **30/30**, node paths recorded |
| 2 | one run replayed from a snapshot, same terminal state | **PASS** — `CLM-017` resumed from step 3 |
| 3 | the reviewer bounces at least once, graph still terminates | **PASS** — 9 claims bounced; all 30 terminated at 3 budget settings |
| 4 | cost per claim charted, ceiling enforced in code | **PASS** — 4,328 tokens, 144.3 mean; dollar ceiling demonstrably fires |

Replay detail, because "it resumed" is not the same as "it resumed correctly":

```
claim        CLM-017   (a bouncing claim, not a happy path)
resumed at   step 3    (immediately after the first reviewer bounce)
terminal     step_limit -> step_limit
fingerprint  4fd3443fca2d8150 vs 4fd3443fca2d8150
path         extractor -> investigator -> reviewer -> investigator -> reviewer
```

The replay had to **re-derive** the bounce from a snapshot taken at the moment of rejection, not
replay a decision already sitting on disk.

---

## Architecture

```
ClaimMachine.step()
  NEW          -> extractor      typed handoff: ExtractedDocuments
  EXTRACTED    -> investigator   typed handoff: Investigation
                (short circuit if no readable documents)
  INVESTIGATED -> reviewer       typed handoff: Review  (+ bounce decision)
  REVIEWED     -> approver       only reachable path to APPROVED

Budget.check() runs before every node and raises BudgetExceeded on any of:
  steps_used >= max_steps  |  bounces_used >= max_bounces
  tokens_used >= max_tokens  |  dollars_used >= max_dollars
```

**Typed handoffs, no free text.** Every inter-node value is a frozen dataclass with a schema,
validated at the boundary by `validate_handoff()`. A node returning a string raises `TypeError`
naming the stage — not three steps later with a confusing attribute error. There is no
`dict[str, Any]` anywhere in the handoff path.

**Three independent termination conditions, not one.** A step cap alone means a loop runs until
it is expensive. A budget cap alone means a cheap loop runs until it is slow. A bounce cap alone
misses loops that are not `reviewer ↔ investigator` ping-pong. All three are checked in one
place, before any node runs, so no node can route around them.

---

## Findings

**1. The reviewer was computing a correct verdict that the router threw away.** Every
`REVIEWED` state went to the human gate without reading `review.approved`. So a claim the
reviewer had already rejected for exceeding the policy limit still reached `AWAITING_HUMAN` —
19 claims instead of 9. The reviewer was right and the one line that should have read its
verdict did not exist. This is the failure the project exists to prevent, sitting in the router.

**2. Blocking the payout path is necessary but not sufficient.** Bouncing rewinds the stage to
`EXTRACTED` so the investigator re-runs, which looks like an opportunity to reset the
counters. It must not: the step counter, the bounce counter and the budget all carry across the
rewind, which is what makes `max_bounces` a bound rather than a suggestion. `test_the_bounce_counter_survives_the_rewind`
pins it.

**3. The first replay check proved almost nothing, and it passed.** It selected the first claim
with enough snapshots, which was `CLM-001` — a three-node happy path. Replaying that shows the
snapshot round-trips; it says nothing about routing after a loop. Worse, the replay wrote new
snapshots back into the same directory, so the *second* invocation resumed from a snapshot the
*first* one had created. Now it selects a bouncing claim, resumes just after the first bounce,
replays into a **separate** store, and asserts the archive under test was not modified.

**4. The checksum caught a real round-trip bug in my own fingerprint.** Tuples become lists
through JSON and `repr` renders them differently, so `ClaimState.fingerprint()` differed
between save and load and *every* snapshot failed verification. The guard was not
over-triggering; it was correctly reporting that the fingerprint was not a function of content.

---

## What was wrong along the way

- The router ignored `review.approved` (finding 1).
- `with_stage()` derived the dataclass field name from `Stage.value`, so `Stage.EXTRACTED` tried
  to set a field called `extracted` when the field is `documents`. Now an explicit map.
- The corpus builder used `all_docs or "x" not in missing`, and `all_docs` defaults to `True` —
  so every `missing` in the corpus was silently discarded and all 30 claims came out complete.
  The missing-evidence and no-document paths did not exist until this was fixed.
- Three claims intended to be "inflated but within policy" computed to 5,780+, over the 5,000
  limit, so the reviewer rejected them on the limit check before ever reaching the evidence
  check. They were testing the wrong branch.
- The ship gate's loop-guard sweep wiped the shared store before the cost gate read it, so
  "30 claims" reported as "0 claims".
- `_SAFE` accepted `..` as a claim id. The `.json` suffix stopped it escaping, but a guard that
  needs that argument is a guard that will eventually be wrong.
- The replay selected a straight-line run, then polluted the store it was measuring.

---

## Running it

```bash
pip install -r requirements.txt

python eval/shipgate.py       # all four gate requirements; exits 1 on any failure
pytest                        # 44 tests, 90% coverage
```

No API key and no network. The nodes are deterministic functions of their input plus a token
cost, which is what makes the replay assertion meaningful — a live model would introduce
sampling variance the check could never distinguish from a logic bug. The LLM path is behind the
same node interface and shares every termination control.

## Structure

```
src/
  state.py     ClaimState, the three typed handoffs, validation, fingerprint
  budget.py    ceilings, charge accounting, the three gates, RunReport
  store.py     external state: snapshots, checksums, resume, run archive
  nodes.py     extractor / investigator / reviewer / approver
  machine.py   routing and the loop; the only module that imports the others
corpus/claims.py   30 synthetic claims built backwards from the terminal paths
eval/shipgate.py   the four gate checks + figures
tests/             44 tests
```

Dependency direction is strictly one-way: `machine` imports `nodes`, `budget`, `store` and
`state`; nothing imports `machine`. Reading `machine.step` gives you the entire control flow.

## Limitations

- **The nodes are not models.** They are deterministic functions. The termination, budget and
  replay machinery is real and fully exercised; the *judgement* is not. Swapping in an LLM
  changes determinism, which is why replay would then need a tolerance rather than equality.
- **Cost figures are from the offline model, which is free.** The dollar ceiling is proven with a
  priced model in the gate, but the per-claim chart is in tokens. A real run would need the
  LLM path enabled to have a meaningful dollar figure.
- **The store is a file backend, not Redis or Postgres.** Chosen because this project is about
  replay and termination semantics, and a networked store changes neither while adding a
  failure mode. Swapping it means reimplementing three methods; nothing above `store.py` moves.
- **The 30 claims are mine.** They are built to drive branches, so they cover the paths I
  anticipated. A real claims book would find others.

## Licence

Code: MIT. No third-party data; the 30 claims are synthetic and the amounts are invented.
