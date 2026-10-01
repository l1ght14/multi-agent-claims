"""Casefile: a bounded multi-agent claim pipeline.

The library is five modules with one direction of dependency - `machine` imports `nodes`,
`budget`, `store` and `state`; nothing imports `machine`. That is what keeps the graph legible:
you can read `machine.step` and know the whole control flow without opening anything else.

Termination lives in `budget`, routing lives in `machine`, judgement lives in `nodes`, and
persistence lives in `store`. None of them reaches into another.
"""
