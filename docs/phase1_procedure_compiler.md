# Phase 1: the procedure graph and its compiler

## What the graph is

A procedure is a partial order over steps. Each step installs or removes one of the 11 components
of the IndustReal toy car, and the step ids follow the dataset's `procedure_info.json` (install of
component k is id 3k, removal is 3k + 2). The graph lives in `stepwitness/procedures/*.json` and the
code in `stepwitness/procedure.py`.

Two rules do the real work:

- `allowed(completed)` returns the steps whose preconditions are all complete. It is recomputed after
  every completion, so several steps can be allowed at once. This is how the dataset's 48 valid build
  orders are expressed without listing them.
- `observable_only_with` says that a chassis without any pin, or a bracket without its screw, does not
  count as an installed component. The dataset labels those as intermediate, not as states.

## Validation result

`Procedure.validate` enumerates every set of steps closed under preconditions, computes the state code
each one produces, keeps the observable ones, and compares with the dataset's labelled states.

| Procedure | Reachable observable states | Labelled states | Missing | Extra |
|---|---:|---:|---:|---:|
| assembly | 13 | states 1 to 13 | 0 | 0 |
| maintenance | 11 | states 13 to 22 plus 10 | 0 | 0 |

Together the two graphs reach exactly the 22 labelled states and nothing else. `tests/test_procedure.py`
locks this in.

## The compiler

`scripts/compile_procedure.py` sends the numbered instruction sheet to a local language model
(Ollama, qwen3:8b) and asks for one record per action line: the component, install or remove, the
lines that must be done first, and a subgoal name. Step ids are computed from the component index in
code, so the model cannot get an id wrong. The draft is turned into a graph, validated against the
labelled states, and diffed against the reviewed graph.

### Attempt 1: original sheet, no thinking

The sheet said "the bracket needs both subgoals complete" without naming the lines.

- Validation: FAIL, 10 reachable observable states the dataset never labelled.
- Differences from the reviewed graph: 6. The model treated "Start from the base" as a step, made the
  bracket and the rear wheels depend on the chassis lines instead of the pin lines, and added a
  redundant dependency of the front wheels on the bracket line.

The validator caught the under-constrained draft without any human reading it: with the bracket
allowed before the pins, the graph reaches states such as "bracket on, rear pins out", which the
dataset never labelled because its instructions never allow it.

### Attempt 2: sheet names the lines explicitly, no thinking

The sheet now says "which means lines 3, 5 and 6 done" and "line 1 is not a step".

- Assembly: validation FAIL (10 extra states), 3 differences. The model dropped the base step, but still
  wired the bracket and rear wheels to the chassis lines rather than the pin lines it was told.
- Maintenance: validation FAIL (6 extra states), 1 difference: the short chassis was not made to depend
  on removing the old one.

### Attempt 3: thinking enabled

See `runs/phase1/*.think.report.md` once complete; results are appended below when measured.

## What this shows

The 8B model is a usable drafter and an unreliable reader of order constraints. That is fine, because
the design never trusts the draft: the validator rejects any graph that reaches an unlabelled state,
and the reviewer fixes the few wrong edges. For a real plant the labelled-state check is replaced by
the plant's own list of acceptable intermediate states, or simply by the reviewer.

The reviewed graphs committed in the repo are the ones every later phase uses.
