"""The procedure graph: the rulebook that says which steps may complete next.

A procedure is a partial order over steps. Each step installs or removes one component. Component k
is digit k of the 11-digit state code that IndustReal uses: 1 means correctly installed, 0 means not
installed, -1 means installed incorrectly. Step ids follow the dataset's procedure_info.json so that
results plug straight into its metric code: id 3k installs component k, 3k + 1 is "incorrectly
installed k", and 3k + 2 removes k.

Two ideas carry the design:

* ``allowed(completed)`` is the set of steps whose preconditions are all complete. It changes after
  every completion. More than one step can be allowed at once, which is how flexible build orders are
  expressed without listing every order.
* Some states are physically unobservable: a chassis that has been placed but not pinned is not a
  "completed" chassis in the dataset's labels, because it is not fixed yet. ``observable_only_with``
  records which companion component must be present for a component to count. Reachable observable
  states must match the dataset's labelled states exactly, and ``validate`` checks that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

N_COMPONENTS = 11
COMPONENTS = (
    "base",
    "front chassis",
    "front chassis pin",
    "rear chassis",
    "short rear chassis",
    "front-rear chassis pin",
    "rear-rear chassis pin",
    "front bracket",
    "front bracket screw",
    "front wheel assembly",
    "rear wheel assembly",
)

State = tuple[int, ...]

PROCEDURES_DIR = Path(__file__).resolve().parent / "procedures"


def parse_state(code: str) -> State:
    """'11100000000' -> (1, 1, 1, 0, ...). Also accepts comma-separated ints with -1 entries."""
    if "," in code:
        values = tuple(int(v) for v in code.split(","))
    else:
        values = tuple(int(ch) for ch in code.strip())
    if len(values) != N_COMPONENTS:
        raise ValueError(f"state must have {N_COMPONENTS} digits, got {len(values)}: {code!r}")
    return values


def state_str(state: State) -> str:
    """(1, 1, 1, 0, ...) -> '11100000000'. A -1 is written as '-1', so error states are not 11 chars."""
    return "".join(str(v) for v in state)


def install_id(component: int) -> int:
    return 3 * component


def remove_id(component: int) -> int:
    return 3 * component + 2


def error_id(component: int) -> int:
    return 3 * component + 1


@dataclass(frozen=True)
class Step:
    id: int
    component: int
    kind: str  # 'install' or 'remove'
    requires: frozenset[int]  # step ids that must be complete first
    subgoal: str
    line: int  # 1-based line in the instruction sheet
    text: str

    def __post_init__(self) -> None:
        if self.kind not in ("install", "remove"):
            raise ValueError(f"step {self.id}: kind must be install or remove, got {self.kind!r}")
        expected = install_id(self.component) if self.kind == "install" else remove_id(self.component)
        if self.id != expected:
            raise ValueError(f"step id {self.id} does not match {self.kind} of component {self.component} (expected {expected})")


@dataclass
class ValidationReport:
    reachable_observable: set[State]
    reachable_unobservable: set[State]
    labelled: set[State]

    @property
    def missing(self) -> set[State]:
        """Labelled states the procedure can never reach: a graph error."""
        return self.labelled - self.reachable_observable

    @property
    def extra(self) -> set[State]:
        """Observable states the procedure reaches that the dataset never labelled: a graph error."""
        return self.reachable_observable - self.labelled

    @property
    def ok(self) -> bool:
        return not self.missing and not self.extra


class Procedure:
    def __init__(
        self,
        name: str,
        start_state: State,
        steps: list[Step],
        observable_only_with: dict[int, frozenset[int]] | None = None,
    ):
        self.name = name
        self.start_state = start_state
        self.steps: dict[int, Step] = {s.id: s for s in steps}
        self.observable_only_with = observable_only_with or {}
        for s in steps:
            unknown = s.requires - self.steps.keys()
            if unknown:
                raise ValueError(f"step {s.id} requires unknown steps {sorted(unknown)}")
        self._check_acyclic()

    # ----- construction -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "Procedure":
        steps = [
            Step(
                id=int(s["id"]),
                component=int(s["component"]),
                kind=s["kind"],
                requires=frozenset(int(r) for r in s.get("requires", [])),
                subgoal=s.get("subgoal", ""),
                line=int(s.get("line", 0)),
                text=s.get("text", ""),
            )
            for s in d["steps"]
        ]
        oow = {int(k): frozenset(int(v) for v in vs) for k, vs in d.get("observable_only_with", {}).items()}
        return cls(d["name"], parse_state(d["start_state"]), steps, oow)

    @classmethod
    def load(cls, path: str | Path) -> "Procedure":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def builtin(cls, name: str) -> "Procedure":
        return cls.load(PROCEDURES_DIR / f"{name}.json")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_state": state_str(self.start_state),
            "observable_only_with": {str(k): sorted(v) for k, v in self.observable_only_with.items()},
            "steps": [
                {
                    "id": s.id,
                    "component": s.component,
                    "kind": s.kind,
                    "requires": sorted(s.requires),
                    "subgoal": s.subgoal,
                    "line": s.line,
                    "text": s.text,
                }
                for s in sorted(self.steps.values(), key=lambda s: (s.line, s.id))
            ],
        }

    # ----- the partial order --------------------------------------------------------------------

    def _check_acyclic(self) -> None:
        seen: set[int] = set()
        stack: set[int] = set()

        def visit(i: int) -> None:
            if i in stack:
                raise ValueError(f"precondition cycle through step {i}")
            if i in seen:
                return
            stack.add(i)
            for r in self.steps[i].requires:
                visit(r)
            stack.discard(i)
            seen.add(i)

        for i in self.steps:
            visit(i)

    def allowed(self, completed: set[int] | frozenset[int]) -> set[int]:
        """Steps that may complete next: not yet complete, all preconditions complete."""
        done = set(completed)
        return {i for i, s in self.steps.items() if i not in done and s.requires <= done}

    def is_downset(self, completed: set[int] | frozenset[int]) -> bool:
        done = set(completed)
        return all(self.steps[i].requires <= done for i in done)

    def topological(self, completed: set[int] | frozenset[int]) -> list[int]:
        """The completed steps in an order consistent with their preconditions."""
        done = set(completed)
        if not self.is_downset(done):
            raise ValueError(f"{sorted(done)} is not closed under preconditions")
        order: list[int] = []
        remaining = set(done)
        while remaining:
            ready = sorted(i for i in remaining if self.steps[i].requires <= set(order))
            if not ready:
                raise ValueError("cycle")
            order.extend(ready)
            remaining -= set(ready)
        return order

    def apply(self, state: State, step_id: int) -> State:
        s = self.steps[step_id]
        values = list(state)
        values[s.component] = 1 if s.kind == "install" else 0
        return tuple(values)

    def state_after(self, completed: set[int] | frozenset[int]) -> State:
        """The state code once the given down-set of steps is complete."""
        state = self.start_state
        for i in self.topological(completed):
            state = self.apply(state, i)
        return state

    def missing_steps(self, state: State, completed: set[int] | frozenset[int]) -> set[int]:
        """Steps whose effect is visible in ``state`` but that are not in ``completed``."""
        done = set(completed)
        current = self.state_after(done)
        out = set()
        for i, s in self.steps.items():
            if i in done:
                continue
            target = 1 if s.kind == "install" else 0
            if state[s.component] == target and current[s.component] != target:
                out.add(i)
        return out

    # ----- observability and validation ---------------------------------------------------------

    def is_observable(self, state: State) -> bool:
        for component, companions in self.observable_only_with.items():
            if state[component] == 1 and not any(state[c] == 1 for c in companions):
                return False
        return True

    def downsets(self) -> list[frozenset[int]]:
        """Every set of steps closed under preconditions, i.e. every reachable completed-set."""
        ids = sorted(self.steps)
        found: set[frozenset[int]] = {frozenset()}
        frontier = [frozenset()]
        while frontier:
            nxt = []
            for done in frontier:
                for i in self.allowed(done):
                    new = done | {i}
                    if new not in found:
                        found.add(new)
                        nxt.append(new)
            frontier = nxt
        assert all(set(d) <= set(ids) for d in found)
        return sorted(found, key=lambda d: (len(d), sorted(d)))

    def reachable_states(self) -> dict[State, list[frozenset[int]]]:
        out: dict[State, list[frozenset[int]]] = {}
        for done in self.downsets():
            out.setdefault(self.state_after(done), []).append(done)
        return out

    def validate(self, labelled: set[State]) -> ValidationReport:
        reachable = set(self.reachable_states())
        observable = {s for s in reachable if self.is_observable(s)}
        return ValidationReport(observable, reachable - observable, set(labelled))


def labelled_states_industreal() -> dict[int, State]:
    """The 22 correct assembly states labelled in IndustReal, numbered as in the dataset."""
    codes = [
        "10000000000", "10010010000", "10010100000", "10010110000", "11100000000",
        "11110010000", "11110100000", "11110110000", "11110111100", "11110111110",
        "11110110001", "11110111101", "11110111111", "11110101111", "11110011111",
        "11110011110", "11110101110", "11100001110", "11101101110", "11101011110",
        "11101111110", "11101111111",
    ]
    return {i + 1: parse_state(c) for i, c in enumerate(codes)}
