"""The online step recognizer: turns per-frame verifier outputs into step completions and flags.

It keeps the accumulate-and-decay core of the IndustReal B3 baseline, because that core works and
keeping it makes the numbers comparable, and changes three things:

1. The allowed set comes from the procedure graph and moves after every completion, instead of one
   fixed list per procedure. Steps outside the allowed set do not accumulate.
2. A state that implies steps nobody observed raises omission flags instead of completing them.
3. An execution-error verdict from the verifier holds the step open and raises a flag.

Per frame the recognizer receives a posterior over label states (correct states and error states as
comma codes). It converts each candidate correct state into the set of steps that would take the
current state there, adds the posterior mass to those steps, decays the rest, and completes a step
when its total crosses the threshold and the graph allows it.

Events are dicts with a ``type`` of completion, omission, wrong_order, execution_error or hold,
plus frame, step id, component, confidence and evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stepwitness.procedure import Procedure, State, install_id, remove_id
from stepwitness.verifier import code_to_tuple, correct_of, error_component


@dataclass
class RecognizerConfig:
    threshold: float = 8.0  # accumulated posterior mass needed to complete a step (B3 uses 8 with confidences)
    decay: float = 0.75  # per-frame multiplier for steps not supported this frame (B3 uses 0.75)
    min_posterior: float = 0.30  # the top correct state must carry at least this much mass to count
    error_margin: float = 0.10  # posterior mass on error states of a component above correct, per frame
    error_frames: int = 6  # consecutive frames of error evidence before an execution_error flag
    omission_frames: int = 6  # consecutive frames implying a skipped step before an omission flag
    hold_frames: int = 15  # consecutive low-confidence frames before a hold event
    hold_floor: float = 0.25  # top posterior below this counts as low confidence
    reach: int = 0  # credit only mass on states within this many transitions of the current state (0 = off)
    stable_frames: int = 0  # the top state must hold this many consecutive confident frames before it earns credit (0 = off)


@dataclass
class Recognizer:
    procedure: Procedure
    classes: list[str]  # verifier class order, comma codes
    config: RecognizerConfig = field(default_factory=RecognizerConfig)

    def __post_init__(self) -> None:
        self.completed: set[int] = set()
        self.state: State = self.procedure.start_state
        self.cum: dict[int, float] = {i: 0.0 for i in self.procedure.steps}
        self.events: list[dict] = []
        self.frame = -1
        self._error_streak: dict[int, int] = {}
        self._omission_streak: dict[int, int] = {}
        self._hold_streak = 0
        self._flagged_omissions: set[int] = set()
        self._flagged_errors: dict[int, int] = {}  # component -> frame of last flag
        self._class_tuples = [code_to_tuple(c) for c in self.classes]
        self._reach_cache: dict[tuple, int | None] = {}
        self._last_top: str | None = None
        self._top_streak = 0
        self._class_is_error = [error_component(c) is not None for c in self.classes]
        self._class_correct = [correct_of(c) for c in self.classes]

    # ----- helpers ---------------------------------------------------------------------------

    def steps_toward(self, target: tuple) -> set[int]:
        """Steps of the procedure whose effect turns the current state into ``target``."""
        out = set()
        for i, s in self.procedure.steps.items():
            if i in self.completed:
                continue
            want = 1 if s.kind == "install" else 0
            if target[s.component] == want and self.state[s.component] != want:
                out.add(i)
        return out

    def allowed(self) -> set[int]:
        return self.procedure.allowed(self.completed)

    def reach_of(self, target: tuple) -> int | None:
        """How many steps the procedure needs to turn the current state into ``target``, or None when
        no allowed transition does so. The current state itself has reach 0."""
        key = (self.state, target)
        if key in self._reach_cache:
            return self._reach_cache[key]
        steps = self.steps_toward(target)
        out: int | None = None
        if self.transition_allowed(target):
            try:
                order = self.procedure.topological(self.completed | steps) if steps else []
            except ValueError:  # the implied step set is not closed under preconditions: not a transition
                order = None
            if order is not None:
                state = self.state
                for i in order:
                    if i in steps:
                        state = self.procedure.apply(state, i)
                if state == target:
                    out = len(steps)
        self._reach_cache[key] = out
        return out

    def mask_to_reach(self, state_probs: np.ndarray, state_codes: list[str]) -> np.ndarray:
        """Zero the posterior on states the procedure cannot be in within ``config.reach`` transitions.
        No renormalisation: mass placed on impossible states is simply not credited."""
        k = self.config.reach
        if k <= 0:
            return state_probs
        keep = np.zeros(len(state_codes), dtype=bool)
        for i, c in enumerate(state_codes):
            r = self.reach_of(code_to_tuple(c))
            keep[i] = r is not None and r <= k
        return np.where(keep, state_probs, 0.0)

    def _emit(self, **event) -> dict:
        event.setdefault("frame", self.frame)
        self.events.append(event)
        return event

    # ----- main update -----------------------------------------------------------------------

    def update(self, posterior: np.ndarray, frame: int, evidence: dict | None = None) -> list[dict]:
        """Feed one frame of a posterior over ``self.classes`` (correct and error codes).
        Returns the events raised on this frame."""
        top = int(posterior.argmax())
        top_p = float(posterior[top])
        # mass per correct state; error classes contribute to the error tally of their component when
        # their correct state is one the procedure could be in now (reachable by an allowed transition)
        correct_mass: dict[str, float] = {}
        error_mass: dict[int, float] = {}
        for ci, p in enumerate(posterior):
            if p < 1e-6:
                continue
            if self._class_is_error[ci]:
                comp = error_component(self.classes[ci])
                if self.transition_allowed(code_to_tuple(self._class_correct[ci])):
                    error_mass[comp] = error_mass.get(comp, 0.0) + float(p)
            else:
                correct_mass[self.classes[ci]] = correct_mass.get(self.classes[ci], 0.0) + float(p)
        return self._step(self.classes[top], top_p, correct_mass, error_mass, frame, evidence)

    def update_parts(self, state_probs: np.ndarray, state_codes: list[str], comp_err_probs: np.ndarray, frame: int, evidence: dict | None = None) -> list[dict]:
        """Feed one frame from the two-head verifier: a posterior over correct states and, per
        component, the probability that it is installed wrongly."""
        state_probs = self.mask_to_reach(np.asarray(state_probs, dtype=np.float64), state_codes)
        top = int(state_probs.argmax())
        top_code = state_codes[top]
        top_p = float(state_probs[top])
        top_tuple = code_to_tuple(top_code)
        # the state head answers "which correct state is this closest to"; the error head answers
        # "is component k wrong" independently. Turn the pair into the competing masses the core uses:
        # error mass per installed component, and correct mass discounted by that error evidence.
        error_mass: dict[int, float] = {}
        if self.transition_allowed(top_tuple):
            for k, q in enumerate(comp_err_probs):
                if q > 0.05 and top_tuple[k] == 1:
                    error_mass[k] = float(q)
        correct_mass: dict[str, float] = {}
        for i, p in enumerate(state_probs):
            if p <= 1e-6:
                continue
            t = code_to_tuple(state_codes[i])
            disc = 1.0
            for k, q in error_mass.items():
                if t[k] == 1:
                    disc *= 1.0 - q
            correct_mass[state_codes[i]] = float(p) * disc
        bad = {k for k, q in error_mass.items() if q > 0.5}
        return self._step(top_code, top_p, correct_mass, error_mass, frame, evidence, bad_components=bad)

    def _step(self, top_code: str, top_p: float, correct_mass: dict[str, float], error_mass: dict[int, float], frame: int, evidence: dict | None, bad_components: set[int] | None = None) -> list[dict]:
        self.frame = frame
        before = len(self.events)
        cfg = self.config

        # hold: nothing confident for a while
        if top_p < cfg.hold_floor:
            self._hold_streak += 1
            if self._hold_streak == cfg.hold_frames:
                self._emit(type="hold", confidence=top_p, evidence=evidence)
        else:
            self._hold_streak = 0

        # accumulate toward the steps suggested by the single most likely state, as B3 does with its
        # top detection. A step whose preconditions are complete, or are themselves part of the same
        # observed transition (a chassis and its pin appear together), is allowed; a step whose
        # preconditions are visibly absent is only implied, and feeds the omission logic.
        supported: set[int] = set()
        implied: dict[int, float] = {}
        if top_p >= cfg.min_posterior and top_code == self._last_top:
            self._top_streak += 1
        elif top_p >= cfg.min_posterior:
            self._last_top, self._top_streak = top_code, 1
        else:
            self._last_top, self._top_streak = None, 0
        stable = cfg.stable_frames <= 0 or self._top_streak >= cfg.stable_frames
        if top_p >= cfg.min_posterior and stable:
            bad_set = set(bad_components or ())
            if error_component(top_code) is not None:
                bad_set.add(error_component(top_code))
            # an error class still says every other digit of its correct state is in place; the
            # components read as wrong do not accumulate toward completion
            target = code_to_tuple(correct_of(top_code))
            transition = {i for i in self.steps_toward(target) if self.procedure.steps[i].component not in bad_set}
            for i in transition:
                if self.procedure.steps[i].requires <= (self.completed | transition):
                    self.cum[i] += top_p
                    supported.add(i)
                else:
                    implied[i] = implied.get(i, 0.0) + top_p
        for i in self.cum:
            if i not in supported:
                self.cum[i] *= cfg.decay

        # omissions: a step keeps being implied by the observed state while its preconditions are not done
        for i in list(self._omission_streak):
            if i not in implied:
                self._omission_streak[i] = 0
        for i, p in implied.items():
            self._omission_streak[i] = self._omission_streak.get(i, 0) + 1
            if self._omission_streak[i] == cfg.omission_frames and i not in self._flagged_omissions:
                missing = sorted(self.procedure.steps[i].requires - self.completed)
                self._flagged_omissions.add(i)
                self._emit(type="omission", step=i, component=self.procedure.steps[i].component,
                           implies=missing, line=self.procedure.steps[i].line, confidence=p, evidence=evidence)

        # execution errors: error mass on a component beats its correct mass for several frames
        for comp, em in error_mass.items():
            cm = sum(p for code, p in correct_mass.items() if code_to_tuple(code)[comp] == 1)
            if em - cm > cfg.error_margin:
                self._error_streak[comp] = self._error_streak.get(comp, 0) + 1
            else:
                self._error_streak[comp] = 0
            if self._error_streak.get(comp, 0) == cfg.error_frames:
                step = install_id(comp)
                last = self._flagged_errors.get(comp)
                if last is None or frame - last > 50:
                    self._flagged_errors[comp] = frame
                    self._emit(type="execution_error", step=step, component=comp,
                               line=self.procedure.steps[step].line if step in self.procedure.steps else None,
                               confidence=em, evidence=evidence)
                # reset the accumulated completion so the error holds the step open
                if step in self.cum:
                    self.cum[step] = 0.0
        for comp in list(self._error_streak):
            if comp not in error_mass:
                self._error_streak[comp] = 0

        # completions, in an order that respects preconditions within the same frame
        ready = {i for i in self.cum if self.cum[i] > cfg.threshold}
        progress = True
        while ready and progress:
            progress = False
            for i in sorted(ready):
                comp = self.procedure.steps[i].component
                if self._error_streak.get(comp, 0) >= cfg.error_frames:
                    self.cum[i] = 0.0
                    ready.discard(i)
                    progress = True
                    break
                if self.procedure.steps[i].requires <= self.completed:
                    self._complete(i, evidence)
                    ready.discard(i)
                    progress = True
                    break
        for i in ready:  # crossed the threshold but a precondition is still missing: not completed
            self.cum[i] = 0.0
        return self.events[before:]

    def transition_allowed(self, target: tuple) -> bool:
        """True when every step needed to reach ``target`` from the current state has its
        preconditions complete or inside that same transition."""
        transition = self.steps_toward(target)
        if not transition and target != self.state:
            return target == self.state
        return all(self.procedure.steps[i].requires <= (self.completed | transition) for i in transition)

    def finalize(self, frame: int) -> list[dict]:
        """At the end of a recording, every step of the procedure never completed is an omission."""
        before = len(self.events)
        self.frame = frame
        for i, s in sorted(self.procedure.steps.items()):
            if i not in self.completed and i not in self._flagged_omissions:
                self._flagged_omissions.add(i)
                self._emit(type="omission", step=i, component=s.component, implies=[], line=s.line, confidence=1.0, evidence=None, at_end=True)
        return self.events[before:]

    def _complete(self, step: int, evidence: dict | None) -> None:
        s = self.procedure.steps[step]
        self.completed.add(step)
        self.state = self.procedure.apply(self.state, step)
        conf = self.cum[step]
        self.cum[step] = 0.0
        self._emit(type="completion", step=step, component=s.component, line=s.line, text=s.text, confidence=conf, evidence=evidence)

    # ----- output in the paper's format --------------------------------------------------------

    def y_hat(self, proc_info: list[dict]) -> list[dict]:
        """Completions as IndustReal entries: frame, id, description, conf."""
        desc = {int(a["id"]): a["description"] for a in proc_info}
        return [
            {"frame": e["frame"], "id": e["step"], "description": desc.get(e["step"], ""), "conf": 1}
            for e in self.events
            if e["type"] == "completion"
        ]
