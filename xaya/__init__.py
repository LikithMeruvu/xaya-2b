"""XAYA-2B mini SDK. Loads existing weights; does no training."""
import math
from ._inputs import validate


class XAYA:
    def __init__(self, engine):
        self._engine = engine

    @classmethod
    def from_pretrained(cls, model_dir, *, load_in_4bit=True, device="cuda:0", max_length=8192):
        """Load an extracted XAYA checkpoint; the pinned base downloads on first use."""
        from ._runtime import Runtime
        return cls(Runtime(model_dir, load_in_4bit=load_in_4bit,
                           device=device, max_length=max_length))

    def decide(self, state, question, options, *, primitive="choice", image=None):
        options = validate(question, options, primitive, self._engine.max_options)
        return self._engine.decide(state, question, options, primitive, image)

    def choice(self, state, question, options, *, image=None):
        return self.decide(state, question, options, image=image)

    def score(self, state, question, *, levels=(1, 2, 3, 4, 5), descriptions=None, image=None):
        """Probability distribution over numeric ordinal levels, plus its expected value."""
        levels = list(levels)
        if (not levels or any(isinstance(v, bool) or not isinstance(v, (int, float))
                             or not math.isfinite(v) for v in levels)
                or len(set(levels)) != len(levels) or levels != sorted(levels)):
            raise ValueError("levels must be distinct, finite numbers in ascending order")
        options = [{"label": str(v), "description": (descriptions or {}).get(v)} for v in levels]
        result = self.decide(state, question, options, primitive="score", image=image)
        return {**result, "levels": levels,
                "expected_score": sum(v*p for v, p in zip(levels, result["probabilities"]))}

    def yes_no(self, state, question, *, image=None):
        result = self.decide(state, question, ["Yes", "No"], primitive="noul", image=image)
        return {**result, "yes": result["probabilities"][0], "no": result["probabilities"][1]}


__all__ = ["XAYA"]
