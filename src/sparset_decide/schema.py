"""Runtime-defined questions: option labels never become fixed model classes."""

from dataclasses import dataclass
from typing import Mapping

MAX_OPTIONS = 26


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Question:
    kind: str
    instructions: str
    options: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if self.kind not in {"choice", "noul", "score"}:
            raise ValueError("Unsupported question type")
        _text(self.instructions, "instructions")
        if not 2 <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(f"A question needs 2–{MAX_OPTIONS} options")
        labels = []
        for label, description in self.options:
            labels.append(_text(label, "option label"))
            _text(description, "option description")
        if len(set(labels)) != len(labels):
            raise ValueError("Option labels must be unique")
        if self.kind == "noul" and labels != ["no", "yes"]:
            raise ValueError("Noul requires ordered no/yes options")
        if self.kind == "score" and labels != [str(i) for i in range(len(labels))]:
            raise ValueError("Score levels must be ordered indices starting at zero")

    def answer(self, probabilities):
        import math

        p = list(probabilities)
        if len(p) != len(self.options) or any(not math.isfinite(v) or v < 0 for v in p):
            raise ValueError("Invalid probability vector")
        if not math.isclose(sum(p), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one")
        winner = max(range(len(p)), key=p.__getitem__)
        result = {"type": self.kind, "probabilities": {
            option[0]: probability for option, probability in zip(self.options, p)
        }}
        if self.kind == "choice":
            result["choice"] = self.options[winner][0]
            result["selected_probability"] = p[winner]
        elif self.kind == "noul":
            result["noul"] = p[1]
        else:
            result["score"] = sum(i * value for i, value in enumerate(p))
            result["legend"] = dict(self.options)
        return result


def Choice(instructions: str, criteria: Mapping[str, str]) -> Question:
    if not isinstance(criteria, Mapping):
        raise ValueError("Choice criteria must map labels to descriptions")
    return Question("choice", instructions, tuple(criteria.items()))


def Noul(instructions: str) -> Question:
    return Question("noul", instructions, (("no", "No"), ("yes", "Yes")))


def Score(instructions: str, criteria: list[str]) -> Question:
    if not isinstance(criteria, (list, tuple)):
        raise ValueError("Score criteria must be an ordered list of descriptions")
    return Question("score", instructions, tuple((str(i), v) for i, v in enumerate(criteria)))


def question_from_dict(data: dict) -> Question:
    if not isinstance(data, dict):
        raise ValueError("Each question must be an object")
    kind = data.get("type")
    allowed = {"type", "instructions"} | ({"criteria"} if kind != "noul" else set())
    if set(data) - allowed:
        raise ValueError(f"Unknown question fields: {sorted(set(data) - allowed)}")
    if kind == "choice":
        return Choice(data.get("instructions"), data.get("criteria"))
    if kind == "noul":
        return Noul(data.get("instructions"))
    if kind == "score":
        return Score(data.get("instructions"), data.get("criteria"))
    raise ValueError(f"Unknown question type: {kind!r}")
