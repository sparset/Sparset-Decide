"""Reusable decision rubrics with independent, per-call input data."""

import copy
import json
from pathlib import Path
from collections.abc import Mapping

from .schema import question_from_dict


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class Workflow:
    """Keep rubric definitions, never conversation state or model weights."""

    def __init__(self, data):
        if not isinstance(data, Mapping) or not isinstance(data.get("questions"), Mapping):
            raise ValueError("Workflow needs a questions object")
        if not data["questions"]:
            raise ValueError("Workflow needs at least one question")
        self._definitions = copy.deepcopy(dict(data["questions"]))
        for key, definition in self._definitions.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Question names must be nonempty strings")
            if not isinstance(definition, Mapping):
                raise ValueError(f"Question {key!r} must be an object")
            # Missing instructions are permitted until request binding; other
            # schema errors should fail before an expensive model load.
            check = dict(definition)
            check.setdefault("instructions", "Request supplies this question")
            question_from_dict(check)

    @classmethod
    def load(cls, path="workflow.json"):
        return cls(read_json(path))

    @property
    def question_names(self):
        return tuple(self._definitions)

    def default_prompt(self, name):
        return self._definitions[name].get("instructions")

    def bind(self, *, context=None, prompt=None, prompts=None, messages=None):
        """Return ordinary engine inputs, without mutating this workflow.

        prompt replaces a single field's instructions. For multiple fields,
        prompts maps field names to replacement instructions; omitted fields use
        their saved defaults. Conversation messages are evaluated as input data,
        never installed as the engine's system/assistant messages.
        """
        if prompt is not None and prompts is not None:
            raise ValueError("Use prompt or prompts, not both")
        if prompt is not None:
            if len(self._definitions) != 1:
                raise ValueError("For multiple questions, use prompts keyed by question name")
            prompts = {next(iter(self._definitions)): prompt}
        if prompts is None:
            prompts = {}
        if not isinstance(prompts, Mapping):
            raise ValueError("prompts must map question names to instructions")
        unknown = set(prompts) - set(self._definitions)
        if unknown:
            raise ValueError(f"Unknown prompt question names: {list(unknown)!r}")
        questions = {}
        for key, definition in self._definitions.items():
            bound = dict(definition)
            if key in prompts:
                bound["instructions"] = prompts[key]
            if "instructions" not in bound:
                raise ValueError(f"Supply a prompt for question {key!r}")
            questions[key] = question_from_dict(bound)

        if context is not None and not isinstance(context, str):
            raise ValueError("context must be a string")
        if messages is not None:
            if not isinstance(messages, (list, tuple)) or not messages:
                raise ValueError("messages must be a nonempty list of role/content objects")
            history = []
            for message in messages:
                if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                    raise ValueError("Each message needs exactly role and content")
                if message["role"] not in ("system", "developer", "user", "assistant", "tool"):
                    raise ValueError("Unsupported conversation role")
                if not isinstance(message["content"], str):
                    raise ValueError("Message content must be text")
                history.append(dict(message))
            # Preserve order and roles without interpreting conversation text as
            # engine instructions. All branches receive the same serialized data.
            context = json.dumps({"context": context or "", "messages": history}, ensure_ascii=False)
        if context is None or not context.strip():
            raise ValueError("Supply nonempty context or conversation messages for this request")
        return context, questions

    def bind_request(self, request):
        if not isinstance(request, Mapping):
            raise ValueError("Request must be an object")
        unknown = set(request) - {"context", "prompt", "prompts", "messages"}
        if unknown:
            raise ValueError(f"Unknown request fields: {list(unknown)!r}")
        return self.bind(**request)

    def run(self, engine, *, context=None, prompt=None, prompts=None, messages=None, mode="parallel"):
        """Reuse a loaded engine for arbitrary new inputs on every call."""
        context, questions = self.bind(context=context, prompt=prompt, prompts=prompts, messages=messages)
        return engine.decide(context, questions, mode=mode)
