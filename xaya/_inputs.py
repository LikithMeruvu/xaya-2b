"""Input checks and the byte-equivalent original decision prompt."""
from collections.abc import Mapping


def validate(question, options, primitive, max_options=256):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if primitive not in {"choice", "score", "noul"}:
        raise ValueError("primitive must be choice, score, or noul")
    if isinstance(options, (str, bytes, Mapping)):
        raise ValueError("options must be a sequence of candidates")
    options = list(options)
    if not 1 <= len(options) <= max_options:
        raise ValueError(f"provide between 1 and {max_options} options")
    for option in options:
        if isinstance(option, str):
            label = option
        elif isinstance(option, Mapping):
            label = option.get("label", "")
        else:
            raise ValueError("each option must be a string or a label/description dict")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("each candidate needs a nonempty string label")
    return options


def option_text(option):
    if isinstance(option, str):
        return option
    label = str(option.get("label", ""))
    return label if not option.get("description") else f"{label} - {option['description']}"


def prompt(state, question, options, primitive):
    lines = '\n'.join(f'Option {i+1}: {option_text(o)}' for i, o in enumerate(options))
    return (f"Decision type: {primitive.upper()}\n\nContext:\n{state}\n\n"
            f"Question:\n{question}\n\nAllowed options:\n{lines}\n\nChoose exactly one option.")
