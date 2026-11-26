import re

from lm_eval.tasks.mbpp.utils import pass_at_1  # noqa: F401


def _extract(text):
    m = re.findall(r"```(?:python)?\n(.*?)\n?```", "```python\n" + text, re.DOTALL)
    return m[0] if m else text


def build_predictions(resps, docs):
    return [[_extract(r) for r in resp] for resp in resps]
