r"""Check method-card markdown for GitHub MathJax rendering pitfalls.

GitHub renders ``$...$`` / ``$$...$$`` / ``$`...`$`` with MathJax 3 in
strict mode.  Common failures that produce user-visible errors on the
rendered page:

- ``^``/``_`` without an argument
  ("Missing superscript or subscript argument") — e.g. ``$`class_prior_`$``
- unbalanced ``$`` delimiters, which make the parser swallow plain text
- unbalanced braces or ``\begin``/``\end`` environments

Run:  uv run python scripts/check_math_rendering.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BLOCK_PAT = re.compile(r"```math\n(.*?)```", re.S)
INLINE_PAT = re.compile(r"\$`(.*?)`\$|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", re.S)
# ^/_ followed by whitespace, '}', '&', '\\' (literal double backslash = math
# line break), or end of the math chunk is a missing argument.  A single
# backslash starts a command (^\top) and is legal.
BAD_ARG = re.compile(r"[\^_](?:\s+|[}]|&|\\\\|$)")
EMPTY_GROUP = re.compile(r"[\^_]\{\}")
ENV_PAT = re.compile(r"\\(begin|end)\{([a-zA-Z*]+)\}")


def extract_math(text: str) -> list[tuple[str, str, int]]:
    chunks: list[tuple[str, str, int]] = []
    for m in BLOCK_PAT.finditer(text):
        line = text[: m.start()].count("\n") + 1
        chunks.append((m.group(1), "block", line))
    for m in INLINE_PAT.finditer(text):
        line = text[: m.start()].count("\n") + 1
        chunks.append((m.group(1) or m.group(2) or "", "inline", line))
    return chunks


def check_missing_args(chunk: str, kind: str, line: int) -> list[str]:
    errors: list[str] = []
    for m in BAD_ARG.finditer(chunk):
        start = max(0, m.start() - 20)
        ctx = chunk[start : m.end() + 10].replace("\n", "⏎")
        errors.append(f"{kind}:{line} missing superscript/subscript argument ...{ctx}...")
    for m in EMPTY_GROUP.finditer(chunk):
        start = max(0, m.start() - 20)
        ctx = chunk[start : m.end() + 10].replace("\n", "⏎")
        errors.append(f"{kind}:{line} empty ^{{}}/_{{}} group ...{ctx}...")
    return errors


def check_braces(chunk: str, kind: str, line: int) -> list[str]:
    errors: list[str] = []
    flat = chunk.replace(r"\{", "").replace(r"\}", "")
    depth = 0
    for i, ch in enumerate(flat):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"{kind}:{line} extra '}}' ...{flat[max(0, i - 15) : i + 15]}...")
                depth = 0
    if depth > 0:
        errors.append(f"{kind}:{line} unbalanced braces (depth={depth})")
    envs: list[str] = []
    for m in ENV_PAT.finditer(chunk):
        tag, name = m.group(1), m.group(2)
        if tag == "begin":
            envs.append(name)
        elif not envs or envs[-1] != name:
            errors.append(f"{kind}:{line} \\end{{{name}}} without matching \\begin")
        else:
            envs.pop()
    if envs:
        errors.append(f"{kind}:{line} unclosed \\begin{{{envs[-1]}}}")
    return errors


def main(argv: list[str] | None = None) -> int:
    files = sorted((PROJECT_ROOT / "docs" / "research" / "method_cards").glob("*.md"))
    if not files:
        print("No method cards found; refusing to pass empty scan.", file=sys.stderr)
        return 1
    total = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        errors: list[str] = []
        for chunk, kind, line in extract_math(text):
            errors += check_missing_args(chunk, kind, line)
            errors += check_braces(chunk, kind, line)
        # Unbalanced $ per line (odd count)
        for i, line in enumerate(text.splitlines(), 1):
            if line.count("$") % 2 == 1:
                errors.append(f"line {i}: odd number of '$' delimiters")
        if errors:
            total += len(errors)
            print(f"== {f} ==")
            for e in errors:
                print(f"  {e}")
    print(f"\nTotal issues: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
