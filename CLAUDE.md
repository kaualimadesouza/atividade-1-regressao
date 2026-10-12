# CLAUDE.md

Guidance for Claude Code in this repository. What the project is and how to run it: `README.md`.

## No tests

This repository has no test suite, on purpose. Do not add `tests/`, pytest, coverage settings or
test dependencies. Check a change by running the pipeline, looking at what it writes to `outputs/`
and recompiling the report.

## Review every change

After every change request to this repository, launch the thermo-nuclear review
(`.claude/skills/thermo-nuclear-code-quality-review/SKILL.md`) in a separate agent on the diff of
that change. Findings are claims, not orders: verify each one against the code, apply what holds,
and report which findings were applied and which were rejected, and why.
