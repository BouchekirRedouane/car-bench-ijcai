# Codex Python-Call Purple Agent

This package is a reference implementation for a Python-call DSL harness:

1. Codex Spark returns ordinary chat text plus one fenced `python` action block.
2. The adapter extracts that block and parses it with Python's built-in `ast`
   module.
3. Parsed calls are converted into normal A2A text responses or tool-call
   DataParts for green.

The generated Python is never executed. This is inspired by programmatic tool
calling, but it is not true code execution and it does not add hidden tools.
Green remains the only component that executes CAR-bench tools.

## Accepted DSL

Codex may include a short private note before the action block:

````text
The request needs a specific percentage before changing the shade.

```python
respond("What percentage should I set it to?")
```
````

The code block itself may contain:

```python
respond("Sure, what percentage should I set it to?")
```

```python
open_close_sunshade(percentage=50)
```

```python
get_user_preferences(preference_categories={"vehicle_settings": {"vehicle_settings": True}})
open_close_sunshade(percentage=50)
```

Only top-level direct calls are accepted. Imports, assignments, variables,
attributes, loops, conditionals, comprehensions, positional tool arguments, and
non-literal arguments are rejected.

The code block should contain either `respond(...)` or tool calls. If Codex
emits a valid tool call followed by a premature `respond("Done")`, the parser
keeps the tool call and ignores the response. Green will send the tool result
back, and the agent can confirm completion on the next turn.

The older `{"python_code": "..."}` JSON envelope is still accepted by the
parser for compatibility, but the default prompt uses fenced Python because it
is closer to Codex's natural code-proposal behavior.

## Local Run

```bash
uv run agentbeats-run scenarios/scenario-test-codex-python.toml --show-logs
```

## Docker Run

```bash
uv run python generate_compose.py --scenario scenarios/scenario-docker-codex-python.toml
mkdir -p output
docker compose up --abort-on-container-exit
```

Set `CODEX_HOME_HOST` in `.env` to an absolute host path containing Codex auth.
