# First-Run Validation

Use this checklist with someone who has not worked on Zaxy before. The goal is
to verify that a normal local developer can install, initialize, inspect memory,
and run one example in less than five minutes.

## Commands

```bash
pipx install zaxy-memory
zaxy init
zaxy memory bootstrap --eventloom-path .eventloom
zaxy memory checkout "current project memory and next useful action" --eventloom-path .eventloom
zaxy doctor --eventloom-path .eventloom
python examples/single_agent_memory.py
```

## Report

- Operating system:
- Shell:
- Python version:
- Install method:
- Time to successful `zaxy doctor`:
- Time to first successful example:
- Did any command require Docker, Neo4j, Postgres, or a graph password?
- Where did you get stuck?
- Which error message was least useful?
- What should the quick start say differently?
