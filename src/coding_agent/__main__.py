"""Allow ``python -m coding_agent`` to behave like the console command."""

from coding_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
