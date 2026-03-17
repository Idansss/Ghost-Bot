from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import NoReturn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd or ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _alembic(*args: str) -> list[str]:
    return [sys.executable, "-m", "alembic", *args]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dev", description="Ghost_Bot developer workflow helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Run the app (FastAPI + bot)")

    p_test = sub.add_parser("test", help="Run tests")
    p_test.add_argument("pytest_args", nargs=argparse.REMAINDER)

    sub.add_parser("lint", help="Run ruff lint checks")
    sub.add_parser("format", help="Run ruff formatter")
    sub.add_parser("typecheck", help="Run mypy")

    sub.add_parser("ci", help="Run lint + typecheck + tests")

    sub.add_parser("migrate", help="Apply migrations (alembic upgrade head)")
    p_rev = sub.add_parser("makemigration", help="Create a new Alembic revision (autogenerate)")
    p_rev.add_argument("-m", "--message", default="migration", help="Revision message")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        _run(_python_module("app.main"))
        return 0

    if args.cmd == "test":
        _run(_python_module("pytest", "-q", *getattr(args, "pytest_args", [])))
        return 0

    if args.cmd == "lint":
        _run(_python_module("ruff", "check", "."))
        return 0

    if args.cmd == "format":
        _run(_python_module("ruff", "format", "."))
        return 0

    if args.cmd == "typecheck":
        _run(_python_module("mypy"))
        return 0

    if args.cmd == "ci":
        _run(_python_module("ruff", "check", "."))
        _run(_python_module("mypy"))
        _run(_python_module("pytest", "-q"))
        return 0

    if args.cmd == "migrate":
        _run(_alembic("upgrade", "head"))
        return 0

    if args.cmd == "makemigration":
        _run(_alembic("revision", "--autogenerate", "-m", str(args.message)))
        return 0

    raise SystemExit(f"Unknown command: {args.cmd}")


def _main() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _main()

