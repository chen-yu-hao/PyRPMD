"""Create a portable first calculation using only installed package resources."""

from __future__ import annotations

from importlib.resources import files
import os
from pathlib import Path
import shlex
import sys

from .input import InputError


def _command_text(arguments, *, windows):
    if windows:
        # The call operator also works when the interpreter path has spaces.
        # Single-quoted PowerShell arguments treat $, backticks and spaces literally.
        return "& " + " ".join("'" + argument.replace("'", "''") + "'" for argument in arguments)
    return shlex.join(arguments)


def init_project(directory):
    """Write the small harmonic FES tutorial into a new or empty directory.

    The input is deliberately self-contained: no checkout, geometry file,
    downloaded potential or optional analysis backend is required. Existing
    files are never replaced.
    """
    requested = Path(directory).expanduser()
    if requested.is_symlink():
        raise InputError(f"Starter directory must not be a symbolic link: {requested}")
    target = requested.resolve()
    if target.exists():
        if not target.is_dir():
            raise InputError(f"Starter destination is not a directory: {target}")
        if any(target.iterdir()):
            raise InputError(
                f"Starter directory is not empty: {target}; choose a new or empty directory. "
                "Existing files are never overwritten."
            )
    template = files("pyrpmd").joinpath("templates", "quickstart.inp").read_text(encoding="utf-8")
    created_directory = not target.exists()
    target.mkdir(parents=True, exist_ok=True)
    input_path = target / "quickstart.inp"
    created_input = False
    try:
        # Exclusive creation also protects against another init winning a race.
        with input_path.open("x", encoding="utf-8", newline="\n") as stream:
            created_input = True
            stream.write(template)
    except OSError:
        if created_input:
            input_path.unlink(missing_ok=True)
        if created_directory:
            try:
                target.rmdir()
            except OSError:
                pass  # Preserve files created by another process.
        raise
    output = target / "run"
    command = [sys.executable, "-m", "pyrpmd"]
    steps = [
        [*command, "validate", str(input_path), "--check-pes"],
        [*command, "run", str(input_path)],
        [*command, "analyze", str(output)],
        [*command, "resume", str(output)],
    ]
    return {
        "created": True,
        "directory": str(target),
        "input": str(input_path),
        "output": str(output),
        "task": "fes",
        "purpose": "Short harmonic tutorial; not a converged scientific calculation",
        "next_steps_shell": "powershell" if os.name == "nt" else "posix",
        "next_steps": [_command_text(step, windows=os.name == "nt") for step in steps],
        "next_step_arguments": steps,
    }
