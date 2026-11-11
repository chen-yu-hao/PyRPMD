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


def init_project(directory, task="fes"):
    """Write one bundled workflow tutorial into a new or empty directory.

    The input is deliberately self-contained: no checkout, geometry file,
    downloaded potential or optional analysis backend is required. Existing
    files are never replaced.
    """
    templates = {
        "fes": ("quickstart.inp",), "pimd": ("pimd.inp",),
        "rate": ("rate.inp", "toy_exchange.py", "ts.xyz"),
        "drpmd": ("drpmd.inp", "reactant_a.xyz", "reactant_b.xyz"),
    }
    if task not in templates:
        raise InputError("Starter task must be fes, pimd, rate or drpmd")
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
    resources = files("pyrpmd").joinpath("templates")
    contents = {("quickstart.inp" if index == 0 else name): resources.joinpath(name).read_text(encoding="utf-8")
                for index, name in enumerate(templates[task])}
    created_directory = not target.exists()
    target.mkdir(parents=True, exist_ok=True)
    input_path = target / "quickstart.inp"
    created = []
    try:
        # Exclusive creation also protects against another init winning a race.
        for name, content in contents.items():
            path = target / name
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                created.append(path)
                stream.write(content)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
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
        "task": task,
        "purpose": ("Harmonic free-energy tutorial with an exact reference, F(x) = k x^2 / 2"
                    if task == "fes" else
                    "Short workflow tutorial; not a converged scientific calculation"),
        "next_steps_shell": "powershell" if os.name == "nt" else "posix",
        "next_steps": [_command_text(step, windows=os.name == "nt") for step in steps],
        "next_step_arguments": steps,
    }
