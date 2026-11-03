"""Canonical relative paths for portable authenticated run archives."""

from __future__ import annotations


def archive_path(value):
    """Accept historical Windows separators without weakening confinement.

    Archive names are relative, nonempty path components. Drive/UNC paths,
    parent traversal, alternate streams and ambiguous aliases are not members.
    This parser does not touch disk or alter authenticated metadata bytes.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("Archive paths must be nonempty relative strings")
    canonical = value.replace("\\", "/")
    if (":" in canonical or "\x00" in canonical
            or any(part in ("", ".", "..") for part in canonical.split("/"))):
        raise ValueError("Invalid relative archive path")
    return canonical


def archive_path_mapping(value, *, path_values=False):
    """Normalize a mapping in memory, rejecting duplicate path aliases."""
    if not isinstance(value, dict):
        raise ValueError("Archive paths must be recorded in a mapping")
    canonical = {}
    for name, item in value.items():
        path = archive_path(name)
        if path in canonical:
            raise ValueError("Archive mapping contains duplicate path aliases")
        canonical[path] = archive_path(item) if path_values else item
    return canonical
