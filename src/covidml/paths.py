"""Locate project data and outputs without changing the working directory."""

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the covidml checkout above the current directory or ``start``."""
    directory = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "covidml"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the covidml project root. Run from the project "
        "directory or one of its subdirectories, such as notebooks/."
    )
