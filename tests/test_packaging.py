"""Packaging promises, checked rather than asserted in a README.

Each of these is a claim the project makes to someone who has not read the source: that
annotations are visible to their type checker, that installing it pulls nothing else in,
and that the version a bug report quotes matches what the metadata says.
"""

import pathlib
import tomllib

import postflight

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_py_typed_marker_ships_inside_the_package():
    """PEP 561: without this file INSIDE the installed package, every annotation here is
    invisible to a consumer's type checker and the `Typing :: Typed` claim is a lie."""
    assert (ROOT / "postflight" / "py.typed").is_file()


def test_no_runtime_dependencies():
    """The zero-dependency promise is the reason this can be added to anything. A single
    entry here quietly turns it into a version-pin negotiation."""
    assert PYPROJECT["project"].get("dependencies") == []


def test_version_matches_the_package():
    assert PYPROJECT["project"]["version"] == postflight.__version__


def test_declared_python_floor_is_tested():
    """`requires-python` is a promise CI has to keep — claiming 3.11 without running it
    is how 3.12-only syntax reaches someone whose first news of it is a SyntaxError."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    floor = PYPROJECT["project"]["requires-python"].lstrip(">=")
    assert f"'{floor}'" in ci, f"requires-python claims {floor}; CI does not run it"
