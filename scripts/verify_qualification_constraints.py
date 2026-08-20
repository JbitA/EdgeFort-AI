from __future__ import annotations

"""Verify that the committed qualification lock covers the installed dev/runtime closure."""

from importlib import metadata
from pathlib import Path
import platform
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CONSTRAINTS = ROOT / "artifacts" / "qualification-constraints.txt"


def _roots() -> list[str]:
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    raw = list(project.get("dependencies", []))
    raw.extend(project.get("optional-dependencies", {}).get("dev", []))
    return [Requirement(value).name for value in raw]


def _pins() -> dict[str, tuple[str, str]]:
    pins: dict[str, tuple[str, str]] = {}
    for line_number, raw in enumerate(CONSTRAINTS.read_text().splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        requirement = Requirement(text)
        if requirement.marker is not None or requirement.extras:
            raise SystemExit(f"line {line_number}: qualification pins cannot use markers/extras")
        specs = list(requirement.specifier)
        if len(specs) != 1 or specs[0].operator != "==" or specs[0].version.endswith(".*"):
            raise SystemExit(f"line {line_number}: qualification pin must use one exact == version")
        key = canonicalize_name(requirement.name)
        if key in pins:
            raise SystemExit(f"line {line_number}: duplicate qualification pin {requirement.name}")
        pins[key] = (requirement.name, specs[0].version)
    return pins


def _installed_closure(roots: list[str]) -> dict[str, tuple[str, str]]:
    seen: dict[str, tuple[str, str]] = {}

    def visit(name: str) -> None:
        key = canonicalize_name(name)
        if key in seen:
            return
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as e:
            raise SystemExit(f"qualification dependency is not installed: {name}") from e
        actual = distribution.metadata.get("Name", name)
        seen[key] = (actual, distribution.version)
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            visit(requirement.name)

    for root in roots:
        visit(root)
    return seen


def main() -> int:
    if sys.version_info[:2] != (3, 13) or platform.system() != "Linux":
        raise SystemExit("qualification constraints are defined for CPython 3.13 on Linux")
    pins = _pins()
    installed = _installed_closure(_roots())
    missing = sorted(set(installed) - set(pins))
    extra = sorted(set(pins) - set(installed))
    mismatched = sorted(
        (installed[key][0], installed[key][1], pins[key][1])
        for key in set(installed) & set(pins)
        if installed[key][1] != pins[key][1]
    )
    if missing or extra or mismatched:
        raise SystemExit(
            f"qualification lock mismatch: missing={missing}, extra={extra}, versions={mismatched}"
        )
    print(f"qualification lock covers {len(installed)} installed packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
