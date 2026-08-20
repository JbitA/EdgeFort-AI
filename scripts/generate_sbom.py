from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from importlib import metadata
from pathlib import Path
import re
import tomllib
import uuid

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SBOM_PATH = ROOT / "artifacts" / "sbom.spdx.json"


def _project_metadata() -> tuple[str, str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    project = data["project"]
    name = project["name"]
    version = project["version"]
    roots = []
    for raw in project.get("dependencies", []):
        roots.append(Requirement(raw).name)
    return name, version, roots


def _created_timestamp() -> str:
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is not None:
        try:
            epoch = int(source_date_epoch)
        except ValueError as e:
            raise SystemExit("SOURCE_DATE_EPOCH must be an integer") from e
        if epoch < 0:
            raise SystemExit("SOURCE_DATE_EPOCH must be non-negative")
        return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spdx_id(name: str) -> str:
    return f'SPDXRef-Package-{re.sub("[^A-Za-z0-9.-]", "-", name)}'


def build_inventory(root_dependencies: list[str]):
    seen: dict[str, tuple[str, str, str]] = {}
    relationships: list[tuple[str, str]] = []

    def visit(name: str) -> None:
        key = canonicalize_name(name)
        if key in seen:
            return
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as e:
            raise SystemExit(f"runtime dependency is not installed: {name}") from e
        actual = distribution.metadata.get("Name", name)
        version = distribution.version
        spdx = _spdx_id(actual)
        seen[key] = (actual, version, spdx)
        for raw in distribution.requires or []:
            try:
                requirement = Requirement(raw)
            except Exception:
                continue
            if requirement.marker and not requirement.marker.evaluate():
                continue
            child = canonicalize_name(requirement.name)
            visit(requirement.name)
            if child in seen:
                relationships.append((spdx, seen[child][2]))

    for dependency in root_dependencies:
        visit(dependency)
    return seen, relationships


def main() -> int:
    project_name, project_version, roots = _project_metadata()
    seen, transitive_relationships = build_inventory(roots)
    project_spdx = f"SPDXRef-Package-{canonicalize_name(project_name)}"

    packages = [
        {
            "SPDXID": project_spdx,
            "name": project_name,
            "versionInfo": project_version,
            "downloadLocation": "NOASSERTION",
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
        }
    ]
    for actual, version, spdx in sorted(seen.values(), key=lambda item: canonicalize_name(item[0])):
        packages.append(
            {
                "SPDXID": spdx,
                "name": actual,
                "versionInfo": version,
                "downloadLocation": "NOASSERTION",
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
            }
        )

    relationships = []
    for root in roots:
        key = canonicalize_name(root)
        relationships.append(
            {
                "spdxElementId": project_spdx,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": seen[key][2],
            }
        )
    for parent, child in sorted(set(transitive_relationships)):
        relationships.append(
            {
                "spdxElementId": parent,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": child,
            }
        )

    pins = "|".join(
        f"{canonicalize_name(actual)}=={version}"
        for actual, version, _ in sorted(seen.values(), key=lambda item: canonicalize_name(item[0]))
    )
    namespace_uuid = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{canonicalize_name(project_name)}|{project_version}|{pins}",
    )
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{project_name}-runtime-sbom",
        "documentNamespace": f"urn:uuid:{namespace_uuid}",
        "creationInfo": {
            "created": _created_timestamp(),
            "creators": ["Tool: edge-ai-deployment-platform/scripts/generate_sbom.py"],
        },
        "packages": packages,
        "relationships": relationships,
    }

    SBOM_PATH.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    print(f"{len(packages)} runtime packages, {len(relationships)} relationships")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
