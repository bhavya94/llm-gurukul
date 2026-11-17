import ast
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _abilities() -> list[str]:
    """Every package the project installs, except medha: one per ability. pyproject.toml lists them, so a new
    ability is checked here as soon as it is installed."""
    with open(REPO / "pyproject.toml", "rb") as file:
        packages = tomllib.load(file)["tool"]["uv"]["build-backend"]["module-name"]
    abilities = []
    for name in packages:
        if name != "medha":
            abilities.append(name)
    return abilities


def test_the_shared_model_imports_no_ability():
    """Abilities import medha, never the reverse, so every ability can build on the same model."""
    abilities = _abilities()
    assert abilities, "pyproject.toml names no ability"
    for path in sorted((REPO / "medha").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in abilities, f"medha/{path.name} imports {name}"
