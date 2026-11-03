"""Static guard: the official TEST set is reachable only from fedrl/eval, scripts/ and tests/
(plus fedrl/data/scenario.py for evaluation-only personal test sets, and its definition file)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "fedrl"
ALLOWED_DIRS = [ROOT / "eval"]
ALLOWED_FILES = {ROOT / "data" / "scenario.py", ROOT / "data" / "datasets.py"}


def _files():
    for p in sorted(ROOT.rglob("*.py")):
        if p in ALLOWED_FILES or any(d in p.parents for d in ALLOWED_DIRS):
            continue
        yield p


def test_no_get_official_test_import_outside_allowed():
    bad = []
    for p in _files():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and any(a.name == "get_official_test" for a in node.names):
                bad.append(str(p))
            elif isinstance(node, ast.Import) and any("get_official_test" in a.name for a in node.names):
                bad.append(str(p))
            elif isinstance(node, ast.Attribute) and node.attr == "get_official_test":
                bad.append(str(p))
            elif isinstance(node, ast.Name) and node.id == "get_official_test":
                bad.append(str(p))
    assert not bad, f"get_official_test used outside fedrl/eval and fedrl/data/scenario.py: {sorted(set(bad))}"


def test_no_direct_test_tensor_access_outside_allowed():
    bad = []
    for p in _files():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in ("test_x", "test_y"):
                bad.append(f"{p}:{node.lineno}")
    assert not bad, f"Direct test-set tensor access outside allowed modules: {bad}"