"""The exact CLI must persist its verdict and return a failing exit status on failure."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("coefficient, expected_exit", [(1, 0), (-1, 1)])
def test_exact_cli_persists_success_and_failure(tmp_path, coefficient, expected_exit):
    # xdot = -sign(x), V = +/-x^2. The negative V must never produce a success exit.
    row = {"system": "fosmc", "certified": True, "degree_used": 2,
           "weights_r": [1], "V_terms": [
               {"coefficient": coefficient, "exponents": ["2"], "term_types": ["abs"]}]}
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"results": [row]}))
    run = subprocess.run(
        [sys.executable, str(ROOT / "review/reverify_exact.py"), str(matrix),
         "--solver", "CLARABEL", "--printed-rationals", "--dump-grams"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert run.returncode == expected_exit, run.stdout + run.stderr
    artifacts = list(tmp_path.glob("reverify_exact_printed_*.json"))
    assert len(artifacts) == 1, run.stdout + run.stderr
    artifact = json.loads(artifacts[0].read_text())
    assert artifact["systems"][0]["exact_pass"] is (expected_exit == 0)
    assert artifacts[0].with_suffix(".md").is_file()
    if expected_exit == 0:
        assert (tmp_path / artifact["systems"][0]["grams_file"]).is_file()
