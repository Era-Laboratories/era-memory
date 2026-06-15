"""
M0 gate: importing the library (base install) must pull ZERO third-party backends.

Run in a subprocess so the assertion is not polluted by other tests importing optional
deps. This enforces the non-goal: no mandatory cloud/GPU/private registry to run Tier 0.
"""

from __future__ import annotations

import subprocess
import sys

_CHECK = """
import sys
import era_memory.wiring        # composition root
import era_memory.memory        # facade
import era_memory.core.search   # core logic
import era_memory.core.pipeline
from era_memory.wiring import build_memory
build_memory(tier=0)            # Tier 0 must construct with no backends

forbidden = {
    "pymilvus", "asyncpg", "redis", "fastembed", "sqlite_vec",
    "httpx", "google", "pgvector", "cryptography", "pydantic", "fastapi",
}
loaded = sorted(forbidden & {m.split(".")[0] for m in sys.modules})
assert not loaded, f"base import loaded backends: {loaded}"
print("OK")
"""


def test_base_import_has_no_backend_deps():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
