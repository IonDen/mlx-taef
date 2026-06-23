"""Test helper: fetch the canonical taew2.1 checkpoint (GitHub-pinned, sha256-verified).

The canonical taew2.1 weights are published only on GitHub (madebyollin/taehv); community HF
mirrors were found to diverge from them. This is the ground-truth source for the conversion test
and the parity fixtures. The runtime package pins a byte-identical HF re-host (ionden/taew2.1);
this helper keeps the tests independent of that re-host.
"""

import hashlib
import urllib.request
from pathlib import Path

TAEW21_COMMIT = "a1c8e6a2ba77b91f284ef98935ec5bd21a41d786"
TAEW21_SHA256 = "04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f"
_URL = (
    f"https://raw.githubusercontent.com/madebyollin/taehv/{TAEW21_COMMIT}"
    "/safetensors/taew2_1.safetensors"
)
_CACHE = Path.home() / ".cache" / "mlx-taef-test" / "taew2_1.safetensors"


def canonical_taew21_path() -> Path:
    """Return a local path to the sha256-verified canonical taew2_1.safetensors (download once)."""
    if _CACHE.exists() and hashlib.sha256(_CACHE.read_bytes()).hexdigest() == TAEW21_SHA256:
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_URL, _CACHE)
    digest = hashlib.sha256(_CACHE.read_bytes()).hexdigest()
    if digest != TAEW21_SHA256:
        _CACHE.unlink(missing_ok=True)
        raise RuntimeError(f"canonical taew2_1 sha256 mismatch: got {digest}")
    return _CACHE
