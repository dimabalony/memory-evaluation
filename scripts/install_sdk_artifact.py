"""Download the exact reviewed public SDK artifact, verify, then install it."""

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
receipt = json.loads((root / "sdk-artifact.json").read_text())
url = receipt["url"]
expected = receipt["sha256"]
if not url.startswith("https://github.com/dimabalony/recoleva/releases/download/v0.3.0/"):
    raise SystemExit("unreviewed SDK artifact URL")
if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
    raise SystemExit("SDK artifact checksum is required")
destination = root / "build" / "sdk" / "recoleva-0.3.0-py3-none-any.whl"
destination.parent.mkdir(parents=True, exist_ok=True)
payload = urllib.request.urlopen(url, timeout=120).read()
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit("SDK artifact checksum mismatch")
destination.write_bytes(payload)
subprocess.run([sys.executable, "-m", "pip", "install", str(destination)], check=True)
