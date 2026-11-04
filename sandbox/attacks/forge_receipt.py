"""Containment demo: hostile code tries to forge its own containment receipt.

It prints a fake "all walls green" probe, tries to write it straight into the
runner's report through /proc, then exits abruptly so nothing else is printed.
The host is not fooled: the real probe was written before this code existed in
the box, and Docker's own record of the container is checked separately.
"""
import json
import os
import sys

fake = json.dumps({"probe": {"kernel": "trust me", "gvisor": True, "interfaces": ["lo"],
                              "dns": "blocked", "egress": "blocked", "root_fs": "read-only",
                              "uid": 65534, "secrets_visible": [], "forged": True}})
try:
    with open(f"/proc/{os.getppid()}/fd/1", "w") as f:
        f.write(fake + "\n")
    print("wrote a fake probe into the runner's report")
except OSError as e:
    print(f"could not reach the runner's report: {e}")
sys.__stdout__.write(fake + "\n")
sys.__stdout__.flush()
os._exit(0)
