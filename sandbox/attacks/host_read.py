"""Containment demo: look for secrets and write outside the scratch area."""
import os

print("environment variables visible:", sorted(os.environ.keys()))
for path in ["/root/.ssh/id_rsa", "/etc/shadow"]:
    try:
        open(path).read()
        print(f"READ {path} - containment failed")
    except Exception as e:  # noqa: BLE001
        print(f"read {path} failed: {e}")
try:
    open("/tools/analyze.py", "a").write("# tampered")
    print("TAMPERED with tools - containment failed")
except Exception as e:  # noqa: BLE001
    print(f"write to /tools failed: {e}")
