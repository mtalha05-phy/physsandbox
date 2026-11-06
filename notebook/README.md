# The live demo's lab notebook, pinned in git

These files let anyone check the live demo's lab notebook against something the server can't change after the fact: git history.

- `ed25519-public-key.pem`: the notebook's public key (key id `cd5618512f391bda`). Compare it with `GET /api/ledger/public-key`; if they differ, the server's key was replaced.
- `genesis.json`: the signed genesis record of the current notebook (started 2026-11-05T20:42:39Z, notebook id `b513c471499665b5`).
- `head-*.json`: signed heads of the notebook, committed from time to time. Each one is the count and last hash at that moment, signed with the key above.

Check the live notebook has not been rolled back or replaced since a committed head:

```bash
curl -s 'https://192-248-159-149.sslip.io/api/ledger?start=COUNT&limit=100' -o export.json   # COUNT from the head file
python tools/verify_receipts.py export.json notebook/ed25519-public-key.pem --expect-head notebook/head-2026-11-06T0359Z.json
```

Notebook restarts are recorded here. The notebook before this one used HMAC receipts (checkable only by the server) and was moved aside on the VM, not deleted, when the demo switched to Ed25519 on 2026-11-05.
