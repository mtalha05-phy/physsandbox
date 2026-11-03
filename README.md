# PhysSandbox

**A glovebox for AI-written science code.**

Chemists handle hazardous material through a glovebox: they can work on it, but it can never touch them. PhysSandbox does the same for the code an AI writes about your experiment. Upload a measurement that came out wrong; PhysSandbox finds the cause, tells you which bench test to run next, and runs every calculation, including code a model wrote after reading your file, in a sealed box with no network that is destroyed after each job.

Built for the **Vultr Agent Rush Hackathon** (lablab.ai), Challenge 1: *Blast Radius Zero*.

## Why it matters

- In physics teaching labs, "nothing works the first time": 14 of 20 electronics lab instructors said exactly that. Students and researchers lose hours guessing at causes.
- Researchers already paste data into chatbots, but a chatbot can't run physics, and unpublished data shouldn't leave the lab. Uploaded files can also carry instructions aimed at the AI (prompt injection).
- No product we found diagnoses why an electricity or optics measurement is off. AI-for-science money is going to biopharma and materials.

## What's new

1. **A closed troubleshooting loop.** PhysSandbox ranks the likely causes, picks the bench test with the most information per minute, then folds in only the *outcome* of the test you ran. On the demo data the leading cause goes from 64% to 92% and is confirmed.
2. **An agent whose code we assume is hostile.** A model on Vultr Serverless Inference writes Python to answer your questions. A static scan highlights risky lines, but nothing relies on it: the glovebox is the safeguard.
3. **Containment receipts measured from inside the box.** Every job probes its own walls (kernel, network interfaces, DNS, outbound connections, disk, user, visible secrets) and the host issues a receipt with SHA-256 hashes of the input, code, output and sandbox image.
4. **A tamper-evident lab notebook.** Receipts are hash-chained: editing or deleting any entry breaks every hash after it, and "Check for tampering" proves the record is intact. That's an audit trail a methods section or a lab safety officer can rely on.

## The glovebox (Blast Radius Zero)

Every job, physics or agent code, runs in a fresh container with:

| Wall | Setting |
| --- | --- |
| Kernel | gVisor (`runsc`): system calls hit a user-space kernel, not the host |
| Network | `--network none`: no interface except loopback, no DNS, no outbound connections |
| Disk | read-only root, 64 MB `noexec` scratch, input mounted read-only |
| Resources | 512 MB memory, 1 CPU, 64 processes, 30 s wall clock |
| Identity | uid 65534 (nobody), all capabilities dropped, `no-new-privileges` |
| Secrets | no host environment passed in; API keys stay outside |

The model never sees raw rows by default, only column names and ranges. Turn on "Let the agent read the raw file" with the poisoned sample to watch a hidden instruction reach the model, the model write code that tries to send your data out, and the glovebox block it. The wall that stopped it lights up red.

**Try to break out** buttons run real attacks inside the box: send data to the internet (blocked), use all the memory (killed at 512 MB), read secrets and tamper with the tools (nothing to read, read-only disk).

## Vultr services used

| Service | Role |
| --- | --- |
| **Vultr Cloud Compute** | One VM (London) runs the web app, the agent loop, gVisor and every sandboxed job |
| **Vultr Serverless Inference** | OpenAI-compatible model that writes the analysis code and the plain-language explanation |

## Architecture

```
Browser ──upload──▶ FastAPI app (Vultr VM)
                      │  schema summary only (no raw rows by default)
                      ├──────────────▶ Vultr Serverless Inference ── code + wording
                      │
                      ├─ run job ─▶ ┌─ gVisor container, no network, read-only ─┐
                      │             │ runner.py: probe walls ▸ run physics/code   │
                      │             └─────────────── destroyed ──────────────────┘
                      ├─ rank causes (fault library, Bayes) ▸ next test (bits/min)
                      └─ receipt ▸ hash-chained lab notebook ▸ containment log
```

## Evaluation on seeded faults

`python eval/run_eval.py` replays the loop on datasets with known faults (`data/manifest.yaml`):

| Case | Truth | Result |
| --- | --- | --- |
| Resistor reads high | thermoelectric offset | ✅ 64%, recommends the zero-current test |
| + zero-current test result | thermoelectric offset | ✅ **92%, confirmed** |
| Reversed-current sweep | thermoelectric offset | ✅ |
| Healthy resistor | no fault | ✅ clear, no test suggested |
| Poisoned file (prompt injection) | thermoelectric offset | ✅ same diagnosis as clean twin |
| Scope trace with hum | ground loop | ✅ 75%, recommends single-point grounding |
| + grounding test result | ground loop | ✅ **92%, confirmed** |
| Clean trace | no fault | ✅ clear |

8/8 correct, run inside the real gVisor sandbox on the Vultr VM. These are synthetic datasets; we make no claim about accuracy on real lab data yet.

## Run it

**On a Vultr VM (Ubuntu 24.04), full containment:**

```bash
git clone https://github.com/mukeshkbj/physsandbox.git && cd physsandbox
cp .env.example .env    # add your Vultr Inference API key and model
sudo bash deploy/setup_vultr.sh
```

**Locally, no isolation (development only):**

```bash
pip install -r requirements.txt && python data/generate.py
SANDBOX_MODE=local uvicorn app.main:app --reload
```

## Repository layout

```
app/        web app: sandbox runner, agent, ranking, receipts ledger, explanation, safety
sandbox/    job image: runner.py (probe + run), physics tools, break-out attack demos
faults/     fault library and next-test catalogue (YAML)
data/       seeded dataset generator and ground truth
eval/       evaluation of the full troubleshooting loop
static/     single-page interface
deploy/     Vultr VM setup script
```

## Safety

Bench recommendations come from a fixed test catalogue with safety notes (never lift a safety earth; laser changes need your laser safety officer). Model text is checked against forbidden advice before it is shown.

## License

MIT
