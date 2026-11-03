# PhysSandbox

**Find out why an experiment gave the wrong answer.** Upload your data, and PhysSandbox calculates what you should have seen, ranks the likely causes and tells you what to test next. Every calculation runs in a locked-down sandbox with no network, so unpublished research data never leaves it.

Built for the **Vultr Agent Rush Hackathon** (lablab.ai), Challenge 1: *Blast Radius Zero*.

## The problem

When a measurement doesn't match the expected value, researchers lose hours guessing. Lab instructors say "nothing works the first time", and general chatbots can't run physics or keep data private. PhysSandbox starts with electricity and magnetism, and optics.

## How it works

1. **Upload** a CSV and say what you expected (e.g. a 10 mΩ resistor).
2. **Calculate** – deterministic physics tools run in the sandbox: expected signal, least-squares fit with uncertainties (lmfit), spectra (FFT).
3. **Rank causes** – each fault in the [fault library](faults/faults.yaml) predicts certain evidence; causes are scored by prior × likelihood.
4. **Recommend the next test** – the test with the most expected information gain (bits) per minute of bench time.
5. **Explain** – a model on Vultr Serverless Inference writes a short explanation. It only sees a schema summary and the tool results, never raw data, and it may only use numbers the tools produced.
6. **Upload the follow-up data** and the loop runs again.

## Vultr services used

| Service | What it does here |
| --- | --- |
| **Vultr Cloud Compute** | One VM runs the web app, the agent loop and every sandboxed job |
| **Vultr Serverless Inference** | OpenAI-compatible model that writes the plain-language explanation |
| **gVisor + Docker on the VM** | One throwaway container per analysis (`app/sandbox.py`) |

## Containment (Blast Radius Zero)

Every job runs with `--runtime=runsc` (gVisor), `--network none`, a read-only root, a 64 MB tmpfs, 512 MB memory, 1 CPU, 64 processes, all capabilities dropped, no environment variables, the upload mounted read-only, and a 30 s timeout. The container is destroyed afterwards. Every block, kill and denial goes to the **containment log** shown in the app.

| Attack (button in the app) | What contains it |
| --- | --- |
| Prompt injection in the CSV (`data/seeded/inject_1.csv`) | Comment lines and free text are dropped before the model sees anything; the diagnosis is unchanged |
| Send data out (`sandbox/attacks/egress.py`) | No network inside the sandbox → `blocked` |
| Runaway memory (`sandbox/attacks/runaway.py`) | Memory limit → `killed` |
| Read host secrets / tamper with tools (`sandbox/attacks/host_read.py`) | gVisor, empty environment, read-only root → `denied` |

Removing network egress breaks the "lethal trifecta" (private data + untrusted content + a way out).

## Evaluation on seeded faults

`python -m eval.run_eval` scores the pipeline on datasets with known faults (`data/manifest.yaml`):

| Dataset | Truth | Top cause |
| --- | --- | --- |
| r_offset_1 | thermoelectric offset | ✅ thermoelectric offset |
| r_offset_1_followup | thermoelectric offset | ✅ thermoelectric offset |
| r_clean | no fault | ✅ no fault |
| inject_1 (prompt injection) | thermoelectric offset | ✅ thermoelectric offset |
| scope_hum_1 | ground loop | ✅ ground loop |
| scope_hum_1_followup | no fault (hum gone) | ✅ no fault |
| scope_clean | no fault | ✅ no fault |

7/7 correct, no false alarms on the clean controls. These are synthetic datasets; we make no claim about accuracy on real lab data yet.

## Run it

**Locally (no isolation, development only):**

```bash
pip install -r requirements.txt
python data/generate.py
SANDBOX_MODE=local uvicorn app.main:app --reload
```

**On a Vultr VM (Ubuntu 24.04) with full containment:**

```bash
git clone https://github.com/<you>/physsandbox.git && cd physsandbox
cp .env.example .env    # add your Vultr Inference key and model
sudo bash deploy/setup_vultr.sh
```

## Repository layout

```
app/            FastAPI app: summariser, sandbox runner, ranking, explanation, safety
sandbox/        Dockerfile for the job image, physics tools, containment attack demos
faults/         Fault library and next-test catalogue (YAML)
data/           Seeded dataset generator and ground truth
eval/           Evaluation script
static/         Single-page web interface
deploy/         Vultr VM setup script
```

## Safety

Recommendations come from a fixed test catalogue with safety notes. Model text is checked against forbidden advice (e.g. lifting a safety earth, bypassing a laser interlock) before it is shown.

## License

MIT
