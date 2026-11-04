# PhysSandbox

**A glovebox for AI-written science code.**

Chemists handle hazardous material through a glovebox: they can work on it, but it can never touch them. PhysSandbox does the same for the code an AI writes about your experiment. Every calculation, including code a model wrote after reading your file, runs in a sealed gVisor box with no network that is destroyed after each job, and every job leaves a signed receipt of the walls it actually had, measured from inside the box and checked again by the host.

What runs inside the glovebox is a physics troubleshooter. Upload a measurement from any of fourteen bench experiments that came out wrong; PhysSandbox finds the likely cause and tells you which bench test to run next.

Built for the **Vultr Agent Rush Hackathon** (lablab.ai), Challenge 1: *Blast Radius Zero*.

## Why it matters

- In physics teaching labs, "nothing works the first time": 14 of 20 electronics lab instructors interviewed said so in effect ([Dounas-Frazer & Lewandowski](https://arxiv.org/pdf/1606.05389)). In quantum-optics teaching labs, alignment can take weeks before any signal appears ([Borish & Lewandowski 2023](https://arxiv.org/pdf/2211.09595)).
- Researchers already paste data into chatbots, but a chatbot can't run physics, and unpublished data shouldn't leave the lab. Uploaded files can also carry instructions aimed at the AI (prompt injection).
- No product we found diagnoses why a physics bench measurement is off. AI-for-science money is going to discovery labs, and instrument copilots answer questions about the tool, not the experiment.

## Who pays

- **First buyer: university physics departments**, for their teaching labs. One department licence covers every bench and every student. The anchor is lab-teaching software such as Labster, from $5,000 a year per institution ([Labster](https://www.labster.com/platform-overview)). Our pricing is a hypothesis to test with instructors, not a validated number.
- **Why they would pay:** instructors get fewer "it doesn't work" queues, students learn the expert habit of predict, measure, compare and test, and the lab keeps a signed record of what was run. Unpublished data never leaves the box.
- **Next: R&D test benches.** In a vendor-sponsored survey of 305 R&D engineers, 97% reported project delays from test-equipment problems, with misconfiguration and lost calibration the leading causes ([Microwave Journal](https://microwavejournal.com/articles/34478-keysight-survey-reveals-true-cost-delay-from-test-equipment-misconfiguration-maintenance-and-training-issues)).
- **The platform underneath is not physics-specific.** The glovebox, receipts and signed notebook work for any AI-written science code; each new field is a fault library and a set of analyses (YAML plus Python).

## The experiment bench

Fourteen real teaching-lab and research-bench experiments across six areas of physics. Each has its own physics model (fitted inside the glovebox), a fault library of the things that really go wrong on that bench, and bench tests chosen by information per minute.

| Area | Experiment | Model | Faults it tells apart | Test that confirms the demo fault |
| --- | --- | --- | --- | --- |
| Electricity | Low-resistance measurement | V = R·I | thermal EMF, meter zero offset, gain error, lead resistance, self-heating | read V at zero current, leads on sample vs meter shorted |
| Electricity | RC charging curve | V = V₀(1 − e^(−t/RC)) | ×1 probe loading, leaky capacitor, capacitor tolerance, low supply | repeat with a ×10 probe |
| Signals | Oscilloscope trace | spectrum of V(t) | ground loop, magnetic pickup, aliasing | single-point grounding |
| Magnetism | Helmholtz coil calibration | B = (4/5)^(3/2) μ₀NI/R | Earth's field, Hall-probe zero offset, probe gain, coil spacing | read the probe in a zero-gauss chamber |
| Magnetism | Faraday's law of induction | ε = −N dΦ/dt | amplifier offset, ground offset, missing turns, flux-sensor error, reversed leads | hold the magnet still, short the amplifier input |
| Optics | Laser and photodetector alignment | Gaussian beam profile | misalignment, warm-up pointing drift, low power, detector saturation | repeat the scan after a 15-minute warm-up |
| Optics | Malus's law polariser | I = I₀ cos²θ | room light, worn polariser, analyser zero offset | read the crossed position with the beam blocked |
| Optics | Beer-Lambert absorption | I = I₀ e^(−αx) | stray light, detector dark offset, concentration, path length | rescan in the closed enclosure |
| Heat | Newton's law of cooling (thermocouple) | T = T_room + ΔT e^(−t/τ) | cold-junction error, wrong room temperature, poor thermal contact | ice-point check |
| Electricity | Diode I–V curve | V = nV_T ln(I/I_s) + I·R_s | voltmeter across the ammeter (burden), high bulk resistance, diode reversed | move the voltmeter onto the diode's legs |
| Magnetism | Hall effect | V_H = IB/(n e t) | contacts not opposite each other, field lower than set, thermoelectric contact voltage | reverse the field |
| Optics | Diffraction grating | d sin θ = mλ | wrong grating density, laser off its label, angle zero misread | repeat with a known reference line |
| Mechanics | Simple pendulum | T = 2π√(L/g) | release angle too large, timer error, length to the wrong point, miscounted swings | repeat at 5° amplitude |
| Mechanics | Hooke's law spring | mg = k·x | hanger mass left out, ruler zero misread, past the elastic limit | add the hanger's mass to every load |

The Helmholtz, laser-alignment, Faraday and Beer-Lambert experiments, and their CSV formats, come from the team's first prototype by [@mtalha05-phy](https://github.com/mtalha05-phy); his datasets run unchanged here, and his self-heating resistor dataset is part of the evaluation.

## What's new

1. **A closed troubleshooting loop.** PhysSandbox ranks the likely causes, picks the bench test with the most information per minute, then folds in only the *outcome* of the test you ran. On the demo data the leading cause goes from 64% to 92% and is confirmed.
2. **An agent whose code we assume is hostile.** A model on Vultr Serverless Inference writes Python to answer your questions. A static scan highlights risky lines, but nothing relies on it: the glovebox is the safeguard.
3. **Containment receipts the contained code cannot forge.** Every job probes its own walls (kernel, network interfaces, DNS, outbound connections, disk, user, visible secrets) *before any untrusted code exists in the box*, and writes that first. Agent code then runs in a separate, non-dumpable child process, so it can shape its own output but never the probe. The host then checks the container's real configuration and exit state with `docker inspect` (runtime, network mode, read-only root, memory, capabilities, user, OOM kill). The receipt carries SHA-256 hashes of the input, code, output and sandbox image. The **Forge its own receipt** button runs hostile code that tries to fake an all-green receipt; it is caught and logged.
4. **A signed, tamper-evident lab notebook.** Receipts are hash-chained and each is signed with HMAC-SHA256 using a key kept outside the repo and outside every sandbox. Editing, deleting or re-hashing any entry fails "Check for tampering", and every downloaded investigation record carries its receipts' hashes and signatures, so a copy held elsewhere pins that history.
5. **An agent that repairs its own code, sealed.** If the model's script fails in the box, the sandbox's error goes back to the model and it tries again (up to three attempts, each in a fresh container with its own receipt).

## Try it in 90 seconds

Open the demo and press **Take the 90-second tour**. It walks through a real investigation (resistor reads high → offset found → bench test → cause confirmed), a break-out attempt stopped by the network wall, a poisoned file whose hidden instruction reaches the model and is contained, and the tamper check on the lab notebook. Every step runs live in the sandbox.

Every investigation can be downloaded as a **record** for the lab notebook: each step's data hash, fitted values with uncertainties, causes, the recommended test, the sandbox receipt hash and the walls measured from inside the box.

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

The model never sees raw rows by default, only column names and ranges. Turn on "Let the agent read the raw file" with the poisoned sample and the hidden instruction reaches the model. In our tests `deepseek-v4.1-flash` on Vultr ignored it and wrote clean code. We don't rely on that: the **red team** switch substitutes a fixed script, the code a fooled model would write (read the file, send it to an outside server), and the glovebox runs it anyway. That script is a template, not model output, and the UI labels it as such. The network wall blocks it and lights up red.

**Try to break out** buttons run real attacks inside the box: send data to the internet (blocked), use all the memory (killed at 512 MB), read secrets and tamper with the tools (nothing to read, read-only disk), and forge its own containment receipt (ignored and flagged).

The public demo is also protected from its visitors: at most two sandboxes run at once, uploads are capped at 1 MB, and each visitor has a request limit, tightest on the endpoints that spend inference credit.

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

`python eval/run_eval.py` replays the whole troubleshooting loop on 57 cases with known faults (`data/manifest.yaml`): for every experiment, a healthy run, its common faults, and the follow-up result of the recommended test.

- **57/57 correct top cause**, run inside the real gVisor sandbox on the Vultr VM.
- **No false alarms**: all fourteen healthy datasets come back "no fault found", with no test suggested.
- **Every follow-up confirms**: in each experiment the recommended test lifts the leading cause from 52–75% to above 90%.
- **Prompt injection changes nothing**: the poisoned file gets the same diagnosis as its clean twin.

These are synthetic datasets we generated with known faults (`data/generate.py`, `data/generate_bench.py`), so the evaluation shows the loop works end to end, not accuracy on real lab data. The four prototype CSVs from @mtalha05-phy run unchanged. Real-bench data from teaching labs is the next evaluation.

## Run it

**On a Vultr VM (Ubuntu 24.04), full containment:**

```bash
git clone https://github.com/mukeshkbj/physsandbox.git && cd physsandbox
cp .env.example .env    # add your Vultr Inference API key and model
sudo bash deploy/setup_vultr.sh
sudo bash deploy/enable_https.sh   # HTTPS via Caddy and an sslip.io name, no domain needed
```

**Locally, no isolation (development only):**

```bash
pip install -r requirements.txt && python data/generate.py
SANDBOX_MODE=local uvicorn app.main:app --reload
```

## Repository layout

```
app/        web app: sandbox runner, agent, ranking, receipts ledger, explanation, safety;
            catalog.yaml describes the experiment bench the interface is built from
sandbox/    job image: runner.py (probe + run), physics models (analyze.py, experiments.py), break-out demos
faults/     fault library and next-test catalogue for all fourteen experiments (YAML)
data/       seeded dataset generator and ground truth
eval/       evaluation of the full troubleshooting loop
static/     single-page interface
deploy/     Vultr VM setup script
```

## Safety

Bench recommendations come from a fixed test catalogue with safety notes (never lift a safety earth; laser changes need your laser safety officer). Model text is checked against forbidden advice before it is shown.

## License

MIT
