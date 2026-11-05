# PhysSandbox

**A glovebox for AI-written science code.**

Chemists handle hazardous material through a glovebox: they can work on it, but it can never touch them. PhysSandbox does the same for the code an AI writes about your experiment. Every calculation, including code a model wrote after reading your file, runs in a sealed gVisor box with no network that is destroyed after each job, and every job leaves a signed receipt of the walls it actually had, measured from inside the box and checked again by the host.

What runs inside the glovebox is a physics troubleshooter. Upload a measurement from any of 31 bench experiments, first-year to graduate level, that came out wrong; PhysSandbox finds the likely cause and tells you which bench test to run next.

Built for the **Vultr Agent Rush Hackathon** (lablab.ai), Challenge 1: *Blast Radius Zero*.

**Live demo:** [192-248-159-149.sslip.io](https://192-248-159-149.sslip.io/) — press *Take the 90-second tour*. Running on a Vultr Cloud Compute VM with gVisor and Vultr Serverless Inference.

## Why it matters

- In physics teaching labs, "nothing works the first time": 14 of 20 electronics lab instructors interviewed said so in effect ([Dounas-Frazer & Lewandowski](https://arxiv.org/pdf/1606.05389)). In quantum-optics teaching labs, alignment can take weeks before any signal appears ([Borish & Lewandowski 2023](https://arxiv.org/pdf/2211.09595)).
- Researchers already paste data into chatbots, but a chatbot can't run physics, and unpublished data shouldn't leave the lab. Uploaded files can also carry instructions aimed at the AI (prompt injection).
- No product we found diagnoses why a physics bench measurement is off. AI-for-science money is going to discovery labs, and instrument copilots answer questions about the tool, not the experiment.

## Who pays

- **First buyer: university physics departments**, for their teaching labs. One department licence covers every bench and every student. The anchor is lab-teaching software such as Labster, from $5,000 a year per institution ([Labster](https://www.labster.com/platform-overview)). Our pricing is a hypothesis to test with instructors, not a validated number.
- **Why containment is what they pay for:** university guidance tells researchers not to put unpublished or confidential research data into generative AI tools (for example [Michigan State University](https://tech.msu.edu/news/2023/08/new-university-guidance-on-generative-ai-use) and the [University of Rochester](https://www.rochester.edu/university-research/updated-advisory-guidance-on-the-use-of-generative-ai-in-research)). A glovebox with no network, run in the department's own cloud account, plus a signed record of every job, is what lets a department say yes to AI in the lab at all. The troubleshooting is what students and instructors see: fewer "it doesn't work" queues and practice in the expert habit of predict, measure, compare and test.
- **Next: R&D test benches.** In a vendor-sponsored survey of 305 R&D engineers, 97% reported project delays from test-equipment problems, with misconfiguration and lost calibration the leading causes ([Microwave Journal](https://microwavejournal.com/articles/34478-keysight-survey-reveals-true-cost-delay-from-test-equipment-misconfiguration-maintenance-and-training-issues)).
- **The platform underneath is not physics-specific.** The glovebox, receipts and signed notebook work for any AI-written science code; each new field is a fault library and a set of analyses (YAML plus Python).

## The experiment bench

31 real teaching-lab and research-bench experiments across 10 areas of physics, from first-year labs to graduate advanced labs. Each has its own physics model (fitted inside the glovebox), a fault library of the things that really go wrong on that bench with a concrete fix for each, and bench tests chosen by information per minute. Every demo case is built so the first measurement is genuinely ambiguous between two causes, and one quick bench test tells them apart.

| Area | Experiment | Model | Faults it tells apart | Test that settles the demo case |
| --- | --- | --- | --- | --- |
| Electricity | Diode I–V curve | V = nV_T ln(I/I_s) + I·R_s | voltmeter across diode and ammeter (ammeter burden voltage); high series resistance in the diode itself (damaged or wrong part); diode connected the wrong way round | move the voltmeter leads onto the diode's own legs |
| Electricity | Low-resistance measurement | V = I·R | thermoelectric (thermal EMF) offset at a contact; voltmeter zero offset; calibration (gain) error in meter or current source; lead or contact resistance included (2-wire style error); sample self-heating at high current | read the voltmeter at zero current, leads on the sample, then shorted at the meter |
| Electricity | RC charging curve | V = V₀(1 − e^(−t/RC)) | oscilloscope probe loading the circuit (x1 probe, 1 MΩ); leaky capacitor; capacitor value outside its tolerance; supply voltage lower than set | repeat the measurement with a x10 (10 MΩ) probe |
| Signals | Oscilloscope trace | spectrum of V(t) | ground loop picking up mains hum; magnetic (inductive) pickup from a nearby transformer or cable loop; aliasing (sample rate too low for the signal) | connect all instruments to one ground point and re-record |
| Magnetism | Faraday's law of induction | ε = −N dΦ/dt | amplifier input offset voltage; ground potential difference between coil circuit and recorder; coil has fewer turns than specified, or a shorted winding; flux sensor calibration error; coil leads connected the wrong way round | hold the magnet still, then short the amplifier input |
| Magnetism | Hall effect | V_H = IB / (n e t) | hall contacts not exactly opposite each other; magnetic field lower than the value you set; thermoelectric voltage at the contacts | repeat with the magnetic field reversed |
| Magnetism | Helmholtz coil calibration | B = (4/5)^(3/2) μ₀NI/R | earth's field or nearby steel adding to the coil field; hall probe zero offset (probe not zeroed); hall probe gain (sensitivity) calibration error; coil spacing or radius not at the Helmholtz condition | read the probe at zero current, in the open and inside a zero-gauss chamber |
| Magnetism | Pulsed NMR spin echo *(graduate)* | A(2τ) = A₀ e^(−2τ/T₂) | RF pulses not 90° and 180° (pulse length mis-set); sample sitting partly outside the RF coil (wrong depth); molecules diffusing through a field gradient (magnet poorly shimmed); sample really relaxes faster (paramagnetic impurity or wrong dopant) | run a nutation sweep (FID amplitude against pulse length) |
| Optics | Beer-Lambert absorption | I = I₀ e^(−αx) | room light leaking into the detector; detector dark-current or amplifier offset; sample concentration higher than labelled; cuvette path length different from assumed | repeat the scan with the room lights off and the lid closed |
| Optics | Diffraction grating | d sin θ = mλ | spectrometer angle zero misread; a grating with a different line density than assumed; a laser wavelength different from its label | repeat with a source of known wavelength (HeNe 632.8 nm) |
| Optics | Laser and photodetector alignment | V = V_d + V_p e^(−2(x−x₀)²/w²) | detector or beam steering misaligned; laser pointing drift during warm-up; low laser power or a dirty optic in the path; photodetector saturating at the beam centre | repeat the scan after the laser has warmed up for 15 minutes |
| Optics | Malus's law polariser | I = I₀ cos²θ | room light reaching the detector; worn or low-extinction polariser; analyser scale not zeroed to the polariser axis | read the detector at the crossed position, then with the laser beam blocked |
| Optics | Michelson interferometer | N = 2d/λ | fringes slipping past uncounted (turning too fast); micrometer-to-mirror lever ratio out of calibration; backlash in the mirror screw (dead travel after reversing) | count the same travel again, slowly, turning one way only |
| Heat | Newton's law of cooling | T = T_room + ΔT e^(−t/τ) | thermocouple cold-junction compensation error; the surroundings aren't at the room temperature you assumed; poor thermal contact between thermocouple and sample | put the thermocouple in an ice-water bath for two minutes |
| Heat | Stefan–Boltzmann law (lamp filament) | P = εσA T⁴,  R/R₀ = (T/T₀)^1.209 | ammeter zero not set (reads current with none flowing); heat lost by conduction through the filament supports; voltage read at the supply, so the leads and ammeter count as lamp | read the ammeter with the lamp circuit open |
| Mechanics | Hooke's law spring | F = mg = k·x | hanger's mass left out of the loads; ruler zero read in the wrong place; spring loaded past its elastic limit; a spring with a different constant than expected | weigh the hanger and add its mass to every load |
| Mechanics | Simple pendulum | T = 2π√(L/g) | too large a release angle; timer running fast or slow; length measured to the top of the bob, not its centre; miscounted swings | repeat three lengths released from 5° or less |
| Waves & sound | Speed of sound (resonance tube) | L + e = v/4f, e ≈ 0.3 D | air in the tube not at the temperature entered; signal generator frequency dial reading off; lengths read from the wrong zero (not the tube's open end) | read the air temperature inside the tube |
| Quantum | Atomic spectral lines (hydrogen) | 1/λ = R(1/2² − 1/n²) | a line from another element mistaken for a hydrogen line; a line assigned the wrong upper level n; spectrometer wavelength calibration offset | remeasure the odd line with a fresh hydrogen tube |
| Quantum | Blackbody radiation | B(λ,T) ∝ λ⁻⁵ / (e^(hc/λkT) − 1) | spectrometer wavelength scale shifted; lamp running below its rated temperature (supply low); stray light or dark current adding a background | record a mercury calibration lamp on the same spectrometer |
| Quantum | Electron diffraction (de Broglie) | λ = h/√(2meV),  2d sin θ = λ,  D = 2L tan 2θ | foil-to-screen distance L wrong (another tube's figure, or measured from the wrong point); accelerating voltage reads off (supply meter gain error); faint third ring (d = 0.107 nm) measured as the outer ring | measure the foil-to-screen distance with a rule |
| Quantum | Franck–Hertz experiment (mercury) | peaks at V = V_c + n·E₁/e | oven too cool (mercury vapour too thin); retarding (counter) voltage set too low; current amplifier saturating (filament or gain too high); accelerating-voltage reading off scale (meter or divider) | read the oven thermometer for a few minutes |
| Quantum | Millikan oil drop | q = n·e | plate spacing wrong in the field E = V/d; plate voltage reads off (supply meter gain error); drops evaporating while being timed | measure the plate spacing with callipers |
| Quantum | Optical pumping of rubidium *(graduate)* | hf = g_F μ_B |k·I + B₀| | coil current read high or low (ammeter or sense resistor); coil constant wrong (turns, radius or spacing); lab field along the coil axis not nulled; coil current reading has a zero offset; ⁸⁵Rb and ⁸⁷Rb resonances swapped | read the coil current on a calibrated ammeter in series |
| Quantum | Photoelectric effect (h/e) | eV_s = hf − φ | light reaching the anode (reverse photocurrent); stopping-voltage meter reading low or high (gain error); a filter leaking a neighbouring, higher-frequency line | mask the anode so only the cathode is lit, and reread every line |
| Quantum | Planck's constant from LEDs | eV = hf − φ | voltmeter reading low or high (gain error); LED wavelengths taken from the datasheet, not measured; one LED isn't the colour on its label | read one LED's turn-on voltage on a calibrated reference meter too |
| Quantum | Quantum tunnelling simulation | T ≈ e^(−2κa), κ = √(2m(V₀−E))/ħ | particle mass wrong in the simulation (units or effective mass); barrier height or energy measured from the wrong zero; solver grid too coarse to resolve small transmissions | rerun the simulation at a second energy (0.25 eV) |
| Quantum | Zeeman effect (cadmium red line) *(graduate)* | Δν̃ = 2μ_B B/hc = (1/2nt)·δ/Δ | field reading off (gaussmeter on the wrong scale or not zeroed); étalon thickness or refractive index wrong in the analysis; rings of neighbouring orders paired wrongly at one field; one field reading wrong (gaussmeter range switched or value misread at that point) | measure the field in the gap with a calibrated gaussmeter |
| Nuclear & particle | Compton scattering *(graduate)* | E′ = E / (1 + (E/m_e c²)(1 − cos θ)) | MCA energy calibration gain off (PMT voltage or temperature drift); MCA energy calibration zero offset; detector arm's zero angle misread; backscatter peak taken for the scattered photopeak at one angle | read known calibration lines with the same MCA settings |
| Nuclear & particle | Radioactive decay (Geiger–Müller) | N = N₀ e^(−λt) + B, t½ = ln 2 / λ | room background not subtracted; long-lived activity in the sample (Cs-137 breakthrough from the generator); counter dead time losing counts at high rates | remove the source and count the room background |
| Condensed matter | Superconducting transition *(graduate)* | R(T) = R_n(T) / (1 + e^(−(T−Tc)/w)) | thermometer not at the sample's temperature (lag or gradient); measuring current too high (near the critical current); sample degraded (oxygen loss lowers Tc); thermal EMF at the voltage contacts (resistance never reaches zero) | record a warming sweep as well and compare it with the cooling sweep |

The Helmholtz, laser-alignment, Faraday and Beer-Lambert experiments, and their CSV formats, come from the team's first prototype by [@mtalha05-phy](https://github.com/mtalha05-phy); his datasets run unchanged here, and his self-heating resistor dataset is part of the evaluation.

## What's new

1. **A closed troubleshooting loop.** PhysSandbox ranks the likely causes, picks the bench test with the most information per minute, then folds in only the *outcome* of the test you ran. On the demo data the leading cause goes from 64% to 92% and is confirmed.
2. **An agent whose code we assume is hostile.** A model on Vultr Serverless Inference writes Python to answer your questions. A static scan highlights risky lines, but nothing relies on it: the glovebox is the safeguard.
3. **Containment receipts that don't take the code's word for it.** Every job probes its own walls (kernel, network interfaces, DNS, outbound connections, disk, user, visible secrets) *before any untrusted code runs* and writes that first; agent code is mounted read-only but not yet executed. Agent code then runs in a separate child process; the runner itself is made non-dumpable, so the child can't reach into the runner's report through `/proc`. The child can shape its own output but not the probe, and only short text fields of its output are kept (no payloads, capped length); the host also reads at most 2 MB from any container. The host then checks the container's real configuration and exit state with `docker inspect` (runtime, network mode, read-only root, memory, capabilities, user, OOM kill). Kills (memory, time) come only from Docker's record and the host's timer. What the code *tried* (reach the internet, write to disk, print a fake receipt) comes from the code's own output and is labelled that way; the wall that stopped it is the part the host verifies. The **Forge its own receipt** button runs code that prints a fake all-green receipt and tries to write into the runner's report; it can't succeed by design, and the attempt is flagged.
4. **A signed, tamper-evident lab notebook.** Receipts are hash-chained and every one is signed with HMAC-SHA256; a signed head record (count and last hash) is rewritten on each append. Stripping signatures, editing an entry and rebuilding the hashes, or deleting entries off the end all fail "Check for tampering". **Tamper with a copy** shows it: it edits a copy of the notebook the way an attacker without the key would, and the check catches it. The key sits outside the repository and every sandbox (`~/.physsandbox/ledger.key`, mode 0600) but on the same VM, so root on the host could still forge; for that threat, every downloaded investigation record carries its receipts' hashes and signatures to keep elsewhere.
5. **An agent that repairs its own code, sealed.** If the model's script fails in the box, the sandbox's error goes back to the model and it tries again (up to three attempts, each in a fresh container with its own receipt).

## Try it in 90 seconds

Open the demo and press **Take the 90-second tour**. It walks through a real investigation (resistor reads high → offset found → bench test → cause confirmed), a break-out attempt stopped by the network wall, a poisoned file whose hidden instruction reaches the model and is contained, and the tamper check on the lab notebook. Every step runs live in the sandbox.

Every investigation can be downloaded as a **PDF report** in one click: the outcome, the fix for the confirmed cause, and for every step the data chart with its fit, the measured values with uncertainties, the likely causes, the recommended test with how to do it and its safety note, plus each sandbox receipt with its signature, checked against the server's own lab notebook. A markdown record is available too.

Adding an experiment group needs no edits to existing code: a physics module (`sandbox/tools/bench_<group>.py`), a fault library (`faults/bench_<group>.yaml`), a catalogue file (`app/catalog.d/<group>.yaml`), a data generator (`data/gen_<group>.py`) and its evaluation cases (`data/manifest.d/<group>.yaml`). Five of the groups were built that way in parallel.

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

The model never sees raw rows by default, only column names and ranges. When its code fails, the repair loop sends back only the exception's type and line number, never the message or output, because a failed parse can echo a line of the file. Turn on "Let the agent read the raw file" with the poisoned sample and the hidden instruction reaches the model. The tour does this with the live model first: in our tests `deepseek-v4.1-flash` on Vultr ignored the instruction and wrote clean code. We don't rely on that: the **red team** switch substitutes a fixed script, the code a fooled model would write (read the file, send it to an outside server), and the glovebox runs it anyway. That script is a template, not model output, and the UI labels it as such. The network wall blocks it and lights up red.

**Try to break out** buttons run real attacks inside the box: send data to the internet (blocked), use all the memory (killed at 512 MB), read secrets and tamper with the tools (nothing to read, read-only disk), and forge its own containment receipt (ignored and flagged).

The public demo is also protected from its visitors: at most two sandboxes run at once (jobs run in worker threads, so one slow job never freezes the site), uploads are capped at 1 MB, and each visitor has a request limit, tightest on the endpoints that spend inference credit.

**Development mode** (`SANDBOX_MODE=local`, for working on the physics) has no isolation, so it refuses the break-out demos and agent-written code outright, and every receipt, record and report made in it says "not isolated".

**Known limits.** The app runs as root on the VM because it drives Docker, and Docker access is root-equivalent; a dedicated non-root user in the `docker` group would not change that, so the boundary we rely on is gVisor plus the container flags. Network attempts are detected from the code's own error messages; logging `connect` calls in gVisor (`runsc --strace`) would give host-side evidence of the attempt as well.

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

`python eval/run_eval.py` replays the whole troubleshooting loop on 125 cases with known faults (`data/manifest.yaml` and `data/manifest.d/`): for every experiment, a healthy run, its common faults, and the follow-up result of the recommended test.

- **125/125 correct top cause** across all 31 experiments.
- **No false alarms**: all 31 healthy datasets come back "no fault found", with no test suggested.
- **Every follow-up confirms**: in each experiment the recommended test is the one that tells the two leading causes apart, and its result lifts the leading cause from about 52–65% to above 90%.
- **Not tuned to one lucky dataset**: `eval/run_seeds.sh` regenerates every dataset with ten other random seeds and replays the whole evaluation for each.
- **The diagnosis can't be steered by the file**: the physics parser skips comment lines and the model plays no part in the diagnosis, so the poisoned file gets the same result as its clean twin. An injected instruction can only reach the agent box, where the glovebox contains whatever the model writes.

How the demo cases are built: in each demo the two leading faults predict the same first measurement, by design, so the first split simply reflects their priors (how often each happens on that bench, set by hand); the recommended test is the one whose outcomes the two faults predict differently. That is the situation the tool is for: data that can't separate two causes, and one quick test that can.

The likelihoods in the fault libraries (`faults/*.yaml`) are hand-set from bench experience, not calibrated on data, so the percentages show how strongly the evidence points within that model, not measured frequencies. To check the rules aren't tuned to one lucky dataset, `eval/run_seeds.sh` regenerates every dataset with ten other random seeds (results above). Because the data come from the same models the tool fits, this shows the loop is self-consistent and robust to noise, not that the fault rules match real benches; that needs real lab data.

These are synthetic datasets we generated with known faults (`data/generate*.py`, `data/gen_*.py`), so the evaluation shows the loop works end to end, not accuracy on real lab data. The four prototype CSVs from @mtalha05-phy run unchanged. Real-bench data from teaching labs is the next evaluation.

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
app/        web app: sandbox runner, agent, ranking, receipts ledger, explanation, safety, PDF report;
            catalog.yaml and catalog.d/*.yaml describe the experiment bench the interface is built from
sandbox/    job image: runner.py (probe + run), physics models (analyze.py, experiments.py, bench_*.py), break-out demos
faults/     fault libraries, next-test catalogue and a fix for every fault (YAML; bench_<group>.yaml per experiment group)
data/       seeded dataset generators (generate*.py, gen_*.py) and ground truth (manifest.yaml, manifest.d/)
eval/       evaluation of the full troubleshooting loop
static/     single-page interface
deploy/     Vultr VM setup script
```

## Safety

Bench recommendations come from a fixed test catalogue with safety notes (never lift a safety earth; laser changes need your laser safety officer). Model text is checked against forbidden advice before it is shown.

## License

MIT
