# PhysSandbox

**A glovebox for AI-written science code.**

Chemists handle hazardous material through a glovebox: they can work on it, but it can never touch them. PhysSandbox does the same for the code an AI writes about your experiment. Every calculation, including code a model wrote after reading your file, runs in a sealed gVisor box with no network that is destroyed after each job, and every job leaves a signed receipt of the walls it actually had, measured from inside the box and checked again by the host.

What runs inside the glovebox is a physics troubleshooter. Upload a measurement from any of 51 bench experiments, first-year to graduate level, that came out wrong; PhysSandbox finds the likely cause and tells you which bench test to run next.

Built for the **Vultr Agent Rush Hackathon** (lablab.ai), Challenge 1: *Blast Radius Zero*.

**Live demo:** [192-248-159-149.sslip.io](https://192-248-159-149.sslip.io/) — press *Take the 90-second tour*. Running on a Vultr Cloud Compute VM with gVisor and Vultr Serverless Inference.

## Why it matters

- In physics teaching labs, "nothing works the first time": 14 of 20 electronics lab instructors interviewed said so in effect ([Dounas-Frazer & Lewandowski](https://arxiv.org/pdf/1606.05389)). In quantum-optics teaching labs, alignment can take weeks before any signal appears ([Borish & Lewandowski 2023](https://arxiv.org/pdf/2211.09595)).
- Researchers already paste data into chatbots, but a chatbot can't run physics, and unpublished data shouldn't leave the lab. Uploaded files can also carry instructions aimed at the AI (prompt injection).
- No product we found diagnoses why a physics bench measurement is off. AI-for-science money is going to discovery labs, and instrument copilots answer questions about the tool, not the experiment.

## Who pays

- **First buyer: university physics departments**, for their teaching labs. One department licence covers every bench and every student. The anchor is lab-teaching software such as Labster, from $5,000 a year per institution ([Labster](https://www.labster.com/platform-overview)). Our pricing is a hypothesis to test with instructors, not a validated number.
- **Why containment is what they pay for:** university guidance tells researchers not to put unpublished or confidential research data into generative AI tools (for example [Michigan State University](https://tech.msu.edu/news/2023/08/new-university-guidance-on-generative-ai-use) and the [University of Rochester](https://www.rochester.edu/university-research/updated-advisory-guidance-on-the-use-of-generative-ai-in-research)). The same applies to teaching labs: a student's lab data and write-up are coursework the university must protect as a student record (FERPA in the US, GDPR in the UK and EU), and the graduate advanced-lab and research-bench experiments here (NMR, superconductors, Compton, muon lifetime) produce unpublished data. A glovebox with no network, run in the department's own cloud account, plus a signed record of every job, is what lets a department say yes to AI in the lab at all. The troubleshooting is what students and instructors see: fewer "it doesn't work" queues and practice in the expert habit of predict, measure, compare and test.
- **Next: R&D test benches.** In a vendor-sponsored survey of 305 R&D engineers, 97% reported project delays from test-equipment problems, with misconfiguration and lost calibration the leading causes ([Microwave Journal](https://microwavejournal.com/articles/34478-keysight-survey-reveals-true-cost-delay-from-test-equipment-misconfiguration-maintenance-and-training-issues)).
- **Your own files, not just ours.** Upload any CSV: if its columns have other names (`Current (A)`, `V_mV`), the page asks which is which and renames the header in the browser; the numbers are sent exactly as recorded.
- **The platform underneath is not physics-specific.** The glovebox, receipts and signed notebook work for any AI-written science code; each new field is a fault library and a set of analyses (YAML plus Python).

## The experiment bench

51 real teaching-lab and research-bench experiments across 10 areas of physics, from first-year labs to graduate advanced labs. Each has its own physics model (fitted inside the glovebox), a fault library of the things that really go wrong on that bench with a concrete fix for each, and bench tests chosen by information per minute. Every demo case is built so the first measurement is genuinely ambiguous between two causes, and one quick bench test tells them apart.

| Area | Experiment | Model | Faults it tells apart | Test that settles the demo case |
| --- | --- | --- | --- | --- |
| Electricity | Diode I–V curve | V = nV_T ln(I/I_s) + I·R_s | voltmeter across diode and ammeter (ammeter burden voltage); high series resistance in the diode itself (damaged or wrong part); diode connected the wrong way round | move the voltmeter leads onto the diode's own legs |
| Electricity | Low-resistance measurement | V = I·R | thermoelectric (thermal EMF) offset at a contact; voltmeter zero offset; calibration (gain) error in meter or current source; lead or contact resistance included (2-wire style error); sample self-heating at high current | read the voltmeter at zero current, leads on the sample, then shorted at the meter |
| Electricity | RC charging curve | V = V₀(1 − e^(−t/RC)) | oscilloscope probe loading the circuit (x1 probe, 1 MΩ); leaky capacitor; capacitor value outside its tolerance; supply voltage lower than set | repeat the measurement with a x10 (10 MΩ) probe |
| Electricity | Series RLC resonance | V_R = V R / √((R + R_L)² + (ωL − 1/ωC)²) | generator's 50 Ω output impedance pulling the drive down near resonance (input not held constant); coil resistance left out of the sum, or higher at f₀ than its DC reading; capacitor or inductor not at its marked value; AC voltmeter reading low at high frequency (its bandwidth is too small); coil losses rising with frequency (iron or ferrite core) | watch the circuit's input on CH1 through the sweep |
| Signals | Oscilloscope trace | spectrum of V(t) | ground loop picking up mains hum; magnetic (inductive) pickup from a nearby transformer or cable loop; aliasing (sample rate too low for the signal) | connect all instruments to one ground point and re-record |
| Magnetism | e/m of the electron (fine-beam tube) | e/m = 2V/(B r)²,  B = (4/5)^(3/2) μ₀NI/R | beam diameter read with parallax (scale not in the plane of the beam); coil constant wrong (turns, radius or spacing of the Helmholtz pair); Earth's field along the coil axis not cancelled; voltmeter reads the supply, not the anode–cathode voltage (Wehnelt bias counted in) | re-read the diameters on a mirror scale |
| Magnetism | Electron spin resonance (DPPH) *(graduate)* | h f = g μ_B B | field zero offset (Hall probe not zeroed, or a residual field along the coils); resonance read off the edge of the line (modulation phase not set); field calibration off (coil constant or gaussmeter range); RF frequency misread at one point (counter digits or range) | reverse the field and read the resonances again |
| Magnetism | Faraday's law of induction | ε = −N dΦ/dt | amplifier input offset voltage; ground potential difference between coil circuit and recorder; coil has fewer turns than specified, or a shorted winding; flux sensor calibration error; coil leads connected the wrong way round | hold the magnet still, then short the amplifier input |
| Magnetism | Hall effect | V_H = IB / (n e t) | Hall contacts not exactly opposite each other; magnetic field lower than the value you set; thermoelectric voltage at the contacts | repeat with the magnetic field reversed |
| Magnetism | Helmholtz coil calibration | B = (4/5)^(3/2) μ₀NI/R | Earth's field or nearby steel adding to the coil field; Hall probe zero offset (probe not zeroed); Hall probe gain (sensitivity) calibration error; coil spacing or radius not at the Helmholtz condition | read the probe at zero current, in the open and inside a zero-gauss chamber |
| Magnetism | Magnetic hysteresis (B–H loop) | H = N₁I/ℓ,  B = Φ/(N₂A) | search coil's turns not as stated; core cross-section wrong (outside size used, tape and lamination gaps included); fluxmeter (integrator) drifting during the loop; fluxmeter zeroed with the core still magnetised (not demagnetised first) | compare the search coil with a freshly wound 10-turn coil |
| Magnetism | Pulsed NMR spin echo *(graduate)* | A(2τ) = A₀ e^(−2τ/T₂) | RF pulses not 90° and 180° (pulse length mis-set); sample sitting partly outside the RF coil (wrong depth); molecules diffusing through a field gradient (magnet poorly shimmed); sample really relaxes faster (paramagnetic impurity or wrong dopant) | run a nutation sweep (FID amplitude against pulse length) |
| Optics | Beer-Lambert absorption | I = I₀ e^(−αx) | room light leaking into the detector; detector dark-current or amplifier offset; sample concentration higher than labelled; cuvette path length different from assumed | repeat the scan with the room lights off and the lid closed |
| Optics | Brewster's angle (Fresnel reflection) | R_p = \|r_p\|²,  tan θ_B = n | polariser not set exactly for p (some s-polarised light in the beam); polariser's extinction too poor (it passes some of the crossed polarisation); rotation stage's zero not at normal incidence (every angle offset); glass not the type entered (its refractive index differs); back-face reflection reaching the detector too (a second spot beside the first); detector saturating at the brightest angles; reflected beam walking off the detector at large angles (plate's face off the stage's axis) | at the minimum, turn the polariser either side of its setting |
| Optics | Diffraction grating | d sin θ = mλ | spectrometer angle zero misread; a grating with a different line density than assumed; a laser wavelength different from its label | repeat with a source of known wavelength (HeNe 632.8 nm) |
| Optics | Faraday rotation *(graduate)* | θ = V·B·L | field not uniform along the rod (rod off-centre or sticking out of the coil); field reading high or low (Hall probe or gaussmeter calibration); analyser zero misread (extinction at zero field set wrong); gaussmeter not zeroed (field reads non-zero at zero current) | map the field along the rod with the Hall probe |
| Optics | Focal length of a thin lens | 1/u + 1/v = 1/f | lens position read at the holder's index mark, which is off the lens itself; thick or compound lens (its principal planes are not at the middle of its mount); image judged sharp by eye in one go (the depth of focus spreads the image distances); wrong lens picked up (its focal length is not the one entered) | check the holder's index mark against the lens itself |
| Optics | Laser and photodetector alignment | V = V_d + V_p e^(−2(x−x₀)²/w²) | detector or beam steering misaligned; laser pointing drift during warm-up; low laser power or a dirty optic in the path; photodetector saturating at the beam centre | repeat the scan after the laser has warmed up for 15 minutes |
| Optics | Malus's law polariser | I = I₀ cos²θ | room light reaching the detector; worn or low-extinction polariser; analyser scale not zeroed to the polariser axis | read the detector at the crossed position, then with the laser beam blocked |
| Optics | Michelson interferometer | N = 2d/λ | fringes slipping past uncounted (turning too fast); micrometer-to-mirror lever ratio out of calibration; backlash in the mirror screw (dead travel after reversing) | count the same travel again, slowly, turning one way only |
| Optics | Young's double slit | y_m = m λ L / d | slit-to-screen distance L measured from the wrong point (the laser or the slide holder, not the slits); slit separation not the value on the slide's label; fringes past the envelope's dark band numbered wrongly (a hidden order skipped, or the dark band counted as a fringe) | measure the slit-to-screen distance again with a steel tape, from the slide itself |
| Heat | Newton's law of cooling | T = T_room + ΔT e^(−t/τ) | thermocouple cold-junction compensation error; the surroundings aren't at the room temperature you assumed; poor thermal contact between thermocouple and sample | put the thermocouple in an ice-water bath for two minutes |
| Heat | Stefan–Boltzmann law (lamp filament) | P = εσA T⁴,  R/R₀ = (T/T₀)^1.209 | ammeter zero not set (reads current with none flowing); heat lost by conduction through the filament supports; voltage read at the supply, so the leads and ammeter count as lamp | read the ammeter with the lamp circuit open |
| Heat | Thermal conductivity (Lee's disc) | k = m c (dT/dt)\|T₂ · x / (A (T₁ − T₂)) | the two thermometers disagree (calibration offset between them); steady state not reached (disc still warming when T₂ was read); disc cooled with its top face bare (sample not put back on top for the cooling run) | read both thermometers side by side in the steam chamber |
| Heat | Thermocouple calibration (type K) | emf = S(T − T_ref) + q(T − T_ref)²,  S ≈ 41 µV/K | reference junction not at 0 °C (ice melted, or the junction out of the ice); voltmeter zero offset on its millivolt range; wrong thermocouple type (a type J, iron–constantan, read as type K); thermocouple legs connected the wrong way round (the emf comes out negative); hot junction not in good thermal contact with the bath (reads low at high temperature) | read a thermometer in the reference flask |
| Mechanics | Boyle's law (syringe and pressure sensor) | p (V + V₀) = nRT | tubing volume left out of the dead volume; air leaking out at the plunger seal or a joint; compressed too fast, so the gas is warm when read; pressure sensor zero offset | measure the connecting tube's length and bore |
| Mechanics | Cavendish torsion balance *(graduate)* | G = 2π² L r² θ / (M T²) ÷ (1 − β),  θ = ΔS / 4D | mirror-to-scale distance D not what was entered; large spheres not against the case (centre distance r larger than assumed); equilibrium drifting (room temperature or fibre creep); air currents in the case (spot jittering more than the reading precision) | measure the mirror-to-scale distance with a steel tape |
| Mechanics | Driven damped oscillator (resonance) | A = a ω₀² / √((ω₀² − ω²)² + (γω)²) | drive frequency read from the motor's dial, which is off; oscillating mass not the one f₀ was timed with (vane or slotted mass added); eddy-current brake current not the one the damping was measured at; amplitudes read before the transient had died away; amplitude limited by an end stop at the top of the curve | time 20 crank turns with a stopwatch at three dial settings |
| Mechanics | Hooke's law spring | F = mg = k·x | hanger's mass left out of the loads; ruler zero read in the wrong place; spring loaded past its elastic limit; a spring with a different constant than expected | weigh the hanger and add its mass to every load |
| Mechanics | Kater's reversible pendulum | T₁ = T₂ = T at the crossing,  g = 4π² L / T² | swung from a larger angle than the amplitude entered; knife-edge separation L measured wrongly; swings miscounted at one slider position; one knife edge blunt or slipping on its plate; too few swings timed for the precision entered | read the swing amplitude on a protractor behind the bar |
| Mechanics | Simple pendulum | T = 2π√(L/g) | too large a release angle; timer running fast or slow; length measured to the top of the bob, not its centre; miscounted swings | repeat three lengths released from 5° or less |
| Mechanics | Viscosity by Stokes' law (falling ball) | v = 2r²g(ρ_s − ρ_l) / 9η, corrected for the tube wall | glycerol not at the temperature entered; glycerol has absorbed water from the air; ball diameters read on a micrometer that wasn't zeroed; air bubbles clinging to the smallest balls (dropped in dry) | read the glycerol's temperature with a thermometer in the tube |
| Waves & sound | Speed of sound (resonance tube) | L + e = v/4f, e ≈ 0.3 D | air in the tube not at the temperature entered; signal generator frequency dial reading off; lengths read from the wrong zero (not the tube's open end) | read the air temperature inside the tube |
| Quantum | Atomic spectral lines (hydrogen) | 1/λ = R(1/2² − 1/n²) | a line from another element mistaken for a hydrogen line; a line assigned the wrong upper level n; spectrometer wavelength calibration offset | remeasure the odd line with a fresh hydrogen tube |
| Quantum | Blackbody radiation | B(λ,T) ∝ λ⁻⁵ / (e^(hc/λkT) − 1) | spectrometer wavelength scale shifted; lamp running below its rated temperature (supply low); stray light or dark current adding a background | record a mercury calibration lamp on the same spectrometer |
| Quantum | Electron diffraction (de Broglie) | λ = h/√(2meV),  2d sin θ = λ,  D = 2L tan 2θ | foil-to-screen distance L wrong (another tube's figure, or measured from the wrong point); accelerating voltage reads off (supply meter gain error); faint third ring (d = 0.107 nm) measured as the outer ring | measure the foil-to-screen distance with a rule |
| Quantum | Franck–Hertz experiment (mercury) | peaks at V = V_c + n·E₁/e | oven too cool (mercury vapour too thin); retarding (counter) voltage set too low; current amplifier saturating (filament or gain too high); accelerating-voltage reading off scale (meter or divider) | read the oven thermometer for a few minutes |
| Quantum | Millikan oil drop | q = n·e | plate spacing wrong in the field E = V/d; plate voltage reads off (supply meter gain error); drops evaporating while being timed | measure the plate spacing with callipers |
| Quantum | Optical pumping of rubidium *(graduate)* | hf = g_F μ_B \|k·I + B₀\| | coil current read high or low (ammeter or sense resistor); coil constant wrong (turns, radius or spacing); lab field along the coil axis not nulled; coil current reading has a zero offset; ⁸⁵Rb and ⁸⁷Rb resonances swapped | read the coil current on a calibrated ammeter in series |
| Quantum | Photoelectric effect (h/e) | eV_s = hf − φ | light reaching the anode (reverse photocurrent); stopping-voltage meter reading low or high (gain error); a filter leaking a neighbouring, higher-frequency line | mask the anode so only the cathode is lit, and reread every line |
| Quantum | Planck's constant from LEDs | eV = hf − φ | voltmeter reading low or high (gain error); LED wavelengths taken from the datasheet, not measured; one LED isn't the colour on its label | read one LED's turn-on voltage on a calibrated reference meter too |
| Quantum | Quantum tunnelling simulation | T ≈ e^(−2κa), κ = √(2m(V₀−E))/ħ | particle mass wrong in the simulation (units or effective mass); barrier height or energy measured from the wrong zero; solver grid too coarse to resolve small transmissions | rerun the simulation at a second energy (0.25 eV) |
| Quantum | Zeeman effect (cadmium red line) *(graduate)* | Δν̃ = 2μ_B B/hc = (1/2nt)·δ/Δ | field reading off (gaussmeter on the wrong scale or not zeroed); étalon thickness or refractive index wrong in the analysis; rings of neighbouring orders paired wrongly at one field; one field reading wrong (gaussmeter range switched or value misread at that point) | measure the field in the gap with a calibrated gaussmeter |
| Nuclear & particle | Compton scattering *(graduate)* | E′ = E / (1 + (E/m_e c²)(1 − cos θ)) | MCA energy calibration gain off (PMT voltage or temperature drift); MCA energy calibration zero offset; detector arm's zero angle misread; backscatter peak taken for the scattered photopeak at one angle | read known calibration lines with the same MCA settings |
| Nuclear & particle | Gamma-ray attenuation in lead | N = N₀ e^(−μx),  μ = (μ/ρ)·ρ | background during the run higher than the figure subtracted; gamma rays reaching the tube round the lead (no collimator or tube shield); counting losses at high rate (detector dead time) behind the thinnest absorbers; absorber thicknesses not as recorded (nominal sheet sizes used); photons scattered inside the lead still counted (broad beam) | count with the source removed |
| Nuclear & particle | Muon lifetime *(graduate)* | N(t) = A e^(−t/τ) + B | light leak in the scintillator's wrapping (room light making random pulses); discriminator threshold too low (photomultiplier noise counted); timer's time scale off (clock frequency not the one the software assumes); photomultiplier afterpulses counted as early decays | read the singles rate with the detector open, then under a black cloth |
| Nuclear & particle | Radioactive decay (Geiger–Müller) | N = N₀ e^(−λt) + B, t½ = ln 2 / λ | room background not subtracted; long-lived activity in the sample (Cs-137 breakthrough from the generator); counter dead time losing counts at high rates | remove the source and count the room background |
| Nuclear & particle | Rutherford scattering (alphas on gold) *(graduate)* | N(θ) ∝ 1/sin⁴(θ/2) | detector angle zero misset (scale zero not on the beam axis); foil too thick (multiple scattering spreads the beam into the small angles); background counts from alphas leaking round the collimator; chamber not pumped down (gas in the alphas' path) | count at equal angles on both sides of the beam |
| Condensed matter | Band gap of germanium *(graduate)* | R ∝ T^(−3/2) e^(Eg/2kT) | sample lagging the oven's thermometer while it heats; oven thermometer reading high (two-wire lead resistance or calibration); readings taken below the intrinsic region (impurity carriers at the cold end); sample current too high (the bar heats itself); contacts degraded (a resistance in series with the bar) | keep logging as the oven cools back down |
| Condensed matter | Superconducting transition *(graduate)* | R(T) = R_n(T) / (1 + e^(−(T−Tc)/w)) | thermometer not at the sample's temperature (lag or gradient); measuring current too high (near the critical current); sample degraded (oxygen loss lowers Tc); thermal EMF at the voltage contacts (resistance never reaches zero) | record a warming sweep as well and compare it with the cooling sweep |
| Condensed matter | X-ray Bragg diffraction (NaCl) *(graduate)* | nλ = 2d sin θ | goniometer zero misset (every angle read from a wrong zero); crystal not at the 2:1 coupled position (turned in its holder); kβ peak taken for Kα at one order; wrong crystal in the holder (not NaCl) | scan the detector through the direct beam with the crystal removed |

The Helmholtz, laser-alignment, Faraday and Beer-Lambert experiments, and their CSV formats, come from the team's first prototype by [@mtalha05-phy](https://github.com/mtalha05-phy); his datasets run unchanged here, and his self-heating resistor dataset is part of the evaluation.

## What's new

1. **A closed troubleshooting loop.** PhysSandbox ranks the likely causes, picks the bench test with the most information per minute, then folds in only the *outcome* of the test you ran. On the demo data the leading cause goes from 64% to 92% and is confirmed.
2. **An agent whose code we assume is hostile.** A model on Vultr Serverless Inference writes Python to answer your questions. A static scan highlights risky lines, but nothing relies on it: the glovebox is the safeguard.
3. **Containment receipts that don't take the code's word for it.** Every job probes its own walls (kernel, network interfaces, DNS, outbound connections, disk, user, visible secrets) *before any untrusted code runs* and writes that first; agent code is mounted read-only but not yet executed. Agent code then runs in a separate child process; the runner itself is made non-dumpable, so the child can't reach into the runner's report through `/proc` (it can reach the tracer's copy of its own output, which the runner already treats as the code's word, and the tracer records the attempt). The child can shape its own output but not the probe, and only short text fields of its output are kept (no payloads, capped length); the host also reads at most 2 MB from any container. The host then checks the container's real configuration and exit state with `docker inspect` (runtime, network mode, read-only root, memory, process and CPU limits, capabilities, privilege flags, user, devices, mounts, no-exec scratch, OOM kill). Kills (memory, time) come only from Docker's record and the host's timer. What the code *tried* (reach the internet, write outside scratch space, read files it may not, start programs, signal other processes) comes from a **system-call tracer**: the runner starts `strace` on the child before the code exists, so wrapping an attempt in `try/except` hides nothing; the receipt says which source it used (`attempts_observed_by`). The wall that stopped each attempt is the part the host verifies. The **Forge its own receipt** button runs code that prints a fake all-green receipt and tries to write into the runner's report; it can't succeed by design, and the attempt is flagged.
4. **A lab notebook you can check without trusting us.** Receipts are hash-chained and each one is signed with Ed25519, as are the notebook's start (a genesis record with its creation time) and a head record (count and last hash) rewritten on each append. The public key is published at `/api/ledger/public-key` and pinned in this repository (`notebook/ed25519-public-key.pem`, key `cd5618512f391bda`), so anyone can check every signature with a short standalone script, `tools/verify_receipts.py`, instead of taking the server's word. Stripping signatures, editing an entry and rebuilding the hashes, or deleting entries off the end all fail. **Tamper with a copy** shows it: it edits a copy of the notebook the way an attacker without the private key would, and the check catches it. Rolling the notebook back to an older copy, or wiping it, can't be caught from the notebook alone; the signed head in every downloaded record catches it from outside ([threat model](#the-lab-notebook-check-it-yourself)).
5. **An agent that repairs its own code, sealed.** If the model's script fails in the box, the sandbox's error goes back to the model and it tries again (up to three attempts, each in a fresh container with its own receipt).

## Try it in 90 seconds

Open the demo and press **Take the 90-second tour**. It walks through a real investigation (resistor reads high → offset found → bench test → cause confirmed), a break-out attempt stopped by the network wall, a poisoned file whose hidden instruction reaches the model and is contained, and the tamper check on the lab notebook. Every step runs live in the sandbox.

Every investigation can be downloaded as a **PDF report** in one click: the outcome, the fix for the confirmed cause, and for every step the data chart with its fit, the measured values with uncertainties, the likely causes, the recommended test with how to do it and its safety note, plus each sandbox receipt with its signature. The browser sends only the steps' receipt hashes: the server builds the PDF from its own stored result of each step, whose SHA-256 is sealed inside that step's signed receipt, and only for the browser session that ran it. A made-up result, a real receipt from another job, or a stored result edited on disk is refused. The PDF also carries its own evidence as an attached file: each step's signed receipt and the exact result it was built from, so anyone can check a report offline, without the server, with `python tools/verify_report.py report.pdf notebook/ed25519-public-key.pem`. It checks every signature, every result against the hash sealed in its receipt, the order of the steps, and that the page text still shows the signed explanation, then prints the signed numbers to compare with the tables. A markdown record is available too; it ends with the lab notebook's signed head, to keep.

Adding an experiment group needs no edits to existing code: a physics module (`sandbox/tools/bench_<group>.py`), a fault library (`faults/bench_<group>.yaml`), a catalogue file (`app/catalog.d/<group>.yaml`), a data generator (`data/gen_<group>.py`) and its evaluation cases (`data/manifest.d/<group>.yaml`). Nine groups (33 of the 51 experiments) were added that way, built in parallel.

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

**Nothing the code writes reaches anyone else.** The public containment log (`/api/containment-log`) and the public receipts hold only text the host wrote itself and counts: when a job fails, the log says which exception it raised, never its message, and the tracer's view of what the code tried is reduced to counts. The paths, addresses and messages the code chose go back only to the person who ran the job. (An independent review found that an earlier version logged the first 200 characters of the error, which hostile code could fill with the data; that channel is closed.)

**Known limits.** The tracer runs inside the box as the same user as the code, so the code could kill it; that kill is itself recorded, and a tracer that stops early is reported as tampering. Fully host-side tracing (gVisor's `runsc --strace` on the VM) would remove even that. The app runs as root on the VM because it drives Docker, and Docker access is root-equivalent; the boundary we rely on is gVisor plus the container flags, and the signing key and inference key live on the same VM. Running the sandboxes on a separate Vultr instance in a private network is the next step.

## The lab notebook: check it yourself

Every job's receipt goes into the lab notebook (`receipts.jsonl`, one JSON record per line):

- **Genesis** (`seq` 0, `kind: "genesis"`): written when the notebook is created, with its start time and a random notebook id. Receipt #1 chains to it.
- **Receipts** (`seq` 1, 2, …): hashes of the job's input, code and output, the walls measured inside the box and checked by the host, the outcome, and `prev`, the hash of the record before it.
- **Head** (`kind: "head"`, its own file, rewritten on every append): `count`, `last_hash`, and the genesis record's hash, time and id.

Every record carries three more fields:

| Field | What it is |
| --- | --- |
| `hash` | SHA-256 of the record's canonical JSON: every field except `hash` and `sig`, keys sorted, no spaces (`json.dumps(sort_keys=True, separators=(",", ":"))`), 64 hex characters |
| `key_id` | the first 16 hex characters of SHA-256 of the raw 32-byte public key; it is inside the hashed part, so it is signed too |
| `sig` | Ed25519 signature over the ASCII text of `hash`, hex-encoded (128 characters) |

The live demo's notebook key is `cd5618512f391bda` (SHA-256 of the raw key `cd5618512f391bda1f6ce328cc125f500eac759bfaf7375f53d35b786929deb4`); its public key is committed at `notebook/ed25519-public-key.pem`, so a key served by the host can be compared with one fixed in git history. The private key is made on first use at `~/.physsandbox/ledger_ed25519.pem` (mode 0600, outside the repository and every sandbox; `LEDGER_SIGNING_KEY_FILE` moves it). The public key is at `GET /api/ledger/public-key` (JSON with the PEM and `key_id`; add `?format=pem` for the bare PEM).

To check the notebook without trusting the server (Python 3.8+ and `pip install cryptography`, nothing else):

```bash
curl -s 'https://HOST/api/ledger?limit=100' -o export.json
curl -s 'https://HOST/api/ledger/public-key?format=pem' -o notebook.pem
python tools/verify_receipts.py export.json notebook.pem
# Later: has anything changed since the head you kept?
python tools/verify_receipts.py export.json notebook.pem --expect-head physsandbox-record.md
```

The script recomputes every hash, checks every signature and link from the genesis record through the receipts to the head, and ignores the server's own verdict in the export. It exits 0 if everything checks out, 1 if it finds tampering, a rollback or a wipe, and 2 if the export doesn't cover the check. `--expect-head` takes any head you saved earlier: a downloaded investigation record (`.md`), a saved `/api/status` or `/api/ledger` response, a head record on its own, or `COUNT:HASH`. An export holds the latest 100 receipts; to compare with an older head, export from it with `/api/ledger?start=COUNT&limit=100`.

**Threat model: what is caught, and what needs a copy of the head kept elsewhere.**

| What someone does to the notebook | Caught? |
| --- | --- |
| Edits a receipt | Yes: it no longer matches its hash |
| Edits a receipt and rebuilds every hash after it | Yes: the signatures no longer match, and only the private key can make new ones |
| Strips the signatures | Yes: unsigned records are rejected |
| Deletes receipts off the end | Yes: the signed head records how many there were |
| Deletes or edits the head | Yes. The next job then starts a new notebook (the old files are archived beside it, not deleted), with a new start time |
| Restores an older copy of both files (rollback) | **Not from the notebook alone**: an older notebook is still a valid notebook, and "Check for tampering" says intact. Caught with an earlier head kept elsewhere: the count went down, or receipt #count no longer has that head's hash |
| Deletes both files (wipe) | The next job starts a new notebook with a new start time, shown as "Notebook started …" in the Lab notebook panel. With an earlier head, the script reports a different notebook |
| Replaces the signing key | Every old signature fails, and the new notebook's `key_id` differs from the one in your saved record and public key |
| Gets root on the VM and reads the private key | Can sign a new history. An earlier head kept elsewhere still catches any change to the receipts up to it, because its hash covers all of them; receipts after it rest on trusting the server |

So the notebook proves on its own that nothing in it was edited, removed or added out of order; to prove it was not rolled back or replaced, you need a head from earlier. Every downloaded investigation record ends with the signed head at the time of download (the record's own receipts are in the chain behind it), and `/api/status` and `/api/ledger` include it as `head_record`. Keep one where the server can't reach it, for example in your email.

**Notebook history.** The live notebook started on 5 November 2026 at 20:42 UTC, when the demo moved from HMAC receipts (which only the server can check) to Ed25519. The earlier HMAC notebook was moved aside on the VM, not deleted, which is why the containment log shows jobs from before that time. `deploy/update.sh fresh` starts a new notebook the same way; we record every such restart here, and snapshots of the signed head are committed under `notebook/`, so a later rollback past a committed head can be caught by anyone with `--expect-head`.

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

`python eval/run_eval.py` replays the whole troubleshooting loop on 288 cases with known faults (`data/manifest.yaml` and `data/manifest.d/`): for every experiment, a healthy run, its common faults, and the follow-up result of the recommended test. It reports the first measurement and the follow-ups separately:

- **First measurement, fault cases: 108/108** name the true cause first, across all 51 experiments.
- **After the recommended test: 51/51** confirm the true cause (above 90%).
- **No false alarms: 51/51** healthy datasets come back "no fault found", with no test suggested.
- **When the odds point the wrong way: 39/39.** In the ambiguous demos the data alone favour the more common of two faults. Partner cases seed the *less* common one instead (one for each of the first 39 experiments): the tool must rank it second, recommend the test that separates the two, and put it first once the test result is in. All 39 do; 29 are already confirmed above 90%, and the other 10 lead and get a second test.
- **Not tuned to one lucky dataset**: `eval/run_seeds.sh` regenerates every dataset with ten other random seeds and replays the whole evaluation for each. On the 39-experiment bench all ten passed (1,570 of 1,570); each of the three newest groups passed seeds +11, +33 and +66.
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
sandbox/    job image: runner.py (probe, system-call tracer, run), physics models (analyze.py, experiments.py, bench_*.py), break-out demos
faults/     fault libraries, next-test catalogue and a fix for every fault (YAML; bench_<group>.yaml per experiment group)
data/       seeded dataset generators (generate*.py, gen_*.py) and ground truth (manifest.yaml, manifest.d/)
eval/       evaluation of the full troubleshooting loop
static/     single-page interface
tools/      verify_receipts.py: checks a lab notebook export with its public key, without trusting the server;
            verify_report.py: checks a downloaded PDF report offline from the evidence attached to it
notebook/   the live demo's pinned Ed25519 public key and committed snapshots of the signed head
deploy/     Vultr VM setup script and in-place update script
```

## Safety

Bench recommendations come from a fixed test catalogue with safety notes (never lift a safety earth; laser changes need your laser safety officer). Model text is checked against forbidden advice before it is shown.

## License

MIT
