# PA-2026-193: Early Detection of Lithium-Ion Battery Cell Venting Using In-Pack Acoustic Sensing Fused with Battery Management Telemetry

**Title:** System and Method for Early Detection of Lithium-Ion Battery Cell Venting Using In-Pack Acoustic Sensing Fused with Battery Management Telemetry

**Filing:** LITF-PA-2026-193
**Published:** October 4, 2026
**Domain:** EV / Battery Safety / Acoustics
**Full Disclosure:** [liveinthefuture.org/priorart/battery-venting-acoustic-detection.html](https://liveinthefuture.org/priorart/battery-venting-acoustic-detection.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published openly as a technical disclosure
> under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> Whether it constitutes prior art, and what it discloses, depends on the facts,
> including public accessibility, timing, and the disclosure's content.
> Publication does not establish novelty, patentability, freedom to operate, or
> public-domain status. This disclosure is offered as evidence of the state of
> the art for examiners and challengers to consider. This is general
> information, not legal advice.

---

## Abstract

Disclosed is a system and method for early warning of lithium-ion battery thermal runaway based on acoustic detection of cell venting. When a cell enters thermal runaway, its pressure-relief vent opens and releases hot gases, producing a distinctive broadband acoustic burst (nominally 2 to 8 kHz, lasting from tens of milliseconds to several seconds) that propagates through the pack enclosure at the speed of sound. A distributed array of MEMS microphones disposed within or on the battery pack continuously monitors the acoustic channel; an on-device classifier distinguishes venting signatures from confounders such as contactor actuation, road noise, coolant pumps, and charger relays. Candidate vent events are fused with battery management system (BMS) telemetry (cell voltage dip, temperature rise rate) for confirmation, localized to the venting module by time-difference-of-arrival (TDOA) analysis across the array, and answered with a graduated response that escalates from driver alert to contactor opening to emergency-services notification. Because sound reaches the sensors in milliseconds while gas sensors wait on diffusion and temperature sensors wait on heat conduction, the acoustic channel is the fastest available precursor signal for thermal runaway in a production pack.

## Technical Field

This disclosure relates to battery safety systems for electric vehicles and stationary energy storage, specifically to detecting the acoustic signature of lithium-ion cell venting as an early precursor of thermal runaway, using in-pack microphone arrays, on-device audio classification, and fusion with battery management telemetry.

## Background

Thermal runaway in lithium-ion battery packs remains the dominant catastrophic failure mode for electric vehicles. Once a single cell enters runaway, exothermic decomposition can propagate cell to cell in seconds to minutes, producing fires that exceed 800°C, release toxic gases including hydrogen fluoride, and resist conventional suppression. Early warning is the difference between a parked car with a warning light and a garage fire.

Production detection today relies almost entirely on the battery management system: per-cell or per-group voltage sensing and a sparse set of temperature sensors (typically one per module or fewer). Both channels lag the event they are meant to catch. Voltage dips only after the cell is already in severe distress, and temperature sensors measure the pack structure, not the cell: heat must conduct from the failing cell through potting, busbars, and module housings before a thermistor registers anything, a delay of tens of seconds to minutes. By the time the BMS flags a thermal event, propagation is often underway.

Gas sensing has been proposed as a faster channel: venting releases hydrogen, carbon monoxide, and volatile organics before flames appear. But gas must diffuse from the venting cell to the sensor location, a process that takes seconds to minutes in a sealed pack with tortuous internal airflow, and gas sensors drift, poison, and cross-react. LITF-PA-2026-119 covering VOC-based thermal runaway detection discloses one such approach; it inherits the diffusion delay.

Laboratory acoustic-emission studies have attached piezoelectric sensors to individual cells and recorded the sounds of venting during abuse tests. This work proves that venting is acoustically distinctive, but it has never been reduced to a deployed pack-level system: lab setups use expensive contact piezoelectric sensors on single cells, record to benchtop equipment, and perform offline analysis. No production or disclosed system places acoustic sensors inside a vehicle pack, classifies venting against the full confounder set of a running vehicle, fuses acoustic detections with BMS telemetry, localizes the venting module, or drives a graduated vehicle-level response.

The gap in the art is a complete deployable system that: (a) senses the acoustic channel inside a production battery pack with low-cost microphones, (b) classifies venting signatures on-device against vehicle confounders in real time, (c) confirms detections by fusion with BMS voltage and temperature telemetry to reach automotive false-positive budgets, (d) localizes the venting module for targeted isolation, and (e) executes a graduated safety response.

## Detailed Description

### 1. Venting acoustic physics

A lithium-ion cell in thermal runaway generates gas internally (electrolyte decomposition, SEI breakdown) until internal pressure ruptures the cell's pressure-relief vent or bursts the casing. The acoustic event has three phases. First, the vent rupture itself: a sharp impulsive transient, typically under 50 ms, with broadband energy extending past 10 kHz. Second, the sustained gas jet: as pressurized hot gas escapes through the vent orifice, it produces turbulent broadband noise concentrated nominally between 2 and 8 kHz, lasting from a fraction of a second to several seconds depending on cell format and state of charge. Third, in some chemistries and formats, a pre-rupture crackling as the jellyroll or pouch delaminates, seconds before the main event.

Inside a pack enclosure, this signal propagates both through the internal airspace (at approximately 343 m/s, faster in the heated gas near the venting cell) and as structure-borne vibration through module housings and the pack tray. The enclosure acts as a reverberant cavity, which smears the impulse but preserves the sustained jet spectrum. The key property exploited here is latency: the acoustic signal reaches every sensor in the pack within a few milliseconds of vent opening, orders of magnitude faster than heat conduction to a thermistor or gas diffusion to a chemical sensor.

### 2. Sensor array hardware

In one embodiment, each battery module carries at least one MEMS microphone (e.g., a bottom-port digital MEMS microphone with flat response to 10 kHz, unit cost under $1.00 in automotive volumes), mounted to the module lid interior or the busbar carrier, facing the cell vent paths. For a typical 4-to-8-module passenger-vehicle pack, this yields 4 to 16 sensing points. Microphones connect via the existing BMS wiring harness (digital PDM or I2S over shielded pairs) to keep incremental harness cost near zero. In another embodiment, microphones mount to the exterior of the pack lid above each module, trading some signal strength for serviceability and avoiding intrusion into the sealed pack volume.

Microphone selection criteria: operating temperature range covering at least -40°C to +105°C (pack interior during fast charging), survival of condensing humidity and vibration per automotive qualification, and a noise floor at least 20 dB below the expected vent jet level at the sensor position. The microphones are sacrificial with respect to the thermal event itself: detection occurs in the first seconds of venting, before flame and heat destroy the sensor, which is acceptable because the sensor's job is complete once the event is classified and reported.

### 3. Acquisition and transient pre-detection

Each microphone channel is sampled continuously (in one embodiment, 48 kHz at 16-bit resolution) and bandpass filtered to the vent band (nominally 1 to 12 kHz), rejecting low-frequency road and drivetrain noise and high-frequency switching noise. A short-time-average over long-time-average (STA/LTA) transient detector, adapted from seismology, runs on the filtered energy envelope: when the short-term energy exceeds the long-term background by a configurable ratio (nominally 6 dB sustained over 20 ms), the system triggers.

On trigger, the system replays a ring buffer holding the preceding 2 seconds of raw audio from all channels and captures the following 5 seconds, producing a 7-second multi-channel event clip centered on the transient. Continuous recording is unnecessary: only triggered clips are classified, keeping compute and storage bounded. The STA/LTA stage is deliberately sensitive (high recall); the classifier in Section 4 provides the precision.

### 4. Vent signature classifier

Each triggered clip is converted to log-mel spectrograms (in one embodiment, 64 mel bins, 25 ms frames, 10 ms hop) and passed to an on-device convolutional neural network classifier. The network is trained on a vent signature library (Section 9) containing labeled venting recordings from controlled abuse tests across cell formats (cylindrical, pouch, prismatic), chemistries (NMC, NCA, LFP), and states of charge, plus a confounder library recorded from real vehicles: high-voltage contactor actuation, main relay clicks, coolant pump cavitation and bearing noise, on-board charger relay chatter, DC fast-charge contactor sequencing, door slams transmitted through the body, pothole impacts, gravel strikes on the pack shield, and hail.

The classifier outputs a vent probability per clip. In one embodiment, the network is quantized to INT8 and executes on the BMS microcontroller or a dedicated audio DSP in under 100 ms per clip, so classification completes while the gas jet is still in progress. A per-vehicle adaptive background model tracks the pack's normal acoustic fingerprint (pump whine harmonics, road-noise spectrum at speed) and subtracts it before classification, suppressing slow confounder drift such as pump aging.

### 5. BMS telemetry fusion and confirmation

Acoustic classification alone cannot meet automotive false-positive budgets: a pothole strike at the right frequency could mimic a vent rupture transient. Confirmation comes from fusion with BMS telemetry. On a candidate vent detection, the fusion module queries the BMS for the module nearest the acoustic localization (Section 6): cell-group voltage dip exceeding a threshold (nominally 50 mV within 10 seconds of the acoustic trigger), temperature rise rate exceeding a threshold (nominally 2°C per minute on the nearest thermistor), and isolation resistance drop. Each signal contributes a likelihood term; a Bayesian update combines them with the acoustic classifier score into a posterior vent probability.

The fusion logic is asymmetric by design: a strong acoustic signature plus any single corroborating BMS signal confirms the event, while a weak acoustic score requires two corroborating signals. This reflects the physics that the acoustic channel leads and the electrical and thermal channels lag. The target system false-positive rate is below one confirmed false alarm per vehicle per year, with detection probability above 99% for venting events that precede propagating runaway.

A deliberate safety-architecture property: the acoustic channel is physically diverse from the voltage and temperature channels. It does not share sensors, wiring, or failure modes with the BMS sensing it corroborates, which supports an ASIL decomposition argument under ISO 26262: two independent channels, each insufficient alone, jointly reaching the required integrity level for a thermal-runaway warning function.

### 6. Localization via TDOA

When three or more microphones trigger on the same event, time-difference-of-arrival analysis estimates the source position. With the speed of sound in the pack airspace and typical inter-microphone spacing of 0.3 to 1.0 meters, TDOA resolution is sufficient for module-level localization (which of 4 to 8 modules contains the venting cell), and in favorable geometries, cell-group-level localization within a module. The localization output serves two purposes: it selects which module's BMS telemetry the fusion module weights most heavily, and it tells the response logic (Section 7) which module to electrically isolate and which region of the pack firefighters should cool first.

### 7. Graduated response

A confirmed venting event triggers a staged response, escalating with continued evidence:

- **Level 1, advisory:** driver warning (instrument cluster message and chime: battery thermal event detected, pull over when safe), event logged with acoustic clip, GPS position, and module localization uploaded via telematics to the manufacturer safety backend.
- **Level 2, mitigation:** BMS limits charge and discharge power, commands maximum battery cooling, and pre-charges the thermal management system for sustained operation. If the vehicle is DC fast charging, the session is terminated.
- **Level 3, isolation:** on confirmation of a second venting event or continued temperature rise indicating propagation, the BMS opens the high-voltage contactors, isolating the pack. Doors unlock, hazard lights activate, and the 12 V system stays alive to power the acoustic monitor and telematics.
- **Level 4, emergency notification:** telematics transmits a thermal-event message with GPS coordinates, venting module location, and pack chemistry to emergency services and the manufacturer's incident response team, giving first responders the module map before they arrive.

Escalation is automatic but each level is independently inhibitable by a higher-authority vehicle controller (e.g., the vehicle may suppress Level 3 contactor opening above a speed threshold where sudden power loss is itself hazardous, deferring isolation until speed falls below the threshold).

### 8. Retrofit and adjacent embodiments

**Cabin-microphone retrofit:** many vehicles already contain hands-free and voice-assistant microphones in the cabin. In one embodiment, the vent classifier runs on the infotainment processor using the existing cabin microphone array. Pack venting couples into the cabin as structure-borne and airborne sound through the floor pan; sensitivity is lower than in-pack sensing, but the embodiment requires no pack hardware changes and can be deployed by software update to vehicles already on the road.

**Micromobility charger-side monitoring:** e-bike and e-scooter battery fires during charging are a major urban fire cause. In one embodiment, a smartphone placed near the charging battery runs the vent classifier on its microphone: the sustained gas-jet signature of a venting pouch cell is detectable at room distances in a quiet indoor environment, and detection triggers an audible alarm and a push notification advising the user to unplug the charger and move the battery outdoors. No hardware is added to the battery or charger.

**Stationary storage:** containerized battery energy storage systems (BESS) hold thousands of cells with long gas-diffusion paths that make chemical sensing slow. In one embodiment, microphone arrays mount on container walls and rack uprights, with per-rack localization directing the container's gas-suppression system to the affected rack and informing the site's fire panel which container and rack to prioritize.

**Post-crash first-responder mode:** thermal runaway can begin minutes to hours after a crash damages cells without immediate venting. In one embodiment, airbag deployment (or a telematics crash flag) arms a low-power acoustic watch: the in-pack array keeps listening on the 12 V bus after the high-voltage system is depowered, and any vent signature triggers the Level 4 emergency broadcast with the crash GPS coordinates, warning responders approaching the vehicle of a developing thermal event.

### 9. Training data: the abuse-test signature library

The classifier's vent signature library is built from controlled abuse testing: nail penetration, overcharge, external heating, and crush tests performed on cells and modules inside instrumented enclosures, with synchronized recording of in-enclosure audio, per-cell voltage, thermocouple temperatures, and high-speed video for ground-truth vent timing. Each test yields labeled vent clips (rupture transient plus jet, with onset times from video) and labeled confounder clips (the abuse apparatus itself: nail gun actuation, heater relay clicks). The library spans the format, chemistry, and state-of-charge matrix so the classifier generalizes across pack designs rather than overfitting to one cell type. Manufacturers deploying the system re-run a reduced abuse matrix on their specific cell and pack geometry to fine-tune the deployed model, a process analogous to crash-test calibration of airbag algorithms.

### 10. Figures Description

- **Figure 1:** Pack cross-section showing MEMS microphone positions relative to modules, cell vent paths, and the BMS controller, with the acoustic propagation paths from a venting cell drawn as wavefronts.
- **Figure 2:** Example log-mel spectrograms contrasting a venting event (rupture transient followed by sustained 2 to 8 kHz jet) against four confounders: contactor actuation, pothole impact, coolant pump cavitation, and door slam.
- **Figure 3:** Signal-processing pipeline: bandpass filter, STA/LTA trigger, ring-buffer replay, spectrogram computation, CNN classification, BMS fusion, TDOA localization, and the graduated response state machine.
- **Figure 4:** Timing diagram comparing detection latency across channels for a representative venting event: acoustic classification at under 1 second, voltage dip at tens of seconds, thermistor response at minutes, gas sensor at minutes.

## Claims

1. A system for early detection of lithium-ion battery thermal runaway, comprising: an acoustic sensor array disposed within or on a battery pack enclosure; an acoustic classifier configured to distinguish cell-venting acoustic signatures from non-venting acoustic events; a battery management system interface providing cell voltage and temperature telemetry; and a fusion module configured to confirm a cell-venting event based on the acoustic classifier output combined with the battery management telemetry.

2. The system of claim 1, wherein the acoustic sensor array comprises MEMS microphones with at least one microphone per battery module, connected via the battery management wiring harness.

3. The system of claim 1, wherein the acoustic classifier comprises a convolutional neural network operating on log-mel spectrograms, trained on venting signatures from controlled abuse tests across cell formats, chemistries, and states of charge, and on a confounder library comprising contactor actuation, road impacts, coolant pump noise, and charger relay events.

4. The system of claim 1, further comprising a short-time-average over long-time-average (STA/LTA) transient pre-detector operating on bandpass-filtered microphone energy, and a ring buffer configured to replay pre-trigger and post-trigger audio to the classifier on detection.

5. The system of claim 1, wherein the fusion module applies asymmetric confirmation logic: a strong acoustic classification score combined with any single corroborating battery management signal (cell voltage dip, temperature rise rate, or isolation resistance drop) confirms the event, while a weak acoustic score requires at least two corroborating signals.

6. The system of claim 1, further comprising a time-difference-of-arrival localization module configured to identify the venting battery module from arrival-time differences across the acoustic sensor array, the localization output selecting which module's telemetry the fusion module weights and directing module-level electrical isolation.

7. The system of claim 1, further comprising a graduated response controller configured to escalate through driver advisory, charge and discharge power limiting with maximum battery cooling, high-voltage contactor opening, and emergency-services notification with venting module location, the escalation driven by confirmation strength and evidence of thermal propagation.

8. The system of claim 1, wherein the acoustic sensor array comprises existing vehicle cabin microphones and the acoustic classifier executes on the infotainment processor, the system being deployable by software update without battery pack hardware modification.

9. A method for monitoring a rechargeable micromobility battery during charging, comprising: capturing ambient audio with a smartphone microphone placed near the charging battery; classifying the audio with a vent-signature classifier trained on pouch-cell venting recordings; and on detection of a venting signature, issuing an audible alarm and a notification advising disconnection of the charger.

10. The system of claim 1, adapted for containerized battery energy storage, wherein microphone arrays mounted on container walls and rack uprights provide per-rack venting localization directing container gas-suppression discharge and fire-panel prioritization.

11. A method of building a battery venting acoustic signature library, comprising: subjecting cells and modules to controlled abuse tests comprising nail penetration, overcharge, external heating, and crush; recording synchronized in-enclosure audio, per-cell voltage, temperature, and high-speed video; labeling vent onset times from the video; and training a classifier on the labeled vent clips and on confounder clips from the abuse apparatus.

12. The system of claim 1, further comprising a post-crash acoustic watch mode armed by airbag deployment or a telematics crash flag, wherein the acoustic sensor array continues monitoring on low-voltage power after high-voltage depowering and triggers an emergency broadcast with crash coordinates on detection of delayed cell venting.

## Implementation Notes

Microphone placement matters more than microphone quality: a $0.60 MEMS microphone positioned with a direct acoustic path to the module's cell vent manifold outperforms a laboratory measurement microphone mounted outside the pack. During pack design, the vent-gas routing (manifolds, burst discs, pack vents) should be co-designed with microphone placement so that every cell's vent path passes within the near field of at least one microphone. The abuse-test matrix in Section 9 is the difference between a research demo and a deployable product: a classifier trained only on one cell format will miss the lower-frequency jet of large prismatic cells and false-trigger on the sharper rupture of small cylindricals. Budget the test matrix accordingly; it is cheaper than one recall. The STA/LTA thresholds must adapt to vehicle state: parked and charging (quiet background, low thresholds), driving on rough roads (elevated background, higher thresholds, heavier reliance on fusion). Log every triggered clip with its classifier score and fusion outcome to the telematics backend; the fleet-wide false-trigger corpus is the training data for the next model revision. Treat the acoustic channel as a diverse redundant safety path in the ISO 26262 work products: it shares no sensors, wiring, or failure modes with the voltage and temperature channels, which is what makes the fusion argument credible to assessors.

## Limitations

This method detects venting, not the internal short or SEI breakdown that precedes it: there is an electrochemical failure interval before any acoustic signal exists, and pre-vent detection requires other techniques such as impedance spectroscopy (see LITF-PA-2026-052). Very slow gas weeps through a degraded seal may fall below the classifier's energy threshold; the system targets the rapid venting that precedes propagating runaway, not micro-leaks. Sealed pack designs with heavy acoustic damping attenuate the high-frequency rupture transient, leaving the lower-frequency jet as the primary signature; the classifier must be trained on the as-built pack acoustic transfer function, not free-field recordings. The microphones are sacrificial in a full thermal event: detection must complete in the first seconds of venting, which the latency budget in Figure 4 supports but which leaves no margin for deferred processing. Confounder overlap is irreducible in principle: a sufficiently violent road impact can mimic a rupture transient, which is why the fusion stage in Section 5, not the classifier alone, carries the false-positive budget. The cabin-microphone retrofit embodiment has lower sensitivity and is unsuitable as a primary safety channel; it is a software-deployable supplement for the existing fleet.

## Prior Art References

1. Laboratory acoustic-emission studies of lithium-ion cells using contact piezoelectric sensors during abuse testing: establish that cell venting produces distinctive acoustic signatures, but are confined to single cells, benchtop recording, and offline analysis with no pack-level deployment, classification, telemetry fusion, or vehicle response
2. LITF-PA-2026-119, "System and Method for Battery Thermal Runaway Detection Using VOC Sensing": gas-phase detection of venting; the present disclosure uses the acoustic channel, which propagates in milliseconds rather than diffusing over seconds to minutes
3. LITF-PA-2026-052, "Battery Internal Temperature Estimation Using Electrochemical Impedance Spectroscopy": pre-vent electrochemical detection; complementary to the present disclosure, which targets the venting event itself
4. Production battery management systems (per-cell voltage sensing, sparse thermistor arrays): the incumbent detection approach whose thermal and electrical lag motivates the present disclosure
5. Allen, R. V., "Automatic earthquake recognition and timing from single traces," *Bulletin of the Seismological Society of America*, vol. 68, no. 5, 1978: the STA/LTA transient detection algorithm adapted in Section 3
6. ISO 26262 (road vehicles, functional safety) and UL 2580 (batteries for use in electric vehicles): the safety standards framing the ASIL decomposition and pack qualification context
7. Log-mel spectrogram convolutional networks for acoustic event classification: the standard on-device audio classification technique applied here to vent signatures
