# PA-2026-198: Verifying Life-Safety Alarm Audibility Coverage in Residential Structures Using Distributed Acoustic Sensing

**Title:** System and Method for Verifying Life-Safety Alarm Audibility Coverage in Residential Structures Using Distributed Acoustic Sensing

**Filing:** LITF-PA-2026-198
**Published:** October 9, 2026
**Domain:** Life Safety / Acoustics
**Full Disclosure:** [liveinthefuture.org/priorart/smoke-alarm-audibility-coverage-audit.html](https://liveinthefuture.org/priorart/smoke-alarm-audibility-coverage-audit.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> effective as of the publication date above. It is intended to be cited as
> prior art against later-filed patent applications covering the subject matter
> described. The inventors, through this publication, reserve their rights
> under 35 U.S.C. Sec. 102(b)(1). The disclosure is intended to enable a
> person of ordinary skill in the art to make and use the described systems.
> This is general information, not legal advice.

---

## Abstract

Disclosed is a system and method for verifying that smoke alarms and carbon monoxide (CO) alarms installed in a residence remain audible at code-required levels in every sleeping area, using distributed microphones already present in the home. Each microphone node (smart speaker, opted-in smartphone, or dedicated MEMS node) registers its room assignment with a local hub. During a test event, the homeowner triggers each alarm's test function in a guided sequence, or the system coordinates with alarms that support scheduled self-tests. The hub identifies the standardized temporal alarm patterns (T3 for smoke, T4 for CO per ISO 8201) via template correlation in the piezoelectric sounder band near 3.1 kHz, estimates the received A-weighted sound pressure level at each node, and maps levels to rooms. Per-room audibility verdicts are computed against NFPA 72 sleeping-area criteria (75 dBA at the pillow, 15 dB above ambient), with the scored test run performed with bedroom doors closed to represent nighttime conditions. Rooms falling below threshold are flagged as dead zones with remediation guidance and a retest workflow. A longitudinal module tracks each alarm's sounder output across successive tests to project an audibility-based end of life, and an interconnect check confirms that triggering one alarm sounds all interconnected units. All pattern detection runs on-device; nodes transmit only timestamped feature vectors, never raw audio.

## Technical Field

This disclosure relates to residential life-safety systems, specifically to verifying the notification performance of installed smoke and carbon monoxide alarms through distributed acoustic sensing, per-room audibility mapping, and longitudinal sounder degradation tracking.

## Background

Building codes require smoke alarms in specific locations, but almost nothing in residential practice verifies that an installed alarm can actually be heard where people sleep. UL 217 requires residential smoke alarms to produce at least 85 dB at 10 feet under laboratory conditions. NFPA 72 requires audible notification in sleeping areas to reach 75 dBA at the pillow and 15 dB above average ambient sound. Between the laboratory and the pillow sit real houses: closed bedroom doors, long hallways, high ceilings, soft furnishings, and HVAC background noise. Field guidance on alarm sound measurement notes that closed interior doors substantially reduce the sound level reaching occupants, and that ceiling height, room materials, and background noise all erode the margin between the rated output and what a sleeper experiences ([alarm decibel and sound requirements survey](https://www.linkedin.com/pulse/smoke-alarm-decibel-sound-requirements-what-ul-217-en-tim-sh-td2sc)).

The gap this disclosure addresses is verification. Existing consumer systems listen for alarms for a different purpose:

- Alarm-event detectors: Apple HomePod sound recognition, Amazon Alexa Guard, and Google Nest Aware detect that an alarm is currently sounding and push a remote notification ([TechHive](https://www.techhive.com/article/1790156/apple-homepods-can-now-detect-the-sound-of-smoke-alarms.html)). They answer "is an alarm going off right now," not "would anyone hear it."

- Alarm listeners: The Ring Alarm Smoke and CO Listener, Ecolink Firefighter, and SonicAlert HomeAware II listen for an existing alarm's sounder and relay the event to a security panel or notification system ([SonicAlert](https://www.sonicalert.com/The-HomeAware-Fire-CO-Alert-Internal-Smoke-CO-Listener-Package_2?gad_source=5&gclid=EAIaIQobChMI9a3o19mZigMVIGFHAR3gWgeYEAAYASAAEgJY3fD_BwE)). They extend notification, they do not audit coverage.

- Detector self-maintenance: [EP 4358049 A1](https://patentimages.storage.googleapis.com/94/aa/98/9738e640bc0b50/EP4358049A1.pdf) (Bosch) describes a smoke detector with a microphone performing predictive maintenance on the detector itself. It monitors the device, not the acoustic path to the occupants.

- Emergency sound monitoring: [WO2015035187A1](https://patents.google.com/patent/WO2015035187A1/en) describes monitoring sound during an in-building emergency to guide evacuation. It operates during emergencies under central command, not as a consumer coverage audit during normal occupancy.

No existing consumer system measures, per room, whether installed alarms meet audibility criteria; maps the dead zones; tracks sounder aging against audibility rather than calendar age; or verifies interconnect propagation acoustically. That combination is the subject of this disclosure.

## Detailed Description

### 1. System Architecture

The system comprises a set of microphone nodes, a local hub, and a node registry. Nodes are microphones already present in the home: smart speakers with always-on microphones, smartphones opted in for the duration of a test, or dedicated low-cost MEMS microphone nodes placed in rooms lacking coverage. Each node registers with the hub once: a room assignment, an approximate position within the room, and a device class (used for calibration offsets). The registry marks which rooms are sleeping areas, since those carry the strictest audibility criteria. The hub is a local edge device or a phone application; it coordinates test runs, collects per-node feature vectors, and renders the coverage map. No raw audio leaves any node.

### 2. Test Stimulus and Capture Protocol

Two stimulus classes are supported. In guided sequential testing, the hub's companion app walks the homeowner through pressing the test button on each alarm one at a time, in a displayed order. Sequential testing isolates each alarm as the sole sound source, which makes per-alarm attribution exact. In scheduled self-test coordination, the hub aligns its capture window with alarms that perform automated self-tests, where the alarm model exposes the schedule. Every capture begins with a 60-second ambient pre-roll during which each node measures its ambient floor (HVAC, appliances, traffic) so the ambient-plus-15 dB criterion can be evaluated honestly rather than assumed.

Each protocol run is performed twice: once with interior doors open and once with bedroom doors closed. The doors-closed run is the scored run, because it represents nighttime sleeping conditions when audibility matters most. The hub records the door state with each run and never scores an open-door run as a coverage pass for a sleeping area.

### 3. Alarm Pattern Identification

Nodes run a template correlator against the standardized temporal patterns: T3 for smoke (three 0.5-second pulses separated by 0.5-second pauses, repeating after a 1.5-second pause, per ISO 8201) and T4 for CO (four pulses on the same timing). Correlation is computed in the piezoelectric sounder band centered near 3.1 kHz, where residential alarm sounders concentrate their energy, with harmonic tracking to reject narrowband impostors such as microwave beeps. The classifier additionally distinguishes the full alarm cadence from the test-button cadence (often abbreviated), the low-battery chirp (a single short chirp roughly once per minute), and the end-of-life chirp pattern, so a maintenance chirp is never scored as a coverage pass. Detection runs on-device inside a rolling buffer of at most 60 seconds; the buffer is discarded after feature extraction.

### 4. Received-Level Estimation Without Absolute Calibration

Consumer microphones are uncalibrated, and absolute SPL from a phone or smart speaker is not trustworthy. The system therefore works primarily in relative terms. For each test event, the node closest to the sounding alarm (normally in the same room) is designated the reference node, and its received level anchors the event. Every other node reports attenuation relative to the reference, which cancels per-device gain error for the comparisons that matter: whether the hallway alarm reaches the far bedroom at anything like the level it reaches the near one.

Where absolute numbers are required, namely the 75 dBA sleeping-area threshold, the hub applies a per-device-class calibration offset learned from fleet measurements of identical hardware (microphone gain is consistent within a device model even though it is unknown per unit) and reports the verdict with a confidence interval instead of a point value. A MARGINAL band around the threshold absorbs the residual calibration uncertainty; only levels clearly above the band score PASS, and only levels clearly below it score FAIL. This keeps uncalibrated hardware from manufacturing false confidence.

### 5. Coverage Mapping and Dead-Zone Detection

The hub maps per-node received levels to rooms via the registry and issues a per-room verdict for each sounding alarm: PASS, MARGINAL, or FAIL against max(75 dBA, ambient + 15 dB) for sleeping areas, and against ambient + 15 dB for non-sleeping areas. Results render as a per-floor coverage map in the companion app. Any sleeping area scoring FAIL in the doors-closed run is a dead zone. The remediation engine then proposes concrete fixes in priority order: add an interconnected alarm inside the failing room (the fix codes increasingly require), relocate an existing alarm closer to the sleeping area, or install a listed auxiliary notification appliance where alarm relocation is impractical. Each recommendation carries a retest action; the dead zone is not cleared until a subsequent doors-closed run scores PASS.

### 6. Longitudinal Sounder Degradation Tracking

Piezoelectric sounders lose output with age: crystal aging, dust loading of the sound port, and battery sag in non-hardwired units all reduce radiated level. The hub stores each alarm's reference-anchored peak level from every test and fits a decay trend per alarm. From the trend it projects the date the alarm will fall below the audibility threshold in the farthest room it is expected to serve. That projection is an audibility-based end of life, and it can arrive well before the nominal ten-year replacement date printed on the unit, or well after it for a sounder aging gracefully. The system distinguishes source decay from path changes (a moved bookshelf, a replaced hollow-core door with a solid one) by a simple test: if every node's relative level shifted together, the acoustic path changed; if only the alarm's anchored level moved, the sounder decayed. Path changes trigger a re-baseline prompt, not a replacement recommendation.

### 7. Interconnect Propagation Verification

In interconnected systems (wired or wireless), triggering one alarm should sound every unit. During a guided sequential test the hub expects each alarm's acoustic fingerprint to appear in the capture when its turn comes, and during any single-alarm trigger it listens for the full set of expected fingerprints across the home. A unit that never appears indicates an interconnect failure, a dead sounder, or a removed alarm, and is flagged distinctly from a coverage FAIL: the problem is not that the alarm is too quiet in the bedroom, it is that the bedroom alarm never sounded at all. Where simultaneous sounding makes per-alarm attribution ambiguous, the hub falls back to per-unit piezo resonant-frequency fingerprints: each sounder's exact resonant peak differs by tens of hertz from unit to unit and is stable over time, so overlapping T3 cadences can still be separated by their spectral fingerprints.

### 8. Privacy and Data Minimization

All pattern detection and level estimation run on the node. The only data transmitted to the hub are timestamped feature vectors: pattern identifier, band-limited received level, ambient floor, and node identifier. Raw audio never leaves the node, is held only in the short rolling buffer needed for correlation, and is discarded after feature extraction. Smartphone participation is opt-in per test run, with an explicit on-screen indicator while the microphone is active.

## Claims

1. A system for verifying life-safety alarm audibility coverage in a residence, comprising: a plurality of microphone nodes distributed across rooms of the residence, each node registered with a room assignment in a node registry marking sleeping areas; a test coordination module that captures a per-node ambient sound floor during a pre-roll interval and then captures each installed smoke or carbon monoxide alarm's test signal in a guided sequential order isolating one alarm at a time; a pattern identification module that detects standardized temporal alarm patterns, including the T3 smoke pattern and the T4 carbon monoxide pattern, via template correlation in the piezoelectric sounder frequency band; a received-level estimation module that designates a reference node nearest the sounding alarm and computes per-node attenuation relative to the reference without requiring absolute microphone calibration; and a coverage mapping module that issues a per-room audibility verdict against a threshold of the greater of 75 dBA and the measured ambient floor plus 15 dB for sleeping areas, wherein the scored test run is performed with bedroom doors closed.

2. The system of claim 1, wherein the pattern identification module further distinguishes a full alarm cadence from a test-button cadence, a low-battery chirp, and an end-of-life chirp pattern, and wherein a maintenance chirp is never scored as a coverage pass.

3. The system of claim 1, wherein the received-level estimation module applies a per-device-class calibration offset learned from fleet measurements of identical hardware for absolute-threshold comparisons, and reports verdicts with a marginal band around the threshold absorbing residual calibration uncertainty, such that only levels clearly above the band score a pass and only levels clearly below it score a fail.

4. The system of claim 1, further comprising a door-state conditioning module that executes each protocol run twice, with interior doors open and with bedroom doors closed, records the door state with each run, and scores sleeping-area coverage only from the doors-closed run.

5. The system of claim 1, further comprising a remediation engine that, for any sleeping area scoring below threshold in the doors-closed run, proposes in priority order adding an interconnected alarm inside the failing room, relocating an existing alarm closer to the sleeping area, or installing a listed auxiliary notification appliance, and that holds the dead-zone flag until a subsequent doors-closed run scores a pass.

6. The system of claim 1, further comprising a longitudinal degradation module that stores each alarm's reference-anchored peak received level from successive tests, fits a per-alarm decay trend, and projects an audibility-based end-of-life date at which the alarm will fall below threshold in the farthest room it serves, independent of the calendar replacement date printed on the unit.

7. The system of claim 6, wherein the longitudinal degradation module distinguishes sounder decay from acoustic path changes by testing whether all nodes' relative levels shifted together, indicating a path change that triggers a re-baseline prompt, or only the alarm's anchored level moved, indicating sounder decay.

8. The system of claim 1, further comprising an interconnect verification module that, when a single alarm is triggered, listens for the expected acoustic fingerprint of every interconnected unit and flags any unit whose fingerprint never appears as an interconnect failure distinct from a coverage failure.

9. The system of claim 8, wherein per-alarm identity during simultaneous sounding is resolved by per-unit piezoelectric resonant-frequency fingerprints, each sounder's resonant peak differing stably from unit to unit, allowing overlapping temporal patterns to be separated spectrally.

10. The system of claim 1, wherein all pattern detection and level estimation execute on the microphone node within a rolling audio buffer of at most 60 seconds that is discarded after feature extraction, and wherein the node transmits to the hub only timestamped feature vectors comprising a pattern identifier, a band-limited received level, the ambient floor, and the node identifier, with no raw audio leaving the node.

11. A method for verifying life-safety alarm audibility coverage in a residence, comprising: registering a plurality of microphone nodes with room assignments marking sleeping areas; measuring a per-node ambient sound floor during a pre-roll interval; triggering each installed smoke or carbon monoxide alarm's test signal in a guided sequence isolating one alarm at a time, with bedroom doors closed; detecting standardized T3 and T4 temporal alarm patterns via template correlation in the piezoelectric sounder band; estimating per-node received levels relative to a reference node nearest the sounding alarm; issuing per-room audibility verdicts against the greater of 75 dBA and ambient plus 15 dB for sleeping areas; flagging sub-threshold sleeping areas as dead zones with remediation guidance; tracking per-alarm sounder decay across successive tests to project an audibility-based end of life; and verifying interconnect propagation by confirming each interconnected unit's acoustic fingerprint when a single alarm is triggered.

## Implementation Notes

Run tests during the day and warn the household first. A full T3 cadence at close range exceeds 85 dB by design; it will startle people and terrify pets. The guided workflow should announce which alarm is about to sound and offer a countdown. This is a solved social problem, not a technical one, but skipping it guarantees the system gets used exactly once.

Sequential guided testing is load-bearing, not a convenience. In an interconnected home, pressing one test button fires every sounder, and per-alarm attribution from a single simultaneous capture depends on the resonant-frequency fingerprint fallback, which needs clean per-unit templates. The app should therefore default to walking the homeowner alarm by alarm, using the hush or test control to isolate units where the wiring allows it, and only fall back to simultaneous capture with spectral separation where isolation is impossible.

Keep phones out of pockets and off soft furniture during scoring runs. A phone face-down on a couch cushion can read 10 dB low for reasons that have nothing to do with the alarm, and the system will happily flag a dead zone that is really a couch. The companion app should require stationary placement, face-up on a hard surface, or exclude phones from scoring and use them as corroboration only. Smart speakers and dedicated nodes, which do not move, are the trustworthy witnesses.

Measure ambient in the state the house actually sleeps in. HVAC cycling changes the ambient floor by several dB, and the ambient-plus-15 dB criterion is only honest if the pre-roll reflects sleeping conditions. Schedule tests with the HVAC in its normal nighttime mode, note the mode with the run, and do not compare a summer cooling-cycle run against a winter idle run without saying so. The longitudinal trend module should condition on HVAC state or it will read seasonal ambient drift as sounder decay.

Test battery-powered alarms at the battery state they live in. A test performed minutes after a battery change flatters the sounder; the number that matters is the output on a mid-life battery at 3 a.m. in February. The app should record battery age or measured terminal voltage with each run for battery-only units, and the decay trend should treat a battery replacement as a step to be modeled, not as sounder rejuvenation.

Say plainly what the system does not verify. This audits the notification path: the sounder, the acoustic path, and the audibility at the pillow. It does not verify the detection path: whether smoke can reach the sensing chamber, whether the chamber is contaminated, whether the sensor has drifted. Pair the coverage audit with the alarm's own sensor self-test and with physical smoke-entry testing per the manufacturer's guidance. A PASS on audibility with a dead sensing chamber is a house that will burn quietly, and the documentation must not let anyone confuse the two.

Renters get a document, not just a map. A coverage map with FAIL marks on two bedrooms is exactly the artifact a tenant needs when asking a landlord to add interconnected alarms, and exactly the artifact a landlord wants before spending the money. Export a one-page PDF per test run: the map, the verdicts, the ambient conditions, and the remediation list. Date it. Both sides keep a copy.

What this system buys is the replacement of an assumption with a measurement. Every installed alarm carries an implicit claim: the people sleeping down the hall will hear this. That claim is made once, at installation, and never tested again, while doors get replaced, furniture moves, sounders age, and children grow into the far bedroom. A twice-yearly doors-closed test that takes ten minutes turns the claim into a number, and the number either confirms the assumption or names the room where it fails. The cost of the measurement is a phone app and microphones the house already owns; the cost of the assumption is measured in the fire reports.

## Prior Art References

- [Apple HomePods can now detect the sound of smoke alarms (TechHive)](https://www.techhive.com/article/1790156/apple-homepods-can-now-detect-the-sound-of-smoke-alarms.html): Smart speaker sound recognition detecting that a smoke or CO alarm is currently sounding, for remote notification; no per-room audibility measurement or coverage mapping

- [Sensory TrulySecure Sound ID](https://mobileidworld.com/sensory-always-listening-tech-home-security-801073/): On-device environmental sound identification (doorbell, smoke alarm) for smart speakers; event detection, not audibility auditing

- [SonicAlert HomeAware II](https://www.sonicalert.com/The-HomeAware-Fire-CO-Alert-Internal-Smoke-CO-Listener-Package_2?gad_source=5&gclid=EAIaIQobChMI9a3o19mZigMVIGFHAR3gWgeYEAAYASAAEgJY3fD_BwE): Smoke/CO listener relaying alarm events to notification systems; requires placement where it can hear the alarm, but performs no coverage audit

- [EP 4358049 A1](https://patentimages.storage.googleapis.com/94/aa/98/9738e640bc0b50/EP4358049A1.pdf) (Bosch): Smoke detector with microphone for predictive maintenance of the detector itself via ambient sound sampling

- [WO2015035187A1](https://patents.google.com/patent/WO2015035187A1/en): Systems and methods for monitoring sound during an in-building emergency for evacuation guidance; emergency-time centralized monitoring, not consumer coverage auditing

- [Smoke alarm decibel and sound requirements (UL 217, EN 14604, NFPA 72 survey)](https://www.linkedin.com/pulse/smoke-alarm-decibel-sound-requirements-what-ul-217-en-tim-sh-td2sc): Rated output requirements, measurement procedure, and factors eroding audibility including closed doors, ceiling height, room materials, and background noise

- NFPA 72, National Fire Alarm and Signaling Code: Audible notification requirements for sleeping areas (75 dBA at the pillow, 15 dB above average ambient)

- UL 217, Standard for Smoke Alarms: Minimum 85 dB sound output at 10 feet

- ISO 8201: Audible emergency evacuation signal temporal patterns (T3) and four-pulse CO alarm pattern (T4)

- [35 U.S.C. § 102](https://www.law.cornell.edu/uscode/text/35/102): Conditions for patentability; novelty and prior art
