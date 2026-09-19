# PA-2026-176: Pre-Ignition Detection of Vegetation-Contact Faults on Overhead Distribution Lines Using Pole-Mounted RF Partial Discharge Sensing and Edge Classification

**Title:** System and Method for Pre-Ignition Detection of Vegetation-Contact Faults on Overhead Distribution Lines Using Pole-Mounted Radio-Frequency Partial Discharge Sensing and Edge Classification

**Filing:** LITF-PA-2026-176
**Published:** September 19, 2026
**Domain:** Smart Grid / Wildfire Prevention
**Full Disclosure:** [liveinthefuture.org/priorart/vegetation-contact-preignition-rf-detection.html](https://liveinthefuture.org/priorart/vegetation-contact-preignition-rf-detection.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published as defensive prior art under
> [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> The inventions described herein are dedicated to the public domain as of the
> publication date above.

---

## Abstract

Disclosed is a system and method for detecting, before ignition, faults in which vegetation contacts an energized overhead distribution conductor. A network of radio-frequency (RF) sensor nodes is mounted on distribution poles. Each node passively listens in the 30 MHz to 1 GHz band for the broadband electromagnetic emissions produced by partial discharge at the vegetation-conductor contact point, where electric field distortion at the contact site exceeds the local breakdown threshold and produces intermittent discharge. An edge classifier running on each node distinguishes the vegetation-contact partial discharge signature from corona, insulator contamination discharge, switching transients, lightning, and load noise, using spectral-temporal features including 120 Hz arc re-ignition modulation synchronized to the power line cycle, pulse rise time, spectral centroid, and bandwidth. Arrival timestamps from adjacent nodes, disciplined to a common GPS time base, feed a time-difference-of-arrival localizer that places the contact within a span between poles. A risk fusion module combines the detection with fire weather index, fuel moisture, wind speed, vegetation clearance records, and feeder loading to produce a pre-ignition risk score. When the score crosses threshold, the system issues a protective action (fast trip or recloser block on the affected lateral) within seconds of contact onset and dispatches a vegetation-management crew to the localized span. The design targets the detection gap in which conventional overcurrent protection cannot see high-impedance vegetation faults drawing 1 to 100 A, below normal load current.

## Technical Field

This invention relates to power distribution protection and wildfire prevention, specifically to passive radio-frequency sensing of partial discharge for pre-ignition detection and localization of vegetation-contact faults on overhead distribution lines, with edge classification and risk-fused protective action.

## Background

Vegetation contacting energized conductors is one of the leading ignition sources of catastrophic wildfires. CAL FIRE determined that the November 8, 2018 Camp Fire, the deadliest wildfire in California history with 85 fatalities and nearly 19,000 structures destroyed, was sparked when PG&E transmission lines came into contact with dry vegetation at two separate locations (Weather Channel). PG&E equipment also started the 2021 Dixie Fire, which burned 963,405 acres, the largest single wildfire in California history (californiatoday.com). An analysis of 25 years of California fire perimeter data found power line equipment ignited at least 419 wildfires since 2000, burning 1.49 million acres across 51 of 58 counties.

The electrical signature of a vegetation-contact fault is a high-impedance fault. Downed conductors and vegetation contacts draw fault currents from near zero to under 100 A depending on the contact surface, often below normal feeder load, so conventional overcurrent protection cannot see them (Schweitzer Engineering Laboratories). Research on staged high-impedance fault tests found that conventional overcurrent protection misses 30 to 50 percent of downed conductor events on distribution feeders, the fault type most likely to ignite wildfires (Emanuel et al., IEEE Transactions on Power Delivery 2014).

Researchers studying tree-conductor faults treat them as high-impedance grounding faults and have shown that fault signatures are most detectable during arc re-ignition phases (Elkalashy et al., via AIP Advances), and that activating protection within 5 seconds of vegetation-conductor contact can prevent most fires initiated by such faults (Ozansoy et al., via AIP Advances). Independent electromagnetic modeling of vegetation contacting insulated medium-voltage conductors shows electric field concentration at the contact point far exceeding the partial discharge inception threshold, confirming that tree contact produces partial discharge through electric field distortion (MDPI Energies).

Existing approaches leave the pre-ignition detection gap open. Pole-mounted line sensors from vendors such as Gridware monitor vibration, temperature, and acoustic emissions per span, but cost $500 to $2,000 per device plus cellular backhaul, pricing out full coverage for utilities with hundreds of thousands of poles (Gridware, via prior disclosure PA-2026-034). Smart-meter "last gasp" messages arrive 2 to 15 minutes after the fault, well beyond the window in which a sustained arc can ignite vegetation. Satellite and drone thermal inspection finds degraded equipment but cannot detect real-time fault events. Substation-based high-impedance fault detection analyzes current waveforms from the feeder head, but vegetation-contact partial discharge radiates broadband RF energy that is far more detectable at the pole than as current at the substation.

The gap in the art is a pole-mounted, low-cost, passive RF sensing system that: (a) detects partial discharge specifically from vegetation-conductor contact in the 30 MHz to 1 GHz band, (b) classifies the discharge at the edge to reject corona, insulator contamination, and switching transients, (c) localizes the contact to a span via time-difference-of-arrival across adjacent poles, and (d) converts the detection into a protective action within the seconds-scale window before ignition.

## Detailed Description

### 1. Sensor Node Hardware

Each pole-mounted node comprises: a broadband passive antenna (log-periodic or discone, 30 MHz to 1 GHz) mounted 1 to 2 meters below the lowest phase conductor on the pole; an RF front end with low-noise amplifier, tunable bandpass preselection, and a software-defined radio receiver sampling at 2.4 MS/s or higher; a microcontroller or system-on-chip with edge inference capability (e.g., ARM Cortex-A class, unit cost under $15); a GPS receiver providing a pulse-per-second disciplined timestamp with better than 20 ns accuracy for time-difference-of-arrival localization; a solar cell (2 to 5 W) with rechargeable battery for autonomous operation; and a sub-GHz mesh radio (LoRa or equivalent) or cellular LTE-M modem for backhaul. Target bill-of-materials cost per node: under $120. Nodes mount to the pole with a banded bracket requiring no outage and no conductor contact, so installation does not de-energize the feeder.

### 2. Partial Discharge Signal Acquisition

Vegetation contacting an energized conductor distorts the local electric field at the contact site, and the distorted field exceeds the partial discharge inception threshold, producing intermittent broadband RF emission pulses. The node continuously digitizes the RF band and runs a streaming pulse detector: samples are compared against an adaptive noise floor estimated over a sliding 10-second window, and excursions exceeding 12 dB above the floor with rise times under 100 ns are captured as pulse events with 1 microsecond of pre-trigger and 50 microseconds of post-trigger data. A per-pulse record contains: GPS timestamp (nanosecond resolution), peak amplitude, rise time, pulse width, spectral magnitude in 64 sub-bands across 30 MHz to 1 GHz, and the phase angle of the 60 Hz power line cycle at pulse onset (recovered from a narrowband 60 Hz reference channel or from mains-synchronized clock).

A slow-path spectral monitor computes a 1-second averaged power spectral density every 10 seconds to track background evolution (new transmitters, seasonal corona changes) and to update the adaptive noise floor and per-band exclusion masks for known interferers (FM broadcast, cellular, public-safety bands).

### 3. Edge Classification of Vegetation-Contact Partial Discharge

A gradient-boosted decision tree classifier (quantized, under 200 KB model size) running on the node scores each pulse event against five classes: (i) vegetation-contact partial discharge, (ii) corona discharge, (iii) insulator contamination surface discharge, (iv) switching transients and capacitor bank operations, and (v) lightning and other impulsive noise. Handcrafted features include: 120 Hz modulation index of the pulse envelope over a 1-second window (vegetation-contact arcing re-ignites each half cycle, producing strong 120 Hz modulation synchronized to the power line cycle, while corona shows weaker and less phase-locked modulation); spectral centroid and occupied bandwidth (vegetation-contact PD exhibits broadband energy extending above 300 MHz from short rise-time pulses, while insulator contamination PD concentrates below 150 MHz); pulse rise time and decay constant; pulse repetition rate and its phase-of-cycle histogram (vegetation-contact PD clusters near voltage peaks with polarity asymmetry tied to the contact geometry); and burst persistence (vegetation contact produces sustained bursts over seconds to minutes as the branch sways, whereas switching transients are isolated).

A detection is declared only when at least 20 vegetation-class pulses occur within a rolling 60-second window with median per-pulse confidence above 0.75, suppressing isolated false triggers. The classifier is trained on labeled data from staged fault tests on de-energized-then-re-energized test spans with controlled vegetation contact, corona measurements on clean conductors, and recorded utility switching events, with per-utility transfer learning to adapt to local interference environments.

### 4. Time-Difference-of-Arrival Localization

RF pulses propagate at the speed of light, so a discharge event at a contact point reaches adjacent pole nodes with arrival-time differences proportional to the difference in path lengths. With adjacent poles spaced 40 to 80 meters on typical distribution laterals, arrival-time differences span 130 to 270 ns. The GPS-disciplined timestamps (20 ns accuracy) resolve these differences. When two or more nodes report vegetation-class pulses within a 5 microsecond correlation window, a hyperbolic multilateration solver estimates the discharge position along the line route. With three reporting nodes, localization accuracy is approximately plus or minus 10 meters, sufficient to identify the span and the nearest pole. The localizer fuses the RF arrival times with the utility's GIS pole-location database so the output is a specific span identifier and pole tag, not just coordinates.

### 5. Pre-Ignition Risk Fusion

A risk fusion module converts detections into a pre-ignition risk score from 0 to 100. Inputs: detection confidence and burst persistence from the classifier; National Weather Service fire weather inputs including temperature, relative humidity, wind speed and gusts, and 10-hour fuel moisture; vegetation clearance status of the localized span from the utility's LiDAR clearance inspection database (spans overdue for trimming or with recorded encroachment score higher); feeder loading at the time of detection (higher loading drives higher fault current and faster heating at the contact); and time since last rainfall. The score crosses the protective-action threshold when sustained vegetation-class discharge coincides with fire weather conditions, and crosses a lower advisory threshold at any detection, generating a crew dispatch ticket with the span identifier even in benign weather.

### 6. Protective Action Within the Ignition Window

When the risk score crosses the protective-action threshold, the system issues a control command to the substation or feeder automation controller within 5 seconds of detection onset: trip the affected lateral, or block automatic reclosing (single-shot-to-lockout) so a protection operation does not re-energize into the fault and produce a second arcing episode. Research shows protection action within 5 seconds of vegetation-conductor contact prevents most resulting fires. Simultaneously, the system dispatches an alert to the utility's outage management system and to the vegetation-management crew queue with the localized span, the detection confidence, and the burst history, so a crew can confirm clearance or trim the contact before re-energization. Re-energization requires either crew confirmation of clearance or a 30-minute quiet period with no further vegetation-class pulses.

### 7. Mesh Communication and Data Aggregation

Nodes communicate detections and periodic health beacons over a sub-GHz mesh using a time-slotted protocol. A detection packet carries: node ID, GPS timestamp of the pulse burst, per-class confidence vector, pulse count and median features, and battery and signal-health telemetry, in under 120 bytes. A head-end server aggregates multi-node correlations, runs the TDOA localizer and risk fusion, and exposes a REST API serving detection events, span-level risk scores, and crew dispatch tickets. Raw pulse records are retained on-node in a rolling 24-hour buffer and uploaded on demand for forensic analysis of ignitions that occur despite the system.

### 8. Figures Description

- **Figure 1:** Pole-mounted RF sensor node installation showing the broadband antenna below the crossarm, solar panel, GPS antenna, and mesh radio, with a vegetation branch contacting a phase conductor two spans away.
- **Figure 2:** Example captured RF pulse waveforms and spectrograms for vegetation-contact partial discharge versus corona versus insulator contamination discharge, annotated with 120 Hz re-ignition modulation envelopes.
- **Figure 3:** Time-difference-of-arrival localization geometry for three adjacent pole nodes, showing hyperbolic position lines intersecting at the vegetation contact point on a span.
- **Figure 4:** System data flow from pulse detection through edge classification, multi-node correlation, risk fusion with weather and clearance inputs, to protective action and crew dispatch.

## Claims

1. A system for pre-ignition detection of vegetation-contact faults on overhead distribution lines, comprising: a plurality of radio-frequency sensor nodes mounted on distribution poles, each node containing a broadband passive antenna covering 30 MHz to 1 GHz, a software-defined radio receiver, a GPS-disciplined timestamping circuit, and a processor with edge inference capability; wherein each node passively captures broadband electromagnetic pulses produced by partial discharge at a vegetation-conductor contact site and classifies said pulses at the edge as vegetation-contact partial discharge.
2. The system of claim 1, wherein the edge classifier distinguishes vegetation-contact partial discharge from corona discharge, insulator contamination surface discharge, switching transients, and lightning, using features including 120 Hz arc re-ignition modulation synchronized to the power line cycle, spectral centroid, occupied bandwidth, pulse rise time, phase-of-cycle pulse histogram, and burst persistence.
3. The system of claim 1, further comprising a time-difference-of-arrival localizer that correlates vegetation-class pulse arrival timestamps from at least two adjacent nodes against a GPS common time base and solves hyperbolic position lines against a utility pole-location database to identify the conductor span containing the vegetation contact.
4. The system of claim 1, wherein a detection is declared only when at least a threshold number of vegetation-class pulses occur within a rolling time window with median per-pulse confidence above a configurable threshold, suppressing isolated false triggers.
5. The system of claim 1, further comprising a pre-ignition risk fusion module that computes a risk score from detection confidence and burst persistence combined with fire weather data, fuel moisture, wind speed, vegetation clearance inspection records for the localized span, feeder loading, and time since last rainfall.
6. The system of claim 5, wherein the risk score crossing a protective-action threshold causes issuance of a control command to trip the affected lateral or block automatic reclosing within 5 seconds of detection onset, and dispatches a crew alert identifying the localized span.
7. The system of claim 6, wherein re-energization of the tripped lateral requires crew confirmation of vegetation clearance or a configurable quiet period with no further vegetation-class pulses.
8. The system of claim 1, wherein each node is powered by a solar cell with battery backup, communicates via a sub-GHz mesh network, and mounts to the pole with a banded bracket requiring no de-energization of the feeder for installation.
9. A method for pre-ignition detection of vegetation-contact faults on overhead distribution lines, comprising: passively sensing broadband radio-frequency emissions in the 30 MHz to 1 GHz band from a plurality of pole-mounted nodes; capturing pulse events exceeding an adaptive noise floor; classifying said pulses at the edge as vegetation-contact partial discharge based on 120 Hz re-ignition modulation and spectral-temporal features; localizing the contact to a conductor span via time-difference-of-arrival across adjacent nodes; fusing the detection with fire weather and vegetation clearance data into a pre-ignition risk score; and issuing a protective action within 5 seconds of detection onset when the score crosses threshold.
10. The method of claim 9, further comprising retaining raw pulse records in a rolling on-node buffer and uploading said records on demand for forensic analysis of ignition events.

## Implementation Notes

Deployment economics drive the design: a bill-of-materials under $120 per node makes per-pole coverage on high-fire-risk laterals affordable at roughly one-tenth the cost of per-span dedicated line sensors. A utility prioritizing its highest-risk 20,000 poles (those in high fire threat districts with recorded clearance encroachments) covers the ignition-critical footprint for under $2.4 million in hardware, versus $10 to $40 million for per-pole dedicated sensors. Nodes are passive listeners and emit no probing energy; the RF band monitored is receive-only, so no spectrum license is required for the sensing function.

Known limitations are disclosed. The system cannot detect vegetation contact on de-energized conductors, since partial discharge requires line voltage; it is a complement to, not a replacement for, LiDAR clearance inspection. Heavy precipitation raises the broadband noise floor and can mask weak discharge; the risk fusion module derates detection confidence during active rainfall, which also lowers ignition risk. Dense urban RF environments require per-deployment interference training, provided by the transfer-learning calibration step. The 5-second protective action target applies to the detection-to-command path; actual breaker operation time adds feeder-specific latency, and utilities must validate the full chain against their protection coordination studies before enabling automatic tripping, with advisory-only operation as the conservative initial deployment mode.

## Prior Art References

1. Weather Channel: CAL FIRE: Camp Fire sparked by PG&E transmission lines contacting dry vegetation at two locations
2. californiatoday.com: 419 power-line wildfires since 2000, 1.49 million acres burned in California
3. Schweitzer Engineering Laboratories: High-impedance faults from vegetation contact draw 1 to 100 A, below load current; conventional protection cannot detect downed-conductor faults on dry surfaces
4. AIP Advances: Tree-conductor faults as high-impedance faults; signatures most detectable during arc re-ignition (Elkalashy et al.); protection within 5 s of contact prevents most fires (Ozansoy et al.)
5. MDPI Energies: Electric field distortion at vegetation contact points on insulated conductors exceeds partial discharge inception threshold
6. Frontiers in Energy Research: Hilbert transform methods for high-impedance fault detection in distribution systems
7. Western Protective Relay Conference tutorial: High-impedance fault detection misconceptions; intermittent tree-limb contact arcing
8. PA-2026-034 (prior disclosure): Municipal acoustic sensor networks for electrical fault detection; Gridware pole sensors at $500 to $2,000 per device; smart-meter last-gasp latency 2 to 15 minutes
